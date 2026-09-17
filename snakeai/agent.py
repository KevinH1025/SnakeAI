"""The learner: the Q-network, the replay buffer and the learning step.

The replay buffer lives on the same device as the network, so sampling never copies to the host.
Actions are always chosen for a whole batch of games at once. See docs/BENCHMARKS.md for why.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .config import AgentConfig
from .env import I_DANGER_LEFT, I_DANGER_RIGHT, I_DANGER_STRAIGHT, N_ACTIONS, OBS_DIM

# where the three danger flags sit in the observation, in action order
DANGER_SLOTS = (I_DANGER_STRAIGHT, I_DANGER_LEFT, I_DANGER_RIGHT)


# --------------------------------------------------------------------------- the network


class QNetwork(nn.Module):
    """A plain stack of Linear + ReLU layers.

    In goes an observation, out come three numbers: how good each action looks.
    There are deliberately no BatchNorm or Dropout layers here, so train() and eval() compute
    the same function and no stray mode switch can change what the network does.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS,
                 hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last = obs_dim

        for width in hidden:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width

        layers.append(nn.Linear(last, n_actions)) # one output per action

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- the learning target


def double_dqn_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_q_online: torch.Tensor,
    next_q_target: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """What the Q-value for the action we took should have been.

        target = reward + gamma * (value of the next state) * (0 if we died, else 1)

    The online network picks which action is best next; the target network says what it is worth.
    Note this takes Q values, not states or networks, so it cannot evaluate the wrong state.
    """
    best = next_q_online.argmax(dim=1, keepdim=True) # online net chooses
    next_value = next_q_target.gather(1, best).squeeze(1) # target net prices that choice

    return rewards + gamma * next_value * (1.0 - terminated)


def epsilon_at(step: int, cfg: AgentConfig) -> float:
    """How often to move randomly at this point in training. Falls in a straight line."""
    if cfg.epsilon_decay_steps <= 0:
        return cfg.epsilon_final

    progress = step / cfg.epsilon_decay_steps
    progress = min(max(progress, 0.0), 1.0) # clamp to [0, 1]

    return cfg.epsilon_start + progress * (cfg.epsilon_final - cfg.epsilon_start)


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class Batch:
    """A pile of past moves to learn from."""

    obs: torch.Tensor # the state before
    actions: torch.Tensor # what we did
    rewards: torch.Tensor # what we got
    next_obs: torch.Tensor # the state after
    terminated: torch.Tensor # 1.0 if the game ended there


class ReplayBuffer:
    """Stores past moves in fixed arrays, overwriting the oldest once it is full."""

    def __init__(self, capacity: int, device: torch.device, obs_dim: int = OBS_DIM,
                 seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.device = device

        # Allocated once, up front and reused forever.
        self.obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(self.capacity, dtype=torch.int64, device=device)
        self.rewards = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.terminated = torch.zeros(self.capacity, dtype=torch.float32, device=device)

        self.pos = 0 # where the next row goes
        self.size = 0 # how many rows are filled in

        self.generator = torch.Generator(device=device) # sampling happens on the device
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    @property
    def nbytes(self) -> int:
        total = 0
        for tensor in (self.obs, self.next_obs, self.actions, self.rewards, self.terminated):
            total += tensor.numel() * tensor.element_size()

        return total

    def add_batch(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                  next_obs: torch.Tensor, terminated: torch.Tensor) -> None:
        """Add one row per parallel game, wrapping around the end of the arrays."""
        n = obs.shape[0]

        if n > self.capacity:
            raise ValueError(f"cannot insert {n} transitions into a buffer of capacity {self.capacity}")

        idx = (self.pos + torch.arange(n, device=self.device)) % self.capacity

        self.obs[idx] = obs
        self.next_obs[idx] = next_obs # stays on the same row as its obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.terminated[idx] = terminated

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity) # stops growing once full

    def sample(self, batch_size: int) -> Batch:
        """Pick `batch_size` random rows."""
        if self.size < batch_size:
            raise ValueError(
                f"sample({batch_size}) from a buffer holding {self.size}; "
                "agent.learning_starts must be >= agent.batch_size"
            )

        idx = torch.randint(0, self.size, (batch_size,), device=self.device,
                            generator=self.generator)

        return Batch(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            terminated=self.terminated[idx],
        )

    def state_dict(self) -> dict:
        """The filled rows, moved to the CPU so they can be saved."""
        return {
            "obs": self.obs[: self.size].cpu(),
            "next_obs": self.next_obs[: self.size].cpu(),
            "actions": self.actions[: self.size].cpu(),
            "rewards": self.rewards[: self.size].cpu(),
            "terminated": self.terminated[: self.size].cpu(),
            "pos": self.pos,
            "size": self.size,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state: dict) -> None:
        """Put saved rows back at the front of the arrays.

        `pos` is recomputed rather than trusted, because the saved rows land as a block at the
        front and the old write position no longer points where it used to.
        """
        n = int(state["size"])

        if n > self.capacity:
            raise ValueError(
                f"saved buffer holds {n} transitions but agent.buffer_capacity is {self.capacity}; "
                "raise buffer_capacity or resume with train.save_buffer=false"
            )

        for name in ("obs", "next_obs", "actions", "rewards", "terminated"):
            getattr(self, name)[:n] = state[name].to(self.device)

        self.size = n

        if n == self.capacity:
            self.pos = int(state["pos"]) % self.capacity # full, so the old cursor still means something
        else:
            self.pos = n % self.capacity # carry on after the restored rows


# --------------------------------------------------------------------------- the agent


class DQNAgent:
    """Holds the networks, the optimizer and the replay buffer and runs the learning step."""

    def __init__(self, cfg: AgentConfig, device: torch.device, seed: int = 0,
                 obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS) -> None:
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions

        # Build, move to the device and only then make the optimizer, so the optimizer is
        # pointed at parameters that already live where they will stay.
        self.online = QNetwork(obs_dim, n_actions, tuple(cfg.hidden)).to(device)

        self.target = copy.deepcopy(self.online).to(device) # a frozen copy, updated occasionally
        self.target.requires_grad_(False)
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer(cfg.buffer_capacity, device, obs_dim, seed)
        self.rng = np.random.default_rng(seed) # used for random exploration moves

        self.updates = 0 # how many gradient steps we have taken

        # The target copy happens every so many gradient SAMPLES, not every so many updates,
        # because train.py makes update count depend on num_envs.
        self.samples = 0
        self._samples_at_last_sync = 0
        self._sync_every_samples = cfg.target_sync_steps * cfg.batch_size

    # -- choosing moves -----------------------------------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> np.ndarray:
        """Pick an action for each game. Takes (n_envs, obs_dim), returns (n_envs,).

        Mostly picks whatever the network rates highest, but with probability `epsilon` it moves
        at random instead, so the agent keeps discovering things.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0) # accept a single observation too

        n = obs_t.shape[0]
        actions = self.online(obs_t).argmax(dim=1).cpu().numpy() # best action per game

        explore = self.rng.random(n) < epsilon # which games move randomly this step
        if explore.any():
            random_actions = self._random_actions(obs_t.shape[0], obs)
            actions = np.where(explore, random_actions, actions)

        return actions.astype(np.int64)

    def _random_actions(self, n: int, obs: np.ndarray) -> np.ndarray:
        """Pick a random action per game, optionally skipping the ones that kill instantly."""
        if not self.cfg.mask_fatal_exploration:
            return self.rng.integers(0, self.n_actions, size=n) # plain uniform over 0, 1, 2

        flat = obs.reshape(n, -1)
        deadly = flat[:, DANGER_SLOTS] == 1.0 # (n, 3): True where that move kills us

        # Give every action a random score, then knock the deadly ones out of the running by
        # scoring them below anything else. argmax then picks uniformly among what is left.
        scores = self.rng.random((n, self.n_actions))
        scores[deadly] = -1.0

        # If every move is deadly there is nothing to choose between, so leave those rows alone
        # and let the random scores decide.
        doomed = deadly.all(axis=1)
        if doomed.any():
            scores[doomed] = self.rng.random((int(doomed.sum()), self.n_actions))

        return scores.argmax(axis=1)

    # -- learning -----------------------------------------------------------

    def learn(self, batch_size: int | None = None) -> float | None:
        """One gradient step: take a pile of past moves and nudge the weights.

        The idea in one line: for each past move, compare what the network SAID that move was
        worth against what it TURNED OUT to be worth and shrink the gap.

        Returns the loss or None if there is not enough experience collected yet.
        """
        if batch_size is None:
            batch_size = self.cfg.batch_size

        # Nothing to learn from yet. Keep collecting.
        if len(self.buffer) < max(batch_size, self.cfg.learning_starts):
            return None

        # STEP 1: grab a random pile of past moves. Random on purpose: consecutive moves look
        # almost identical and training on them in order would teach the same thing over and over.
        batch = self.buffer.sample(batch_size)

        # STEP 2: what the network SAYS. It outputs a value for all three actions, but we only
        # care about the one that was actually taken, so gather() picks that column out per row.
        all_q = self.online(batch.obs) # (batch, 3)
        predicted = all_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1) # (batch,)

        # STEP 3: what it SHOULD have said. This is the reward we really got, plus what we think
        # the state we landed in is worth. no_grad because this is the thing we are aiming AT -
        # we do not want to train the target to meet the prediction halfway.
        with torch.no_grad():
            next_q_online = self.online(batch.next_obs) # picks WHICH next action looks best
            next_q_target = self.target(batch.next_obs) # says what that action is WORTH
            targets = double_dqn_target(
                batch.rewards, batch.terminated, next_q_online, next_q_target, self.cfg.gamma
            )

        # STEP 4: the gap between the two. Huber (smooth L1) rather than plain squared error,
        # because a few wildly wrong targets should not yank the weights around.
        loss = nn.functional.smooth_l1_loss(predicted, targets)

        # STEP 5: backpropagate and take the step.
        self.optimizer.zero_grad(set_to_none=True) # clear last step's gradients
        loss.backward() # work out which weights caused the gap
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip) # keep steps bounded
        self.optimizer.step() # THIS is the line where the weights actually change

        self.updates += 1
        self.samples += batch_size

        if self.samples - self._samples_at_last_sync >= self._sync_every_samples:
            self.sync_target()
            self._samples_at_last_sync = self.samples

        return float(loss.detach())

    def sync_target(self) -> None:
        """Copy the online weights into the frozen target network."""
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False) # load_state_dict does not restore these
        self.target.eval()

    # -- saving and loading -------------------------------------------------

    def state_dict(self, include_buffer: bool = False) -> dict:
        state = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), # needed for an exact resume
            "updates": self.updates,
            "samples": self.samples,
            "samples_at_last_sync": self._samples_at_last_sync,
            "rng": self.rng.bit_generator.state,
        }

        if include_buffer:
            state["buffer"] = self.buffer.state_dict()

        return state

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])

        self.target.load_state_dict(state["target"])
        self.target.requires_grad_(False)
        self.target.eval()

        self.optimizer.load_state_dict(state["optimizer"])

        self.updates = int(state.get("updates", 0))
        self.samples = int(state.get("samples", 0))
        self._samples_at_last_sync = int(state.get("samples_at_last_sync", self.samples))

        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]

        if "buffer" in state:
            self.buffer.load_state_dict(state["buffer"])
