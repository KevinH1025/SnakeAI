"""Learner tests. The first three make the two v1 critical bugs structurally unwritable."""

import dataclasses
import inspect
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from snakeai.agent import (
    Batch, DQNAgent, QNetwork, ReplayBuffer, double_dqn_target, epsilon_at,
)
from snakeai.env import OBS_DIM


# -- the tripwires ----------------------------------------------------------------------


def test_double_dqn_target_signature_excludes_state():
    """v1 wrote `self.target_model(states)` where `next_states` was meant.

    This function takes pre-computed Q values, so there is no state to pass and no network to call.
    If someone ever adds an obs or model parameter, this fails, which is the point.
    """
    params = list(inspect.signature(double_dqn_target).parameters)
    assert params == ["rewards", "terminated", "next_q_online", "next_q_target", "gamma"]
    banned = ("obs", "state", "net", "model", "env", "batch")
    assert not [p for p in params if any(b in p.lower() for b in banned)]


def test_target_network_only_ever_sees_next_obs(cfg, device):
    """A forward-pre-hook spy: the target net must be called exactly once, on next_obs."""
    agent = DQNAgent(cfg.agent, device, seed=0)
    n = 512
    g = torch.Generator(device=device).manual_seed(0)
    obs = torch.randn(n, OBS_DIM, generator=g)
    next_obs = torch.randn(n, OBS_DIM, generator=g)
    agent.buffer.add_batch(obs, torch.randint(0, 3, (n,), generator=g),
                           torch.randn(n, generator=g), next_obs, torch.zeros(n))

    seen = []
    handle = agent.target.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].detach().clone()))
    agent.buffer.generator.manual_seed(123)
    agent.learn()
    handle.remove()

    assert len(seen) == 1, f"target net called {len(seen)} times, expected exactly 1"
    # Reproduce the sampled batch and confirm the tensor the target saw was next_obs, not obs.
    agent.buffer.generator.manual_seed(123)
    batch = agent.buffer.sample(cfg.agent.batch_size)
    assert torch.equal(seen[0], batch.next_obs)
    assert not torch.equal(seen[0], batch.obs)


def test_double_dqn_uses_online_argmax_and_target_value():
    """Online net SELECTS, target net EVALUATES. Vanilla DQN would take the target's own max."""
    rewards = torch.tensor([1.0])
    terminated = torch.tensor([0.0])
    next_q_online = torch.tensor([[0.0, 0.0, 5.0]])   # argmax = 2
    next_q_target = torch.tensor([[9.0, 0.0, 7.0]])   # argmax = 0, value at 2 = 7
    got = double_dqn_target(rewards, terminated, next_q_online, next_q_target, 0.99).item()
    assert got == pytest.approx(1 + 0.99 * 7)
    assert got != pytest.approx(1 + 0.99 * 9), "took the target's max - that is vanilla DQN"


def test_terminated_does_not_bootstrap_and_truncated_does():
    """Truncation keeps its successor's value. Zeroing it teaches that time limits are fatal."""
    rewards = torch.tensor([1.0, 1.0])
    next_q_online = torch.tensor([[0.0, 0.0, 5.0]] * 2)
    next_q_target = torch.tensor([[9.0, 0.0, 7.0]] * 2)
    out = double_dqn_target(rewards, torch.tensor([1.0, 0.0]), next_q_online, next_q_target, 0.99)
    assert out[0].item() == pytest.approx(1.0), "terminated must not bootstrap"
    assert out[1].item() == pytest.approx(1 + 0.99 * 7), "non-terminated must bootstrap"


def test_batch_has_no_truncated_field():
    """A field that must never enter the target is a field that eventually enters the target."""
    names = [f.name for f in dataclasses.fields(Batch)]
    assert names == ["obs", "actions", "rewards", "next_obs", "terminated"]
    assert "truncated" not in names


# -- the BatchNorm family of bugs -------------------------------------------------------


def test_network_has_no_mode_dependent_layers():
    """v1's BatchNorm made train()/eval() change the computation and one stray eval() stuck."""
    net = QNetwork(OBS_DIM, 3, (32, 32))
    bad = [m for m in net.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.Dropout))]
    assert bad == []
    x = torch.randn(4, OBS_DIM)
    net.train()
    a = net(x)
    net.eval()
    assert torch.equal(a, net(x)), "train() and eval() must compute the same function"


def test_act_does_not_change_training_mode(cfg, device):
    agent = DQNAgent(cfg.agent, device, seed=0)
    before = agent.online.training
    agent.act(np.zeros((8, OBS_DIM), dtype=np.float32), epsilon=0.0)
    assert agent.online.training == before


def test_target_is_frozen_between_syncs_and_absent_from_optimizer(cfg, device):
    agent = DQNAgent(cfg.agent, device, seed=0)
    assert not any(p.requires_grad for p in agent.target.parameters())
    optim_params = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    assert not any(id(p) in optim_params for p in agent.target.parameters())

    n = 2048
    g = torch.Generator(device=device).manual_seed(0)
    obs = torch.randn(n, OBS_DIM, generator=g)
    agent.buffer.add_batch(obs, torch.randint(0, 3, (n,), generator=g),
                           torch.randn(n, generator=g), torch.zeros_like(obs), torch.ones(n))
    snapshot = [p.clone() for p in agent.target.parameters()]
    for _ in range(cfg.agent.target_sync_steps - 1):
        agent.learn()
    assert all(torch.equal(a, b) for a, b in zip(snapshot, agent.target.parameters()))
    agent.learn()
    assert any(not torch.equal(a, b) for a, b in zip(snapshot, agent.target.parameters()))


# -- replay -----------------------------------------------------------------------------


def test_replay_alignment_survives_wraparound(device):
    buf = ReplayBuffer(10, device, OBS_DIM, seed=0)
    for i in range(5):
        k = 3
        obs = torch.full((k, OBS_DIM), float(i))
        buf.add_batch(obs, torch.full((k,), i % 3, dtype=torch.int64),
                      torch.full((k,), float(i)), obs + 100.0, torch.zeros(k))
    assert len(buf) == 10
    # every row's next_obs must still be its own obs + 100
    assert torch.all(buf.next_obs[:, 0] - buf.obs[:, 0] == 100.0)
    assert torch.all(buf.rewards == buf.obs[:, 0])


def test_sample_below_batch_size_raises(device):
    buf = ReplayBuffer(64, device, OBS_DIM, seed=0)
    with pytest.raises(ValueError, match="learning_starts"):
        buf.sample(8)


def test_buffer_roundtrip(device):
    buf = ReplayBuffer(100, device, OBS_DIM, seed=0)
    n = 40
    obs = torch.randn(n, OBS_DIM)
    buf.add_batch(obs, torch.randint(0, 3, (n,)), torch.randn(n), obs + 1, torch.zeros(n))
    other = ReplayBuffer(100, device, OBS_DIM, seed=1)
    other.load_state_dict(buf.state_dict())
    assert other.size == buf.size and other.pos == buf.pos
    assert torch.equal(other.obs[:n], buf.obs[:n])


# -- schedules and learning -------------------------------------------------------------


def test_epsilon_schedule_endpoints_and_clamp(cfg):
    a = cfg.agent
    assert epsilon_at(0, a) == pytest.approx(a.epsilon_start)
    assert epsilon_at(a.epsilon_decay_steps, a) == pytest.approx(a.epsilon_final)
    assert epsilon_at(10 ** 9, a) == pytest.approx(a.epsilon_final)
    assert epsilon_at(-5, a) == pytest.approx(a.epsilon_start)
    mid = epsilon_at(a.epsilon_decay_steps // 2, a)
    assert a.epsilon_final < mid < a.epsilon_start


def test_learn_reduces_loss_on_a_learnable_batch(cfg, device):
    agent = DQNAgent(cfg.agent, device, seed=0)
    n = 2048
    g = torch.Generator(device=device).manual_seed(0)
    obs = torch.randn(n, OBS_DIM, generator=g)
    rewards = obs[:, 0] * 2.0 + obs[:, 1]          # a signal the net can actually fit
    agent.buffer.add_batch(obs, torch.randint(0, 3, (n,), generator=g), rewards.contiguous(),
                           torch.zeros_like(obs), torch.ones(n))  # terminated -> target == reward
    losses = [agent.learn() for _ in range(400)]
    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < first / 5, f"loss barely moved: {first} -> {last}"


def test_checkpoint_roundtrip_reproduces_action_sequence(cfg, device):
    agent = DQNAgent(cfg.agent, device, seed=0)
    obs = np.random.default_rng(0).standard_normal((16, OBS_DIM)).astype(np.float32)
    state = agent.state_dict()
    before = agent.act(obs, epsilon=0.0)

    restored = DQNAgent(cfg.agent, device, seed=999)
    restored.load_state_dict(state)
    assert np.array_equal(before, restored.act(obs, epsilon=0.0))
    assert not any(p.requires_grad for p in restored.target.parameters())


def test_target_sync_cadence_is_independent_of_num_envs(cfg, device):
    """Regression: sync must not be counted in optimizer updates.

    train.py holds gradient SAMPLES per transition fixed while capping update COUNT, so a cadence
    measured in updates makes target staleness a function of num_envs. At num_envs=256 that meant
    3 syncs in a 500k-step run; at 1024 the target never synced at all and the bootstrap came from
    the randomly-initialised network for the entire run.
    """
    a = cfg.agent

    def update_plan(owed):  # mirrors train.py
        n = min(a.max_updates_per_iter, owed)
        return n, max(a.batch_size, round(owed * a.batch_size / n))

    steps_per_sync = {}
    for n_envs in (1, 8, 64, 256, 1024):
        agent = DQNAgent(a, device, seed=0)
        total_steps, step, since, syncs = 200_000, 0, 0, 0
        while step < total_steps:
            step += n_envs
            since += n_envs
            owed = since // a.train_every
            if owed:
                n_up, eff = update_plan(int(owed))
                for _ in range(n_up):
                    agent.samples += eff  # count samples without paying for real gradients
                    if agent.samples - agent._samples_at_last_sync >= agent._sync_every_samples:
                        syncs += 1
                        agent._samples_at_last_sync = agent.samples
                since -= owed * a.train_every
        assert syncs > 0, f"num_envs={n_envs} never synced the target network"
        steps_per_sync[n_envs] = step / syncs

    spread = max(steps_per_sync.values()) / min(steps_per_sync.values())
    assert spread < 1.25, f"target staleness varies with num_envs: {steps_per_sync}"


def test_masked_exploration_never_picks_a_move_it_knows_is_fatal(cfg, device):
    """A random move kills a long snake about one time in five and teaches nothing doing it:
    the danger flag already said the move was fatal."""
    from snakeai.agent import DANGER_SLOTS

    agent = DQNAgent(cfg.agent, device, seed=0)
    assert agent.cfg.mask_fatal_exploration is True

    rng = np.random.default_rng(0)
    obs = rng.random((500, OBS_DIM)).astype(np.float32)
    obs[:, DANGER_SLOTS] = 0.0
    obs[:, DANGER_SLOTS[0]] = 1.0 # going straight is fatal in every one of these states
    obs[:, DANGER_SLOTS[1]] = 1.0 # so is turning left

    chosen = agent.act(obs, epsilon=1.0) # explore on every single one
    assert set(chosen.tolist()) == {2}, "only turning right was survivable, so it must pick right"


def test_masked_exploration_still_moves_when_everything_is_fatal(cfg, device):
    """Nothing to choose between, so it must still return a legal action rather than hang."""
    from snakeai.agent import DANGER_SLOTS

    agent = DQNAgent(cfg.agent, device, seed=0)
    obs = np.zeros((64, OBS_DIM), dtype=np.float32)
    obs[:, DANGER_SLOTS] = 1.0 # every move kills

    chosen = agent.act(obs, epsilon=1.0)
    assert chosen.shape == (64,)
    assert set(chosen.tolist()) <= {0, 1, 2}


def test_masking_can_be_turned_off_for_comparison(cfg, device):
    import dataclasses
    from snakeai.agent import DANGER_SLOTS

    plain = dataclasses.replace(cfg.agent, mask_fatal_exploration=False)
    agent = DQNAgent(plain, device, seed=0)

    obs = np.zeros((500, OBS_DIM), dtype=np.float32)
    obs[:, DANGER_SLOTS[0]] = 1.0 # straight is fatal, but masking is off

    chosen = agent.act(obs, epsilon=1.0)
    assert 0 in chosen.tolist(), "with masking off it should still sometimes pick the fatal move"
