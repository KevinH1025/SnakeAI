"""Many Snake games stepped together, so the network can choose all their moves at once.

step() gives back the observation that really follows each move. The observation to act on next
lives separately in current_obs, which is also where a reset lands. Keeping those two apart is
the whole point of this class: the array you store in replay is never the one you act on next.
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .env import OBS_DIM, SnakeEnv


def _seed_for(base: int, index: int) -> int:
    """The seed for game `index` in a run seeded with `base`."""
    return base * 1_000_003 + index # spread the games apart, all from one run seed


class VecSnakeEnv:
    """n_envs independent SnakeEnv instances behind a batched interface."""

    def __init__(self, cfg: EnvConfig, n_envs: int, seed: int = 0) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs must be >= 1, got {n_envs}")
        self.cfg = cfg
        self.n_envs = n_envs

        # A different seed per game, but all derived from `seed`, so the run still reproduces.
        self.envs = []
        for i in range(n_envs):
            self.envs.append(SnakeEnv(cfg, seed=_seed_for(seed, i))) # one game per slot

        # Scratch buffers refilled by every step() call, one row per game.
        self.current_obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32) # what to act on next
        self._next_obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32) # what goes into replay
        self._rewards = np.zeros(n_envs, dtype=np.float32) # reward from the last move
        self._terminated = np.zeros(n_envs, dtype=np.float32) # 1.0 where the snake died
        self._truncated = np.zeros(n_envs, dtype=np.float32) # 1.0 where the step cap hit

        # Running totals for the episode each game is currently in.
        self.episode_return = np.zeros(n_envs, dtype=np.float64)
        self.episode_steps = np.zeros(n_envs, dtype=np.int64)

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        for i, env in enumerate(self.envs):
            obs = env.reset(seed=None if seed is None else _seed_for(seed, i))
            self.current_obs[i] = obs
        self.episode_return[:] = 0.0
        self.episode_steps[:] = 0
        return self.current_obs

    def _record_step(self, i: int, obs: np.ndarray, reward: float,
                     terminated: bool, truncated: bool) -> None:
        """Put one env's step outputs into the batched buffers and add them to its totals."""
        self._next_obs[i] = obs # the real successor, this is what replay stores
        self._rewards[i] = reward
        self._terminated[i] = float(terminated)
        self._truncated[i] = float(truncated)

        self.episode_return[i] += reward # running total for the episode in progress
        self.episode_steps[i] += 1

    def _episode_info(self, i: int, info: dict, terminated: bool, truncated: bool) -> dict:
        """The report for a game whose episode ended on this step."""
        return {
            **info, # what the env itself reported, score and reason
            "env": i,
            "return": float(self.episode_return[i]),
            "episode_steps": int(self.episode_steps[i]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def _start_new_episode(self, i: int, env: SnakeEnv) -> None:
        """Reset one game and clear the totals for the episode that just ended."""
        self.current_obs[i] = env.reset() # the reset lands HERE, not in _next_obs
        self.episode_return[i] = 0.0
        self.episode_steps[i] = 0

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step every env once.

        Returns (next_obs, rewards, terminated, truncated, finished). `finished` holds one info
        dict per game whose episode ended on this step.
        """
        finished: list[dict] = []

        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(int(actions[i]))
            self._record_step(i, obs, reward, terminated, truncated)

            done = terminated or truncated # died, or ran out of the hunger clock
            if done:
                finished.append(self._episode_info(i, info, terminated, truncated))
                self._start_new_episode(i, env)
            else:
                self.current_obs[i] = obs # still alive, so next_obs and current_obs agree

        return self._next_obs, self._rewards, self._terminated, self._truncated, finished

    def _current_scores(self) -> list[int]:
        """The score every game is sitting on right now, running or not."""
        return [env.score for env in self.envs] # env.score is food eaten this game

    def live_scores(self) -> tuple[float, float]:
        """The mean and best score across every game as it stands right now.

        Counts games still in progress, unlike anything built from finished episodes. A strong
        agent plays very long games, so waiting for them to end biases the numbers towards the
        games that died early, which are exactly the bad ones.
        """
        scores = self._current_scores()
        mean = sum(scores) / self.n_envs # over every game, not only the finished ones
        best = max(scores) # n_envs >= 1, so the list is never empty

        return mean, float(best)

    def best_live_score(self) -> float:
        """The highest score any game is currently sitting on."""
        return float(max(self._current_scores())) # same max live_scores reports
