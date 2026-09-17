"""Many Snake games stepped together, so the network can choose all their moves at once.

step() gives back the observation that really follows each move. The observation to act on next
lives separately in current_obs, which is also where a reset lands. Keeping those two apart is
the whole point of this class: the array you store in replay is never the one you act on next.
"""

from __future__ import annotations

import numpy as np

from .config import EnvConfig
from .env import OBS_DIM, SnakeEnv


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
            self.envs.append(SnakeEnv(cfg, seed=seed * 1_000_003 + i))

        self.current_obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32) # what to act on next
        self._next_obs = np.zeros((n_envs, OBS_DIM), dtype=np.float32) # what goes into replay
        self._rewards = np.zeros(n_envs, dtype=np.float32)
        self._terminated = np.zeros(n_envs, dtype=np.float32)
        self._truncated = np.zeros(n_envs, dtype=np.float32)

        # Running totals for the episode each game is currently in.
        self.episode_return = np.zeros(n_envs, dtype=np.float64)
        self.episode_steps = np.zeros(n_envs, dtype=np.int64)

        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        for i, env in enumerate(self.envs):
            obs = env.reset(seed=None if seed is None else seed * 1_000_003 + i)
            self.current_obs[i] = obs
        self.episode_return[:] = 0.0
        self.episode_steps[:] = 0
        return self.current_obs

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step every env once.

        Returns (next_obs, rewards, terminated, truncated, finished). `finished` holds one info
        dict per game whose episode ended on this step.
        """
        finished: list[dict] = []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(int(actions[i]))
            self._next_obs[i] = obs # the real successor, this is what replay stores
            self._rewards[i] = reward
            self._terminated[i] = float(terminated)
            self._truncated[i] = float(truncated)

            self.episode_return[i] += reward
            self.episode_steps[i] += 1

            if terminated or truncated:
                finished.append({
                    **info,
                    "env": i,
                    "return": float(self.episode_return[i]),
                    "episode_steps": int(self.episode_steps[i]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                })
                self.current_obs[i] = env.reset() # the reset lands HERE, not in _next_obs
                self.episode_return[i] = 0.0
                self.episode_steps[i] = 0

            else:
                self.current_obs[i] = obs # still alive, so next_obs and current_obs agree

        return self._next_obs, self._rewards, self._terminated, self._truncated, finished
