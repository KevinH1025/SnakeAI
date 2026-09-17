"""Vectorised env tests. The first one is the reason this file exists.

Auto-resetting vector envs are the classic place to reintroduce v1's worst data bug: returning the
post-reset observation as the transition's next_obs.
"""

import numpy as np
import pytest

from snakeai.config import apply_overrides, preset
from snakeai.env import SnakeEnv
from snakeai.vecenv import VecSnakeEnv


def test_next_obs_is_the_true_successor_never_a_reset(cfg):
    """A single vec env must be transition-identical to a raw env fed the same actions.

    Crucially this includes the terminal step: v1 read get_state() after update() had already
    called reset(), so every terminal transition stored a brand-new episode's first observation.
    """
    seed = 5
    vec = VecSnakeEnv(cfg.env, n_envs=1, seed=seed)
    raw = SnakeEnv(cfg.env, seed=seed * 1_000_003)
    raw.reset(seed=seed * 1_000_003)

    np.testing.assert_allclose(vec.current_obs[0], raw.observe())
    rng = np.random.default_rng(0)
    saw_terminal = False
    for _ in range(4000):
        action = int(rng.integers(3))
        next_obs, rewards, terminated, truncated, finished = vec.step(np.array([action]))
        raw_obs, raw_reward, raw_term, raw_trunc, _ = raw.step(action)

        np.testing.assert_allclose(next_obs[0], raw_obs, atol=1e-6,
                                   err_msg="next_obs diverged from the true successor")
        assert rewards[0] == pytest.approx(raw_reward, abs=1e-6)
        assert bool(terminated[0]) == raw_term
        assert bool(truncated[0]) == raw_trunc

        if raw_term or raw_trunc:
            saw_terminal = True
            # The reset must land in current_obs and must NOT be what we just stored.
            assert finished, "episode ended but nothing was reported as finished"
            assert not np.allclose(vec.current_obs[0], next_obs[0]), \
                "current_obs still holds the terminal observation - reset did not happen"
            break
    assert saw_terminal, "never reached a terminal step"


def test_finished_reports_only_on_episode_end(cfg):
    vec = VecSnakeEnv(cfg.env, n_envs=8, seed=1)
    rng = np.random.default_rng(2)
    for _ in range(300):
        _, _, terminated, truncated, finished = vec.step(rng.integers(0, 3, size=8))
        expected = int((terminated + truncated).sum())
        assert len(finished) == expected
        for info in finished:
            assert info["terminated"] or info["truncated"]
            assert info["episode_steps"] >= 1
            assert set(("env", "score", "return", "reason")).issubset(info)


def test_episode_counters_reset_with_the_episode(cfg):
    vec = VecSnakeEnv(cfg.env, n_envs=4, seed=3)
    rng = np.random.default_rng(4)
    for _ in range(500):
        _, _, _, _, finished = vec.step(rng.integers(0, 3, size=4))
        for info in finished:
            assert vec.episode_steps[info["env"]] == 0
            assert vec.episode_return[info["env"]] == 0.0


def test_num_envs_one_is_supported(cfg):
    vec = VecSnakeEnv(cfg.env, n_envs=1, seed=0)
    obs = vec.reset(seed=0)
    assert obs.shape == (1, vec.current_obs.shape[1])
    with pytest.raises(ValueError):
        VecSnakeEnv(cfg.env, n_envs=0, seed=0)


def test_envs_are_independent(cfg):
    """Distinct seeds per env: they must not all be the same game in lockstep."""
    vec = VecSnakeEnv(cfg.env, n_envs=16, seed=7)
    rng = np.random.default_rng(8)
    for _ in range(40):
        vec.step(rng.integers(0, 3, size=16))
    rows = {tuple(np.round(row, 5)) for row in vec.current_obs}
    assert len(rows) > 1, "all parallel envs are in identical states"
