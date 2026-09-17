"""Environment tests. Every one of these pins a specific v1 defect."""

import dataclasses

import numpy as np
import pytest

from snakeai import env as E
from snakeai.config import apply_overrides, preset


def small_env(seed=0):
    return E.SnakeEnv(preset("small").env, seed=seed)


# -- episode boundaries -----------------------------------------------------------------


def test_step_after_episode_end_raises():
    """v1's update() called reset() inline, so 'the episode is over' was never observable."""
    env = small_env(1)
    env.reset(seed=1)
    while not env.done:
        env.step(0)
    with pytest.raises(RuntimeError, match="finished episode"):
        env.step(0)


def test_terminal_step_does_not_mutate_snake():
    """A fatal head is never committed, so the terminal observation is of the board that died."""
    env = small_env(2)
    env.reset(seed=2)
    while True:
        snake_before, score_before = list(env.snake), env.score
        _, _, terminated, truncated, info = env.step(0)
        if terminated:
            assert list(env.snake) == snake_before
            assert env.score == score_before
            assert info["crash_cell"] is not None
            return
        if truncated:
            env.reset()


def test_truncation_is_not_termination():
    """The hunger clock is a time limit. v1 reported it as death, which zeroed the bootstrap."""
    cfg = apply_overrides(preset("small"), ["env.max_steps_without_food=16"]).env
    env = E.SnakeEnv(cfg, seed=3)
    env.reset(seed=3)
    for _ in range(5000):
        if env.done:
            if env.truncated:
                assert env.last_event is E.Event.STARVED
                assert env.terminated is False, "starving must never set terminated"
                return
            env.reset()
        env.step(0 if env.steps % 3 else 1)  # circle without eating
    pytest.skip("did not starve within the step budget")


# -- observation ------------------------------------------------------------------------


def test_obs_shape_dtype_range_and_not_aliased():
    env = small_env(4)
    obs = env.reset(seed=4)
    assert obs.shape == (E.OBS_DIM,) and obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    # A reused scratch buffer would make every replay row alias the latest observation.
    assert env.observe() is not env.observe()


def test_obs_is_rotation_equivariant():
    """Heading-relative encoding: the same situation rotated must look identical.

    v1 encoded danger in screen coordinates plus a heading one-hot, so the network had to learn
    twelve conjunctions to answer 'will I die if I go straight'.
    """
    cfg = preset("small").env

    def obs_for(heading):
        env = E.SnakeEnv(cfg, seed=11)
        env.reset(seed=11)
        env.heading = heading
        f = E.DIRECTIONS[heading]
        r = E.DIRECTIONS[(heading + 1) % 4]
        b = E.DIRECTIONS[(heading + 2) % 4]
        head = (4, 4)
        env.snake = [(head[0] + b[0] * i, head[1] + b[1] * i) for i in range(3)]
        env.occupied = set(env.snake)
        env.food = (head[0] + f[0] * 3 + r[0], head[1] + f[1] * 3 + r[1])
        env.steps_since_food = 7
        return env.observe()

    base = obs_for(0)
    for heading in range(1, 4):
        np.testing.assert_allclose(base, obs_for(heading), atol=1e-6)


def test_danger_flags_match_real_transitions():
    """The observation cannot disagree with the dynamics: both call _would_die."""
    cfg = preset("small").env
    idx = (E.I_DANGER_STRAIGHT, E.I_DANGER_LEFT, E.I_DANGER_RIGHT)
    rng = np.random.default_rng(0)
    checked = 0
    for episode in range(60):
        env = E.SnakeEnv(cfg, seed=episode)
        env.reset(seed=episode)
        while not env.done:
            predicted = bool(env.observe()[idx[(action := int(rng.integers(3)))]])
            _, _, terminated, _, _ = env.step(action)
            assert predicted == terminated, "danger flag disagreed with the actual outcome"
            checked += 1
    assert checked > 500


def test_tail_cell_is_not_fatal_unless_growing():
    """The tail vacates as the head arrives. v1 treated it as lethal, teaching a false fear."""
    cfg = preset("small").env
    env = E.SnakeEnv(cfg, seed=5)
    env.reset(seed=5)
    env.heading = E.RIGHT
    env.snake = [(3, 3), (3, 4), (4, 4), (4, 3)]  # head at (3,3), tail at (4,3) directly right
    env.occupied = set(env.snake)
    env.food = (0, 0)
    assert env._would_die((3, 3), E.RIGHT) is False, "moving into the vacating tail is legal"

    # If the snake were growing, the tail would NOT vacate, so the same move becomes a collision.
    # That state is unreachable in play (see test_food_never_spawns_on_the_snake), but the
    # predicate must still be self-consistent about it.
    env.food = (4, 3)
    assert env._would_die((3, 3), E.RIGHT) is True, "a tail that does not vacate is solid"

    # And a body cell that is not the tail is always fatal.
    assert env._would_die((3, 3), E.DOWN) is True


def test_food_never_spawns_on_the_snake():
    """The only reason 'food on the tail' never arises. v1's Food.__init__ skipped this check."""
    cfg = preset("small").env
    for seed in range(40):
        env = E.SnakeEnv(cfg, seed=seed)
        env.reset(seed=seed)
        assert env.food not in env.occupied
        while not env.done:
            env.step(int(np.random.default_rng(env.steps + seed).integers(3)))
            if env.food is not None:
                assert env.food not in env.occupied, "food spawned inside the snake"


# -- reward -----------------------------------------------------------------------------


def test_step_outcome_fields_are_exactly_three():
    """The struct is the safety property: no snake, no length, no clock is in scope."""
    names = [f.name for f in dataclasses.fields(E.StepOutcome)]
    assert names == ["event", "prev_dist_norm", "next_dist_norm"]


def test_reward_is_bounded_and_length_independent():
    """v1's `reward += len(body) * 0.1` grew without bound and swamped the food signal."""
    rc = preset("small").env.rewards
    values = [
        E.compute_reward(E.StepOutcome(ev, p, n), rc)
        for ev in E.Event for p in (0.0, 0.5, 1.0) for n in (0.0, 0.5, 1.0)
    ]
    assert min(values) > -2.0 and max(values) < 3.0

    # And empirically: death costs the same at every snake length.
    cfg = preset("small").env
    deaths = []
    for extra in (0, 5, 20):
        env = E.SnakeEnv(cfg, seed=7)
        env.reset(seed=7)
        env.heading = E.RIGHT
        env.snake = [(6 - i, 2) for i in range(1 + extra)]
        env.occupied = set(env.snake)
        env.food = (0, 7)
        env.snake[0] = (cfg.grid_w - 1, 2)  # hard against the right wall
        env.occupied = set(env.snake)
        _, reward, terminated, _, _ = env.step(0)
        assert terminated
        deaths.append(round(reward, 6))
    assert len(set(deaths)) == 1, f"death reward varied with length: {deaths}"


def test_shaping_telescopes_and_terminal_has_no_shaping():
    """Potential-based shaping sums to ~0 around a closed loop, so it cannot be farmed."""
    rc = preset("small").env.rewards
    loop = [0.2, 0.4, 0.6, 0.4]
    total = sum(
        rc.shaping_gamma * (-rc.shaping_scale * nxt) - (-rc.shaping_scale * cur)
        for cur, nxt in zip(loop, loop[1:] + loop[:1])
    )
    assert abs(total) < 0.02, f"closed loop accrued {total}"

    # A terminal state's potential is defined to be zero.
    terminal = E.compute_reward(E.StepOutcome(E.Event.HIT_WALL, 0.5, 0.9), rc)
    expected = rc.step + rc.death + rc.shaping_gamma * 0.0 - (-rc.shaping_scale * 0.5)
    assert terminal == pytest.approx(expected)


# -- determinism ------------------------------------------------------------------------


def test_same_seed_same_trajectory():
    def rollout(seed):
        env = small_env(seed)
        env.reset(seed=seed)
        rng = np.random.default_rng(99)
        out = []
        for _ in range(400):
            if env.done:
                env.reset()
            _, reward, term, trunc, info = env.step(int(rng.integers(3)))
            out.append((round(reward, 6), term, trunc, info["reason"]))
        return out

    assert rollout(3) == rollout(3)
    assert rollout(3) != rollout(4)


def test_full_board_is_a_win_not_a_crash():
    """Filling the board is success. v1's Food.spawn raised RuntimeError instead."""
    cfg = apply_overrides(preset("small"), ["env.grid_w=5", "env.grid_h=5"]).env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    cells = [(x, y) for y in range(5) for x in range(5)]
    env.snake = cells[:24]
    env.occupied = set(env.snake)
    env.heading = E.RIGHT
    env.snake = [(3, 0), (2, 0), (1, 0), (0, 0)] + [c for c in cells[5:] if c not in {(0, 0), (1, 0), (2, 0), (3, 0)}]
    env.occupied = set(env.snake)
    env.food = (4, 0)
    _, reward, terminated, _, info = env.step(0)
    assert terminated and info["reason"] == "won"
    assert reward > 0


def test_free_space_sees_a_pocket_the_danger_flag_misses():
    """A one-cell lookahead says "safe" about a move into a sealed dead end."""
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.RIGHT

    # Head at (10,10). The cells that seal the pocket come early in the body and the tail is
    # parked well away from it, so the tail vacating cannot open the pocket up.
    env.snake = [
        (10, 10),                                              # head
        (11, 9), (12, 9), (13, 10), (12, 11), (11, 11),        # the walls of the pocket
        (9, 10), (8, 10), (7, 10), (6, 10), (5, 10),           # trailing body, tail last
    ]
    env.occupied = set(env.snake)
    env.food = (30, 25)
    obs = env.observe()

    assert obs[E.I_DANGER_STRAIGHT] == 0.0, "the next cell really is empty"
    assert obs[E.I_FREE_STRAIGHT] < 0.3, "but it is a pocket and free_straight must say so"
    assert obs[E.I_FREE_LEFT] > 0.9 and obs[E.I_FREE_RIGHT] > 0.9, "the other ways out are open"


def test_free_space_agrees_with_the_danger_flags_about_the_tail():
    """Moving onto the tail is legal, so it must not read as zero room.

    The tail steps out of the way as the head arrives. If _free_spaces treated the whole body as
    solid, the agent would see "safe" and "no room" for the same move.
    """
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.RIGHT
    env.snake = [(4, 4), (3, 4), (2, 4), (2, 3), (3, 3), (4, 3)] # (4,3) is the tail, directly left
    env.occupied = set(env.snake)
    env.food = (30, 25)
    obs = env.observe()

    assert obs[E.I_DANGER_LEFT] == 0.0, "turning left onto the vacating tail is legal"
    assert obs[E.I_FREE_LEFT] > 0.0, "so it must not report zero room"


def test_free_space_is_one_on_an_open_board():
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.RIGHT
    env.snake = [(10, 10)] + [(10 - i, 10) for i in range(1, 6)]
    env.occupied = set(env.snake)
    env.food = (30, 25)
    obs = env.observe()
    for i in (E.I_FREE_STRAIGHT, E.I_FREE_LEFT, E.I_FREE_RIGHT):
        assert obs[i] == pytest.approx(1.0)


def test_free_space_zero_when_the_move_is_fatal():
    """If the cell cannot be entered at all, there is no space behind it."""
    cfg = preset("small").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.RIGHT
    env.snake = [(cfg.grid_w - 1, 3), (cfg.grid_w - 2, 3), (cfg.grid_w - 3, 3)]
    env.occupied = set(env.snake)
    env.food = (0, 0)
    obs = env.observe()
    assert obs[E.I_DANGER_STRAIGHT] == 1.0, "straight is into the wall"
    assert obs[E.I_FREE_STRAIGHT] == 0.0, "so free_straight must be exactly 0"
