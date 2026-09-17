"""The region labelling has two implementations, compiled and plain. They must agree."""

import numpy as np
import pytest

from snakeai import env as E
from snakeai.config import preset
from snakeai.regions import (HAVE_NUMBA, label_regions, label_regions_python, reach_counts,
                             reach_counts_python)


def real_boards(n=60, seed=1):
    """Boards from an actual game rather than hand built ones, so the shapes are realistic."""
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    boards = []

    while len(boards) < n:
        if env.done:
            env.reset()
        env.step(int(rng.integers(3)))
        if len(env.snake) > 4:
            boards.append(env)
            env = E.SnakeEnv(cfg, seed=seed + len(boards))
            env.reset(seed=seed + len(boards))
            for _ in range(len(boards) * 3):
                if env.done:
                    env.reset()
                env.step(int(rng.integers(3)))

    return boards


def call(fn, env):
    """Run one labelling implementation against a board."""
    width, height = env.cfg.grid_w, env.cfg.grid_h
    cells = width * height

    blocked = np.zeros(cells, np.uint8)
    for x, y in env.snake:
        blocked[x + y * width] = 1
    tail_x, tail_y = env.snake[-1]
    tail = tail_x + tail_y * width
    blocked[tail] = 0

    entries = np.full(3, -1, np.int32)
    head = env.snake[0]
    for slot, action in enumerate((0, 1, 2)):
        step = E.DIRECTIONS[E.turn(env.heading, action)]
        x, y = head[0] + step[0], head[1] + step[1]
        if 0 <= x < width and 0 <= y < height and not blocked[x + y * width]:
            entries[slot] = x + y * width

    return fn(blocked, entries, tail, len(env.snake) + 1, width, height,
              np.zeros(cells, np.int32), np.zeros(cells + 2, np.int32), np.zeros(cells + 8, np.int32))


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed, only one implementation exists")
def test_compiled_and_plain_labelling_agree():
    """If these ever diverge the compiled path is silently training on different numbers."""
    for env in real_boards():
        fast_counts, fast_tails = call(label_regions, env)
        slow_counts, slow_tails = call(label_regions_python, env)
        np.testing.assert_array_equal(fast_counts, slow_counts)
        np.testing.assert_array_equal(fast_tails, slow_tails)


def test_labelling_agrees_with_the_flood_implementation():
    """The floods are the original reference. Labelling must give the same answers."""
    for env in real_boards():
        counts, tails, _ = env._regions(env.snake[0])
        blocked = env._blocked_cells()
        assert tuple(int(c) for c in counts) == env._free_spaces(
            env.snake[0], len(env.snake) + 1, blocked)
        assert tuple(float(t) for t in tails) == env._tail_reachable(env.snake[0], blocked)


def test_a_sealed_pocket_is_seen_as_unreachable():
    """The case the whole feature exists for."""
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.UP
    env.snake = [(6, 0)] + [(6, y) for y in range(1, 30)] + [(5, 29)] # a wall splitting the board
    env.occupied = set(env.snake)
    env.food = (1, 1)

    obs = env.observe()
    assert obs[E.I_FREE_LEFT] == obs[E.I_FREE_RIGHT] == 1.0, "both halves are roomy"
    assert obs[E.I_TAIL_LEFT] == 1.0, "the tail is in the left half"
    assert obs[E.I_TAIL_RIGHT] == 0.0, "the right half is sealed off from it"


def call_reach(fn, env, depth=E.REACH_DEPTH):
    """Run one reach implementation against a board, set up exactly as _regions does."""
    width, height = env.cfg.grid_w, env.cfg.grid_h
    cells = width * height

    blocked = np.zeros(cells, np.uint8)
    for x, y in env.snake:
        blocked[x + y * width] = 1
    tail_x, tail_y = env.snake[-1]
    blocked[tail_x + tail_y * width] = 0

    entries = np.full(3, -1, np.int32)
    head = env.snake[0]
    for slot, action in enumerate((0, 1, 2)):
        step = E.DIRECTIONS[E.turn(env.heading, action)]
        x, y = head[0] + step[0], head[1] + step[1]
        if 0 <= x < width and 0 <= y < height and not blocked[x + y * width]:
            entries[slot] = x + y * width

    seen = np.zeros(cells, np.uint8)
    out = fn(blocked, entries, depth, width, height, seen, np.zeros(cells, np.int32))
    assert not seen.any(), "the scratch array was left dirty, so the next call would undercount"
    return out


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed, only one implementation exists")
def test_compiled_and_plain_reach_agree():
    for env in real_boards():
        np.testing.assert_array_equal(call_reach(reach_counts, env),
                                      call_reach(reach_counts_python, env))


def test_reach_counts_the_whole_diamond_when_nothing_is_in_the_way():
    """With an empty board a walk of REACH_DEPTH covers every cell within that many moves."""
    cfg = preset("big").env
    width, height = cfg.grid_w, cfg.grid_h
    cells = width * height

    blocked = np.zeros(cells, np.uint8) # nothing on the board at all
    middle = 20 + 15 * width # far enough from every edge that the whole diamond fits
    entries = np.array([middle, -1, -1], np.int32) # only the first move is playable

    reach = reach_counts(blocked, entries, E.REACH_DEPTH, width, height,
                         np.zeros(cells, np.uint8), np.zeros(cells, np.int32))

    assert int(reach[0]) == E.REACH_MAX
    assert int(reach[1]) == int(reach[2]) == 0, "a move that cannot be played opens nothing up"


def test_reach_separates_moves_that_free_space_reports_as_equal():
    """The gap this feature fills.

    free_* reads off one board labelling, so it hands back the same number for every move that
    opens into the same region, which on a mostly empty board is nearly all of them. Measuring
    a trained net put that at 95% of steps past length 200. reach_* has to break those ties or
    it is not earning its three slots.
    """
    free_slots = (E.I_FREE_STRAIGHT, E.I_FREE_LEFT, E.I_FREE_RIGHT)
    reach_slots = (E.I_REACH_STRAIGHT, E.I_REACH_LEFT, E.I_REACH_RIGHT)
    danger_slots = (E.I_DANGER_STRAIGHT, E.I_DANGER_LEFT, E.I_DANGER_RIGHT)
    tied = 0
    separated = 0

    for env in real_boards(n=120, seed=7):
        obs = env.observe()
        live = [i for i in range(3) if obs[danger_slots[i]] < 0.5] # ignore moves that just die
        if len(live) < 2:
            continue # nothing to tell apart

        if len({round(float(obs[free_slots[i]]), 4) for i in live}) == 1:
            tied += 1
            if len({round(float(obs[reach_slots[i]]), 4) for i in live}) > 1:
                separated += 1

    assert tied >= 20, f"free_* only tied on {tied} boards, so this sample proves nothing"
    assert separated / tied > 0.5, f"reach_* broke only {separated} of {tied} ties"


def test_a_fatal_move_reaches_nothing():
    """Turning left while facing right means going up, and the top row has nothing above it."""
    cfg = preset("big").env
    env = E.SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = E.RIGHT
    env.snake = [(20, 0), (19, 0), (18, 0)] # head on the top row
    env.occupied = set(env.snake)

    obs = env.observe()

    assert obs[E.I_DANGER_LEFT] == 1.0
    assert obs[E.I_REACH_LEFT] == 0.0, "a move that kills should not report room"
    assert obs[E.I_REACH_STRAIGHT] > 0.0, "carrying on along the row is fine"
