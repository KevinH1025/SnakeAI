"""The region labelling has two implementations, compiled and plain. They must agree."""

import numpy as np
import pytest

from snakeai import env as E
from snakeai.config import preset
from snakeai.regions import HAVE_NUMBA, label_regions, label_regions_python


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
        counts, tails = env._regions(env.snake[0])
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
