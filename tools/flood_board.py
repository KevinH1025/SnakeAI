"""The demo position the README media is built from.

Taken from a real game rather than placed by hand, so the body is a shape the snake could
actually have got into. Turning right here walks into a pocket of about seven cells, turning
left stays on the open board, and the two floods that follow are what free_* and tail_* report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from snakeai.config import apply_overrides, preset
from snakeai.env import SnakeEnv

# head first, exactly as the env stores it
BODY = [(9, 10), (9, 11), (10, 11), (11, 11), (12, 11), (12, 12), (13, 12), (14, 12), (15, 12), (15, 11), (15, 10), (15, 9), (14, 9), (13, 9), (12, 9), (11, 9), (10, 9), (9, 9), (8, 9)]
HEADING = 0
FOOD = (9, 5)
GRID = (16, 16)


def build_board():
    cfg = apply_overrides(preset("small"),
                          [f"env.grid_w={GRID[0]}", f"env.grid_h={GRID[1]}"]).env
    env = SnakeEnv(cfg, seed=0)
    env.reset(seed=0)
    env.heading = HEADING
    env.snake = list(BODY)
    env.occupied = set(env.snake)
    env.food = FOOD

    assert len(set(env.snake)) == len(env.snake), "the body crosses itself"
    return env
