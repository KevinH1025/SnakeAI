"""Animate the flood that produces free_* and tail_*, so the README can show how it works."""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import ImageDraw
from snakeai.config import preset
from snakeai.env import DIRECTIONS, SnakeEnv, turn
from tools.flood_board import build_board
from tools.media import CRASH, DIM, FOOD, SNAKE_HEAD, TEXT, board_image, font, save_gif

REGION_A = (60, 120, 220)   # the region the head can still use
REGION_B = (150, 70, 180)   # a region sealed off from it
FRONTIER = (255, 255, 255)


def flood_frames(env, entry, label, cell=22):
    """One frame per ring of the flood, so the spread is visible rather than instant."""
    W, H = env.cfg.grid_w, env.cfg.grid_h
    blocked = set(env.snake)
    tail = env.snake[-1]
    blocked.discard(tail) # the tail steps aside as the head arrives

    frames, filled = [], {}
    layer = [entry]
    seen = {entry}
    while layer:
        for c in layer:
            filled[c] = REGION_A
        overlay = dict(filled)
        for c in layer:
            overlay[c] = FRONTIER # the ring being expanded on this frame
        frames.append(annotate(env, overlay, cell, len(filled), label, tail in filled))

        nxt = []
        for x, y in layer:
            for dx, dy in DIRECTIONS:
                c = (x + dx, y + dy)
                if 0 <= c[0] < W and 0 <= c[1] < H and c not in blocked and c not in seen:
                    seen.add(c); nxt.append(c)
        layer = nxt

    # the finished picture is held by save_gif rather than by repeating frames
    frames.append(annotate(env, dict(filled), cell, len(filled), label, tail in filled))
    return frames


def annotate(env, overlay, cell, count, label, tail_in):
    hud = 62
    img = board_image(env, cell=cell, hud=hud, overlay=overlay)
    d = ImageDraw.Draw(img)
    budget = len(env.snake) + 1
    free = min(count, budget) / budget

    d.text((8, 4), f"flooding from where '{label}' lands", fill=TEXT, font=font(16))
    d.text((8, 24), f"reached {count} cells, body needs {budget}", fill=DIM, font=font(12))
    d.text((216, 24), "ring = tail", fill=CRASH, font=font(12))
    d.text((8, 42), f"free_{label} = {free:.2f}", fill=SNAKE_HEAD, font=font(13))
    d.text((140, 42), f"tail_{label} = {1 if tail_in else 0}",
           fill=SNAKE_HEAD if tail_in else FOOD, font=font(13))

    # ring the tail, since whether the flood swallows it is what tail_* reports
    tx, ty = env.snake[-1]
    x0, y0 = tx * cell, hud + ty * cell
    d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], outline=CRASH, width=2)
    return img


def main():
    env = build_board()
    head = env.snake[0]
    out = Path("docs")

    for name, action in (("straight", 0), ("left", 1), ("right", 2)):
        step = DIRECTIONS[turn(env.heading, action)]
        entry = (head[0] + step[0], head[1] + step[1])
        if entry in set(env.snake):
            print(f"  skipping {name}, it walks into the body")
            continue
        frames = flood_frames(env, entry, name)
        save_gif(frames, out / f"flood_{name}.gif", ms=300, hold_last_ms=3000)


if __name__ == "__main__":
    main()
