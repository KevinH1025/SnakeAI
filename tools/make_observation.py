"""Draw the whole observation next to the board it came from, so the README can show what the
network actually gets: seventeen numbers, no picture of the board at all."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image, ImageDraw
from snakeai.env import DIRECTIONS, OBS_NAMES, turn
from tools.flood_board import build_board
from tools.media import BG, CRASH, DIM, FOOD, GRID, SNAKE_HEAD, TEXT, board_image, font, save_gif

GROUPS = [
    ("is this move instantly fatal?", ["danger_straight", "danger_left", "danger_right"]),
    ("which way is the food?", ["food_ahead", "food_behind", "food_left", "food_right",
                                "food_forward", "food_lateral"]),
    ("how am I doing?", ["length_frac", "hunger_frac"]),
    ("how much room does each move lead into?", ["free_straight", "free_left", "free_right"]),
    ("could I still reach my tail afterwards?", ["tail_straight", "tail_left", "tail_right"]),
]
BAR_POS = (90, 200, 240)
BAR_NEG = (200, 90, 90)


def main():
    env = build_board()
    obs = env.observe()
    values = dict(zip(OBS_NAMES, obs))
    cell = 18

    head = env.snake[0]

    # shade the sealed pocket, so free_straight = 0.36 has something visible behind it
    from collections import deque
    W, Hh = env.cfg.grid_w, env.cfg.grid_h
    blocked = set(env.snake)
    blocked.discard(env.snake[-1])
    step0 = DIRECTIONS[turn(env.heading, 0)]
    start = (head[0] + step0[0], head[1] + step0[1])
    pocket, q = {start}, deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS:
            c = (x + dx, y + dy)
            if 0 <= c[0] < W and 0 <= c[1] < Hh and c not in blocked and c not in pocket:
                pocket.add(c); q.append(c)

    overlay = {c: (46, 60, 96) for c in pocket} # the eight cells straight can reach
    board = board_image(env, cell=cell, overlay=overlay)
    bd = ImageDraw.Draw(board)

    # the move markers go on TOP, or the body paints over the fatal one
    for action, label in ((0, "S"), (1, "L"), (2, "R")):
        step = DIRECTIONS[turn(env.heading, action)]
        x, y = head[0] + step[0], head[1] + step[1]
        fatal = values[f"danger_{['straight', 'left', 'right'][action]}"] > 0.5
        edge = FOOD if fatal else (120, 190, 255)
        bd.rectangle([x * cell, y * cell, x * cell + cell - 1, y * cell + cell - 1],
                     outline=edge, width=2)
        bd.text((x * cell + 5, y * cell + 2), label, fill=edge, font=font(13))

    panel_w = 430
    img = Image.new("RGB", (board.width + panel_w, max(board.height, 470) + 82), BG)
    d = ImageDraw.Draw(img)
    d.text((10, 12), "what the snake sees", fill=TEXT, font=font(19))
    d.text((10, 34), "S, L and R are where each move lands, red means it kills. The shaded\ncells are all that going straight can reach.", fill=DIM, font=font(12))
    img.paste(board, (10, 70))

    x0, y = board.width + 26, 70
    for title, names in GROUPS:
        d.text((x0, y), title, fill=SNAKE_HEAD, font=font(13))
        y += 19
        for name in names:
            v = float(values[name])
            d.text((x0 + 6, y), name, fill=TEXT, font=font(12))
            bx = x0 + 190
            d.rectangle([bx, y + 3, bx + 120, y + 11], outline=GRID) # the -1..1 track
            mid = bx + 60
            d.line([mid, y + 1, mid, y + 13], fill=GRID)
            w = int(abs(v) * 60)
            if w:
                colour = BAR_POS if v > 0 else BAR_NEG
                left, right = (mid, mid + w) if v > 0 else (mid - w, mid) # PIL wants x0 <= x1
                d.rectangle([left, y + 4, right, y + 10], fill=colour)
            d.text((bx + 132, y), f"{v:+.2f}", fill=DIM, font=font(12))
            y += 16
        y += 10

    d.text((x0, y + 4), f"{len(OBS_NAMES)} numbers in total. That is the whole input.",
           fill=CRASH, font=font(13))

    out = Path("docs/observation.png")
    img.save(out)
    print(f"  wrote {out}  {img.size[0]}x{img.size[1]}  {out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
