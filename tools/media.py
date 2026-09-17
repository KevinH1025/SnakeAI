"""Shared board drawing for the README media. Matches play.py's palette."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG = (12, 12, 16)
GRID = (28, 28, 36)
SNAKE_HEAD = (80, 255, 120)
SNAKE_BODY = (40, 170, 80)
FOOD = (255, 70, 70)
CRASH = (255, 200, 0)
TEXT = (235, 235, 240)
DIM = (140, 140, 155)

FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "Arial.ttf"


def font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_PATH), size)
    except OSError:
        return ImageFont.load_default(size)


def board_image(env, cell=14, margin=0, hud=0, overlay=None, crash=None):
    """Draw one board. `overlay` maps (x, y) -> colour and is painted under the snake."""
    w = env.cfg.grid_w * cell + 2 * margin
    h = env.cfg.grid_h * cell + 2 * margin + hud
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def box(x, y, colour, inset=0):
        x0 = margin + x * cell + inset
        y0 = margin + hud + y * cell + inset
        d.rectangle([x0, y0, x0 + cell - 1 - 2 * inset, y0 + cell - 1 - 2 * inset], fill=colour)

    for (x, y), colour in (overlay or {}).items():
        box(x, y, colour)

    for x in range(env.cfg.grid_w + 1): # a faint grid, so single cells stay readable
        px = margin + x * cell
        d.line([px, margin + hud, px, h - margin], fill=GRID)
    for y in range(env.cfg.grid_h + 1):
        py = margin + hud + y * cell
        d.line([margin, py, w - margin, py], fill=GRID)

    for i, (x, y) in enumerate(env.snake):
        box(x, y, SNAKE_HEAD if i == 0 else SNAKE_BODY, inset=1)
    if env.food is not None:
        box(env.food[0], env.food[1], FOOD, inset=3)
    if crash is not None:
        box(crash[0], crash[1], CRASH, inset=1)

    return img


def save_gif(frames, path, ms=60, loop=0, hold_last_ms=0):
    """Write a GIF. `hold_last_ms` pauses on the final frame before the loop restarts.

    Done with a per frame duration rather than by repeating the frame, because PIL's optimiser
    merges identical consecutive frames and silently swallows the pause.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms] * len(frames)
    if hold_last_ms:
        durations[-1] = hold_last_ms

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=loop, optimize=True)
    kb = path.stat().st_size / 1024
    total = (sum(durations)) / 1000
    print(f"  wrote {path}  {len(frames)} frames  {total:.1f}s  {kb:.0f} KB")
