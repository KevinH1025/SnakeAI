"""Stitch greedy games from successive checkpoints into one GIF, so the README can show the
agent getting better rather than just assert that it does."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import ImageDraw
from snakeai.agent import DQNAgent
from snakeai.config import from_dict
from snakeai.env import SnakeEnv
from tools.media import CRASH, DIM, SNAKE_HEAD, TEXT, board_image, font, save_gif

FRAMES_PER_STAGE = 80 # each stage gets the same screen time however long its game ran
SEEDS_PER_STAGE = 24 # one game says nothing, scores at a fixed checkpoint vary over 2x
CELL = 20
HUD = 66 # three lines: the stage, the score, and how the game ended


def load(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = from_dict(state["config"]) if isinstance(state["config"], dict) else state["config"]
    agent = DQNAgent(cfg.agent, device, seed=0)
    weights = {k: v for k, v in state["agent"].items() if k != "buffer"}
    agent.load_state_dict(weights)
    agent.online.eval()
    return agent, cfg, int(state.get("step", 0))


def play(agent, cfg, seed, cap=4000):
    """One greedy game. Returns a board snapshot per move."""
    env = SnakeEnv(cfg.env, seed=seed)
    obs = env.reset(seed=seed)
    shots = [(list(env.snake), env.food, env.score, None)]
    for _ in range(cap):
        action = int(agent.act(obs, epsilon=0.0)[0])
        obs, _, term, trunc, info = env.step(action)
        shots.append((list(env.snake), env.food, env.score, env.crash_cell))
        if term or trunc:
            break
    return shots, env.score, env.last_event.value


def render(cfg, shots, step, score, reason, mean, held):
    """Even a long game gets FRAMES_PER_STAGE frames, so stages are comparable."""
    idx = np.linspace(0, len(shots) - 1, min(FRAMES_PER_STAGE, len(shots))).astype(int)
    env = SnakeEnv(cfg.env, seed=0)
    env.reset(seed=0)
    out = []
    for n, i in enumerate(idx):
        body, food, sc, crash = shots[i]
        env.snake = body
        env.food = food
        img = board_image(env, cell=CELL, hud=HUD, crash=crash)
        d = ImageDraw.Draw(img)
        d.text((8, 4), f"after {step:,} training steps", fill=TEXT, font=font(17))
        d.text((8, 26), f"score {sc}", fill=SNAKE_HEAD, font=font(14))
        d.text((img.width - 150, 27), f"typical game: {mean:.0f}", fill=DIM, font=font(13))
        if i == len(shots) - 1: # its own line, or it collides with the numbers above
            d.text((8, 46), f"game over: {reason}", fill=CRASH, font=font(14))
        out.append(img)
    out += [out[-1]] * held # linger on the end of each game
    return out


def main():
    snaps = sorted(Path(sys.argv[1]).glob("snap_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not snaps:
        raise SystemExit("no snap_*.pt found")
    if len(sys.argv) > 2: # pick which snapshots to show, e.g. 50000,150000,300000,1200000
        wanted = {int(v) for v in sys.argv[2].split(",")}
        snaps = [p for p in snaps if int(p.stem.split("_")[1]) in wanted]
    device = torch.device("cpu")
    seeds = [4242 + i for i in range(SEEDS_PER_STAGE)]

    frames = []
    for path in snaps:
        agent, cfg, step = load(path, device)

        # one game is far too noisy to stand for a stage: scores at a fixed checkpoint range
        # over more than 2x. Play several, report the mean, and show the median game so the
        # picture and the number agree.
        games = [play(agent, cfg, seed=sd) for sd in seeds]
        scores = [g[1] for g in games]
        mean = float(np.mean(scores))
        order = np.argsort(scores)
        shots, score, reason = games[order[len(order) // 2]]

        frames += render(cfg, shots, step, score, reason, mean, held=10)
        print(f"  {step:>9,} steps -> mean {mean:6.1f} over {len(seeds)} games"
              f"  (shown: {score}, {reason})", flush=True)

    save_gif(frames, "docs/learning.gif", ms=100, hold_last_ms=2600)


if __name__ == "__main__":
    main()
