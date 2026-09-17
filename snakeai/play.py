"""Watch a trained agent play or play Snake yourself.

Runs as its own process, so watching costs a running training job nothing. With --follow it
re-reads the checkpoint as the trainer writes new ones and the next game uses the new weights.

Defaults to CPU, which is faster than the GPU for one observation at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pygame
import torch

from .agent import DQNAgent
from .config import Config, from_dict
from .device import resolve_device
from .env import DIRECTIONS, SnakeEnv, turn

FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "Arial.ttf"

BG = (12, 12, 16)
GRID = (28, 28, 36)
SNAKE_HEAD = (80, 255, 120)
SNAKE_BODY = (40, 170, 80)
FOOD = (255, 70, 70)
CRASH = (255, 200, 0)
TEXT = (235, 235, 240)
DIM = (140, 140, 155)


class Renderer:
    def __init__(self, cfg: Config, cell: int = 24, hud_h: int = 96) -> None:
        self.cfg = cfg
        self.cell = cell
        self.hud_h = hud_h
        pygame.init()
        self.w = cfg.env.grid_w * cell
        self.h = cfg.env.grid_h * cell + hud_h
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("SnakeAI")
        try:
            self.font = pygame.font.Font(str(FONT_PATH), 18)
            self.small = pygame.font.Font(str(FONT_PATH), 14)
        except (OSError, FileNotFoundError):
            self.font = pygame.font.SysFont("monospace", 18)
            self.small = pygame.font.SysFont("monospace", 14)
        self.clock = pygame.time.Clock()

    def draw(self, env: SnakeEnv, lines: list[str]) -> None:
        c = self.cell
        self.screen.fill(BG)
        for x in range(env.cfg.grid_w + 1):
            pygame.draw.line(self.screen, GRID, (x * c, 0), (x * c, env.cfg.grid_h * c))
        for y in range(env.cfg.grid_h + 1):
            pygame.draw.line(self.screen, GRID, (0, y * c), (env.cfg.grid_w * c, y * c))

        if env.food is not None:
            fx, fy = env.food
            pygame.draw.rect(self.screen, FOOD, (fx * c + 3, fy * c + 3, c - 6, c - 6), border_radius=4)
        for i, (x, y) in enumerate(env.snake):
            colour = SNAKE_HEAD if i == 0 else SNAKE_BODY
            pygame.draw.rect(self.screen, colour, (x * c + 1, y * c + 1, c - 2, c - 2), border_radius=3)
        if env.crash_cell is not None and env.terminated:
            cx, cy = env.crash_cell
            if 0 <= cx < env.cfg.grid_w and 0 <= cy < env.cfg.grid_h:
                pygame.draw.rect(self.screen, CRASH, (cx * c + 1, cy * c + 1, c - 2, c - 2), 3)

        y0 = env.cfg.grid_h * c + 8
        for i, line in enumerate(lines):
            surface = (self.font if i == 0 else self.small).render(line, True, TEXT if i == 0 else DIM)
            self.screen.blit(surface, (10, y0 + i * 20))
        pygame.display.flip()

    def pump(self) -> bool:
        """Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
        return True


def load_agent(path: Path, device: torch.device) -> tuple[DQNAgent, Config, int]:
    """Load just the weights. A viewer never learns, so it needs no replay buffer."""
    state = torch.load(path, map_location=device, weights_only=False)

    if isinstance(state["config"], dict):
        cfg = from_dict(state["config"])
    else:
        cfg = state["config"]

    viewer_cfg = dataclasses.replace(cfg.agent, buffer_capacity=1) # smallest allowed
    agent = DQNAgent(viewer_cfg, device, seed=0)

    weights = {}
    for key, value in state["agent"].items():
        if key != "buffer": # skip it, an old checkpoint may still have one bundled in
            weights[key] = value

    agent.load_state_dict(weights)

    return agent, cfg, int(state.get("step", 0))


def watch(checkpoint: Path, fps: int, follow: bool, device_spec: str, seed: int | None) -> None:
    device = resolve_device(device_spec)
    agent, cfg, step = load_agent(checkpoint, device)
    mtime = checkpoint.stat().st_mtime
    renderer = Renderer(cfg)
    env = SnakeEnv(cfg.env, seed=seed)
    env.reset(seed=seed)

    episodes, best, total = 0, 0, 0
    running = True
    while running:
        if not renderer.pump(): # closed the window / pressed q
            break # leave NOW: don't spend another frame or a checkpoint reload, first
        if env.done:
            episodes += 1
            total += env.score
            best = max(best, env.score)
            if follow:
                try:
                    current = checkpoint.stat().st_mtime # has the trainer written a new one?
                    if current != mtime:
                        agent, cfg, step = load_agent(checkpoint, device) # hot-reload the weights
                        mtime = current
                except OSError:
                    pass # caught it mid-write; just try again next episode
            env.reset()

        action = int(agent.act(env.observe(), epsilon=0.0)[0]) # batch of 1, greedy
        env.step(action)
        renderer.draw(env, [
            f"score {env.score}    best {best}    episodes {episodes}"
            + (f"    mean {total / episodes:.1f}" if episodes else ""),
            f"checkpoint step {step:,}   device {device.type}   len {len(env.snake)}"
            + ("   [following]" if follow else ""),
            f"last {env.last_event.value}   hunger {env.steps_since_food}/{cfg.env.max_steps_without_food}",
        ])
        renderer.clock.tick(fps)
    pygame.quit()


def play_human(cfg: Config, fps: int, seed: int | None) -> None:
    """Arrow keys / WASD. Absolute input is converted to the egocentric action the env expects."""
    renderer = Renderer(cfg)
    env = SnakeEnv(cfg.env, seed=seed)
    env.reset(seed=seed)
    key_dir = {
        pygame.K_UP: 0, pygame.K_w: 0, pygame.K_RIGHT: 1, pygame.K_d: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2, pygame.K_LEFT: 3, pygame.K_a: 3,
    }
    wanted = env.heading
    best, episodes = 0, 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in key_dir:
                    wanted = key_dir[event.key]
        if not running:
            break
        if env.done:
            episodes += 1
            best = max(best, env.score)
            env.reset()
            wanted = env.heading

        # Pick the egocentric action whose resulting heading matches the key; a 180 is ignored.
        action = 0 # default to straight if the key asks for a 180
        for candidate in (0, 1, 2):
            if turn(env.heading, candidate) == wanted: # this turn produces the heading we want
                action = candidate
                break
        env.step(action)
        renderer.draw(env, [
            f"score {env.score}    best {best}    episodes {episodes}",
            "arrows / wasd to steer, q to quit",
            f"last {env.last_event.value}   hunger {env.steps_since_food}/{cfg.env.max_steps_without_food}",
        ])
        renderer.clock.tick(fps)
    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch a trained SnakeAI agent or play yourself.")
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    parser.add_argument("--follow", action="store_true", help="reload the checkpoint as training writes it")
    parser.add_argument("--human", action="store_true", help="play it yourself")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"),
                        help="cpu is faster than cuda for one observation at a time")
    parser.add_argument("--preset", default="default", help="only used with --human")
    args = parser.parse_args(argv)

    if args.human:
        from .config import preset as get_preset
        play_human(get_preset(args.preset), args.fps, args.seed)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --human is given")
    path = Path(args.checkpoint)
    if not path.exists():
        parser.error(f"no such checkpoint: {path}")
    watch(path, args.fps, args.follow, args.device, args.seed)


if __name__ == "__main__":
    main()
