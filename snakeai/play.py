"""Watch a trained agent play or play Snake yourself.

Runs as its own process, so watching costs a running training job nothing. With --follow it
re-reads the checkpoint as the trainer writes new ones and the next game uses the new weights.

Defaults to CPU, which is faster than the GPU for one observation at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pygame
import torch

from .agent import DQNAgent
from .config import Config, from_dict, preset
from .device import resolve_device
from .env import DOWN, LEFT, RIGHT, SnakeEnv, UP, turn

FONT_PATH = Path(__file__).resolve().parent.parent / "assets" / "Arial.ttf"

BG = (12, 12, 16)
GRID = (28, 28, 36)
SNAKE_HEAD = (80, 255, 120)
SNAKE_BODY = (40, 170, 80)
FOOD = (255, 70, 70)
CRASH = (255, 200, 0)
TEXT = (235, 235, 240)
DIM = (140, 140, 155)

# which compass direction each key asks for
KEY_HEADING = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
}


def load_fonts() -> tuple[pygame.font.Font, pygame.font.Font]:
    """The bundled Arial, falling back to any monospace font pygame can find."""
    try:
        font = pygame.font.Font(str(FONT_PATH), 18) # headline size
        small = pygame.font.Font(str(FONT_PATH), 14) # detail lines
    except (OSError, FileNotFoundError):
        font = pygame.font.SysFont("monospace", 18) # asset missing, use a system face
        small = pygame.font.SysFont("monospace", 14)

    return font, small


class Renderer:
    """The pygame window: the board as a grid of cells with a HUD strip underneath."""

    def __init__(self, cfg: Config, cell: int = 24, hud_h: int = 96) -> None:
        self.cfg = cfg
        self.cell = cell # pixels per board square
        self.hud_h = hud_h # pixels of text strip below the board

        pygame.init()
        self.w = cfg.env.grid_w * cell
        self.h = cfg.env.grid_h * cell + hud_h
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("SnakeAI")

        self.font, self.small = load_fonts()

        self.clock = pygame.time.Clock() # paces the frames

    def draw(self, env: SnakeEnv, lines: list[str]) -> None:
        self.screen.fill(BG) # wipe last frame
        self._draw_grid(env)
        self._draw_food(env)
        self._draw_snake(env)
        self._draw_crash(env)
        self._draw_hud(env, lines)
        pygame.display.flip() # show the frame

    def _draw_grid(self, env: SnakeEnv) -> None:
        cell = self.cell
        width = env.cfg.grid_w * cell
        height = env.cfg.grid_h * cell

        for x in range(env.cfg.grid_w + 1):
            pygame.draw.line(self.screen, GRID, (x * cell, 0), (x * cell, height)) # verticals
        for y in range(env.cfg.grid_h + 1):
            pygame.draw.line(self.screen, GRID, (0, y * cell), (width, y * cell)) # horizontals

    def _draw_food(self, env: SnakeEnv) -> None:
        if env.food is None:
            return # board is full, nothing to draw

        cell = self.cell
        fx, fy = env.food
        rect = (fx * cell + 3, fy * cell + 3, cell - 6, cell - 6) # inset so it reads as a pip
        pygame.draw.rect(self.screen, FOOD, rect, border_radius=4)

    def _draw_snake(self, env: SnakeEnv) -> None:
        cell = self.cell

        for i, (x, y) in enumerate(env.snake):
            colour = SNAKE_HEAD if i == 0 else SNAKE_BODY # the head is the bright one
            rect = (x * cell + 1, y * cell + 1, cell - 2, cell - 2) # 1px gap between segments
            pygame.draw.rect(self.screen, colour, rect, border_radius=3)

    def _draw_crash(self, env: SnakeEnv) -> None:
        if env.crash_cell is None or not env.terminated:
            return # nothing to mark

        cx, cy = env.crash_cell
        if not (0 <= cx < env.cfg.grid_w and 0 <= cy < env.cfg.grid_h):
            return # died off the board

        cell = self.cell
        rect = (cx * cell + 1, cy * cell + 1, cell - 2, cell - 2)
        pygame.draw.rect(self.screen, CRASH, rect, 3) # outline only, 3px border

    def _draw_hud(self, env: SnakeEnv, lines: list[str]) -> None:
        top = env.cfg.grid_h * self.cell + 8 # HUD strip starts just under the board

        for i, line in enumerate(lines):
            font = self.font if i == 0 else self.small # first line is the headline
            colour = TEXT if i == 0 else DIM # the rest are dimmed detail
            surface = font.render(line, True, colour)
            self.screen.blit(surface, (10, top + i * 20)) # 20px between HUD lines

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


def reload_if_newer(checkpoint: Path, device: torch.device, mtime: float,
                    agent: DQNAgent, cfg: Config,
                    step: int) -> tuple[DQNAgent, Config, int, float]:
    """Swap in new weights when the trainer has written a newer checkpoint."""
    try:
        current = checkpoint.stat().st_mtime # has the trainer written a new one?
        if current != mtime:
            agent, cfg, step = load_agent(checkpoint, device) # hot reload the weights
            mtime = current
    except OSError:
        pass # caught it mid write; just try again next episode

    return agent, cfg, step, mtime


def hunger_line(env: SnakeEnv, cfg: Config) -> str:
    """The bottom HUD line: what just happened and how close the snake is to starving."""
    limit = cfg.env.max_steps_without_food # moves allowed between meals

    return f"last {env.last_event.value}   hunger {env.steps_since_food}/{limit}"


def watch(checkpoint: Path, fps: int, follow: bool, device_spec: str,
          seed: int | None) -> None:
    device = resolve_device(device_spec)
    agent, cfg, step = load_agent(checkpoint, device)
    mtime = checkpoint.stat().st_mtime
    renderer = Renderer(cfg)
    env = SnakeEnv(cfg.env, seed=seed)
    env.reset(seed=seed)

    episodes = 0
    best = 0 # best score seen
    total = 0 # scores summed, for the mean
    running = True

    while running:
        if not renderer.pump(): # closed the window / pressed q
            break # leave NOW: don't spend another frame or a checkpoint reload, first
        if env.done:
            episodes += 1
            total += env.score
            best = max(best, env.score)

            if follow:
                agent, cfg, step, mtime = reload_if_newer(checkpoint, device, mtime,
                                                          agent, cfg, step)

            env.reset()

        action = int(agent.act(env.observe(), epsilon=0.0)[0]) # batch of 1, greedy
        env.step(action)

        mean_text = f"    mean {total / episodes:.1f}" if episodes else "" # none on game 1
        follow_text = "   [following]" if follow else "" # shown only with --follow
        scores = f"score {env.score}    best {best}    episodes {episodes}{mean_text}"
        counts = f"device {device.type}   len {len(env.snake)}" # where it runs, how long it is
        status = f"checkpoint step {step:,}   {counts}{follow_text}"

        renderer.draw(env, [scores, status, hunger_line(env, cfg)])
        renderer.clock.tick(fps)

    pygame.quit()


def poll_steering(wanted: int) -> tuple[int, bool]:
    """Drain the pygame events. Returns the direction asked for and False once quitting."""
    running = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False # window closed
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False # q or escape quits
            elif event.key in KEY_HEADING:
                wanted = KEY_HEADING[event.key] # steer toward this compass direction

    return wanted, running


def action_towards(heading: int, wanted: int) -> int:
    """The egocentric action whose resulting heading matches `wanted`; a 180 is ignored."""
    for candidate in (0, 1, 2):
        if turn(heading, candidate) == wanted: # this turn produces the heading we want
            return candidate

    return 0 # the key asks for a 180, so keep going straight


def play_human(cfg: Config, fps: int, seed: int | None) -> None:
    """Arrow keys / WASD. Absolute input becomes the egocentric action the env expects."""
    renderer = Renderer(cfg)

    env = SnakeEnv(cfg.env, seed=seed)
    env.reset(seed=seed)

    wanted = env.heading # the direction the player last asked for
    best = 0
    episodes = 0
    running = True

    while running:
        wanted, running = poll_steering(wanted)
        if not running:
            break

        if env.done:
            episodes += 1
            best = max(best, env.score)
            env.reset()
            wanted = env.heading # the fresh snake faces a random way

        action = action_towards(env.heading, wanted) # the turn that heads where the key asked
        env.step(action)

        renderer.draw(env, [
            f"score {env.score}    best {best}    episodes {episodes}",
            "arrows / wasd to steer, q to quit",
            hunger_line(env, cfg),
        ])
        renderer.clock.tick(fps)

    pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    """Every command line flag the viewer takes."""
    description = "Watch a trained SnakeAI agent or play yourself."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    parser.add_argument("--follow", action="store_true",
                        help="reload the checkpoint as training writes it")
    parser.add_argument("--human", action="store_true", help="play it yourself")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"),
                        help="cpu is faster than cuda for one observation at a time")
    parser.add_argument("--preset", default="default", help="only used with --human")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.human:
        play_human(preset(args.preset), args.fps, args.seed)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --human is given") # nothing to watch
    path = Path(args.checkpoint)
    if not path.exists():
        parser.error(f"no such checkpoint: {path}") # the path given does not exist

    watch(path, args.fps, args.follow, args.device, args.seed)


if __name__ == "__main__":
    main()
