"""Greedy evaluation on a fixed set of seeds.

Every evaluation replays the *same* seeds, so two checkpoints face identical food sequences and
a score difference between them means something. Exploration is off (epsilon = 0) and the
episode outcome histogram is reported because "mean score 12" hides the difference between an
agent that starves and one that crashes into itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent
from .config import (
    Config, add_config_args, apply_overrides, from_dict, load_json, preset,
)
from .device import assert_kernels_available, describe, resolve_device
from .vecenv import VecSnakeEnv


@torch.no_grad()
def evaluate(agent: DQNAgent, cfg: Config, episodes: int | None = None,
             seed: int | None = None, max_steps: int | None = None) -> dict:
    """Run ``episodes`` greedy episodes in parallel and summarise them.

    Episodes still running at ``max_steps`` count as unfinished and are left out of the
    averages.
    """
    episodes = episodes if episodes is not None else cfg.train.eval_episodes
    seed = seed if seed is not None else cfg.train.eval_seed
    max_steps = max_steps if max_steps is not None else cfg.train.eval_max_steps

    vec = VecSnakeEnv(cfg.env, n_envs=episodes, seed=seed) # one env per episode, in parallel
    outcomes = _play_episodes(agent, vec, episodes, max_steps) # greedy rollout, no exploration
    return _summarise(outcomes, episodes) # averages over the episodes that finished


class _Outcomes:
    """One slot per parallel episode, filled in as episodes finish."""

    def __init__(self, episodes: int) -> None:
        self.done = np.zeros(episodes, dtype=bool) # which envs finished their first episode
        self.scores = np.zeros(episodes, dtype=np.float64) # food eaten in that episode
        self.returns = np.zeros(episodes, dtype=np.float64) # summed reward for that episode
        self.lengths = np.zeros(episodes, dtype=np.int64) # snake length when it ended
        self.steps = np.zeros(episodes, dtype=np.int64) # steps that episode lasted
        self.reasons: list[str] = ["unfinished"] * episodes # how each episode ended

    def record(self, info: dict) -> None:
        """Store one finished episode, ignoring anything an env plays after its first."""
        i = info["env"] # which env this info dict came from

        if self.done[i]:
            return # only the FIRST episode of each game counts

        self.done[i] = True # this env's result is locked in now
        self.scores[i] = info["score"]
        self.returns[i] = info["return"]
        self.lengths[i] = info["length"]
        self.steps[i] = info["episode_steps"]
        self.reasons[i] = info["reason"]


def _play_episodes(agent: DQNAgent, vec: VecSnakeEnv, episodes: int,
                   max_steps: int) -> _Outcomes:
    """Step every env greedily until each has finished one episode or max_steps runs out."""
    outcomes = _Outcomes(episodes)

    for _ in range(max_steps):
        actions = agent.act(vec.current_obs, epsilon=0.0) # greedy, no exploration during eval
        _, _, _, _, finished = vec.step(actions) # only the finished infos matter here

        for info in finished:
            outcomes.record(info) # later episodes from the same env are dropped

        if outcomes.done.all():
            break # all finished, no need to burn the rest of max_steps

    return outcomes


def _summarise(outcomes: _Outcomes, episodes: int) -> dict:
    """Average the finished episodes and count how each one ended."""
    done = outcomes.done # mask of the episodes that actually finished
    n = int(done.sum()) # how many actually finished

    def mean(values): # average over the finished episodes only
        return float(values[done].mean()) if n else 0.0

    score_max = float(outcomes.scores[done].max()) if n else 0.0 # best finished episode
    score_min = float(outcomes.scores[done].min()) if n else 0.0 # worst finished episode

    return {
        "episodes": episodes,
        "finished": n,
        "unfinished": episodes - n,
        "score_mean": mean(outcomes.scores),
        "score_max": score_max,
        "score_min": score_min,
        "return_mean": mean(outcomes.returns),
        "length_mean": mean(outcomes.lengths),
        "steps_mean": mean(outcomes.steps),
        "reasons": dict(Counter(outcomes.reasons)),
    }


def _build_parser() -> argparse.ArgumentParser:
    """The command line: the shared config flags plus the checkpoint to evaluate."""
    parser = argparse.ArgumentParser(description="Evaluate a trained SnakeAI checkpoint.")
    add_config_args(parser) # --config, --set and --device, shared with train.py
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=None)
    return parser


def _resolve_config(state: dict, args: argparse.Namespace,
                    parser: argparse.ArgumentParser) -> Config:
    """Start from the config the checkpoint was trained with, then layer the flags on top."""
    # The checkpoint's own config is the baseline. The agent was trained with it and the
    # network shape must match. Anything the user passed is then layered on top rather than
    # discarded, so `--set env.grid_w=30` genuinely evaluates on a different board.
    saved = state.get("config") # the config stored next to the weights

    if isinstance(saved, dict):
        cfg = from_dict(saved) # newer checkpoints store it as plain JSON data
    else:
        cfg = saved or preset("default") # older ones hold a Config object, or nothing at all

    try:
        if args.config:
            cfg = load_json(args.config) # an explicit --config replaces the saved one outright
        cfg = apply_overrides(cfg, args.overrides) # then every --set KEY=VALUE on top
    except (KeyError, ValueError) as exc:
        parser.error(str(exc).strip('"'))

    return cfg


def _print_result(result: dict) -> None:
    """Print the summary as one aligned key and value per line."""
    width = 0 # widest key name, so the values line up

    for key in result:
        width = max(width, len(key))

    for key, value in result.items():
        print(f"  {key:<{width}}  {value}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device or "auto") # honours --device, else picks one
    assert_kernels_available(device)

    path = Path(args.checkpoint)
    if not path.exists():
        parser.error(f"no such checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False) # weights plus config

    cfg = _resolve_config(state, args, parser)

    agent = DQNAgent(cfg.agent, device, seed=cfg.train.seed)
    agent.load_state_dict(state["agent"]) # network weights, no optimiser state needed here

    print(f"device: {describe(device)}")
    print(f"checkpoint: {path}  (step {state.get('step', '?'):,})")

    result = evaluate(agent, cfg, episodes=args.episodes)
    _print_result(result)


if __name__ == "__main__":
    main()
