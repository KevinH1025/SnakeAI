"""Greedy evaluation on a fixed set of seeds.

Every evaluation replays the *same* seeds, so two checkpoints face identical food sequences and a
score difference between them means something. Exploration is off (epsilon = 0) and the episode
outcome histogram is reported because "mean score 12" hides the difference between an agent that
starves and one that crashes into itself.
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

    Episodes still running at ``max_steps`` count as unfinished and are left out of the averages.
    """
    episodes = episodes if episodes is not None else cfg.train.eval_episodes
    seed = seed if seed is not None else cfg.train.eval_seed
    max_steps = max_steps if max_steps is not None else cfg.train.eval_max_steps

    vec = VecSnakeEnv(cfg.env, n_envs=episodes, seed=seed) # one env per episode, all in parallel
    done = np.zeros(episodes, dtype=bool) # which envs have finished their first episode
    scores = np.zeros(episodes, dtype=np.float64)
    returns = np.zeros(episodes, dtype=np.float64)
    lengths = np.zeros(episodes, dtype=np.int64)
    steps_taken = np.zeros(episodes, dtype=np.int64)
    reasons: list[str] = ["unfinished"] * episodes

    for _ in range(max_steps):
        actions = agent.act(vec.current_obs, epsilon=0.0) # greedy, no exploration during eval
        _, _, _, _, finished = vec.step(actions)

        for info in finished:
            i = info["env"]

            if done[i]:
                continue # only the FIRST episode of each game counts

            done[i] = True
            scores[i] = info["score"]
            returns[i] = info["return"]
            lengths[i] = info["length"]
            steps_taken[i] = info["episode_steps"]
            reasons[i] = info["reason"]

        if done.all():
            break # all finished, no need to burn the rest of max_steps

    n = int(done.sum()) # how many actually finished

    def mean(values): # average over the finished episodes only
        return float(values[done].mean()) if n else 0.0

    return {
        "episodes": episodes,
        "finished": n,
        "unfinished": episodes - n,
        "score_mean": mean(scores),
        "score_max": float(scores[done].max()) if n else 0.0,
        "score_min": float(scores[done].min()) if n else 0.0,
        "return_mean": mean(returns),
        "length_mean": mean(lengths),
        "steps_mean": mean(steps_taken),
        "reasons": dict(Counter(reasons)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SnakeAI checkpoint.")
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)

    device = resolve_device(args.device or "auto")
    assert_kernels_available(device)

    path = Path(args.checkpoint)
    if not path.exists():
        parser.error(f"no such checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)

    # The checkpoint's own config is the baseline. The agent was trained with it and the network
    # shape must match. Anything the user passed is then layered on top rather than discarded, so
    # `--set env.grid_w=30` genuinely evaluates on a different board.
    saved = state.get("config")
    cfg = from_dict(saved) if isinstance(saved, dict) else (saved or preset("default"))
    try:
        if args.config:
            cfg = load_json(args.config)
        cfg = apply_overrides(cfg, args.overrides)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc).strip('"'))

    agent = DQNAgent(cfg.agent, device, seed=cfg.train.seed)
    agent.load_state_dict(state["agent"])

    print(f"device: {describe(device)}")
    print(f"checkpoint: {path}  (step {state.get('step', '?'):,})")
    result = evaluate(agent, cfg, episodes=args.episodes)
    width = 0
    for key in result:
        width = max(width, len(key))

    for key, value in result.items():
        print(f"  {key:<{width}}  {value}")


if __name__ == "__main__":
    main()
