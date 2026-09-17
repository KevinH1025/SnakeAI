"""The training loop. No window, no frame rate, nothing that waits on a human.

Metrics go to metrics.csv, drawing happens in play.py and plotting in plot.py, so this module
never imports pygame or matplotlib. A test checks that.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .agent import DQNAgent, epsilon_at
from .config import Config, ConfigError, add_config_args, config_from_args, save_json, to_dict
from .device import assert_kernels_available, describe, resolve_device, resolve_num_envs
from .evaluate import evaluate
from .vecenv import VecSnakeEnv


class Window:
    """The average of the last `n` values added."""

    def __init__(self, n: int) -> None:
        self.values: deque[float] = deque(maxlen=n)

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return float(sum(self.values) / len(self.values)) if self.values else 0.0

    def __len__(self) -> int:
        return len(self.values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save(obj, path: Path) -> None:
    """Write via a .tmp then rename: a crash mid-save must not destroy the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def replay_path(path: Path) -> Path:
    return path.with_suffix(".replay.pt") # sits next to ckpt.pt, e.g. runs/v2/ckpt.replay.pt


def save_checkpoint(path: Path, agent: DQNAgent, cfg: Config, step: int,
                    best_score: float, include_buffer: bool) -> None:
    """Weights go in `path`; the replay buffer goes in a separate file beside it.

    The buffer is hundreds of times bigger than the weights and only resume needs it.
    """
    _atomic_save(
        {
            "agent": agent.state_dict(include_buffer=False), # weights + optimizer only
            "config": to_dict(cfg),
            "step": step,
            "best_score": best_score,
        },
        path,
    )
    if include_buffer:
        _atomic_save({"buffer": agent.buffer.state_dict()}, replay_path(path))


def load_checkpoint(path: Path, agent: DQNAgent, device: torch.device) -> tuple[int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(state["agent"])
    replay = replay_path(path)
    if replay.exists(): # written separately; absent if the run used save_buffer=false
        agent.buffer.load_state_dict(torch.load(replay, map_location=device, weights_only=False)["buffer"])
    return int(state.get("step", 0)), float(state.get("best_score", float("-inf")))


def run_training(cfg: Config, resume: str | None = None, quiet: bool = False) -> dict:
    device = resolve_device(cfg.train.device)
    assert_kernels_available(device)
    n_envs = resolve_num_envs(cfg.train.num_envs, device)
    seed_everything(cfg.train.seed)

    run_dir = Path(cfg.train.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    vec = VecSnakeEnv(cfg.env, n_envs=n_envs, seed=cfg.train.seed) # the parallel games
    agent = DQNAgent(cfg.agent, device, seed=cfg.train.seed) # nets + replay, all on `device`

    step, best_score = 0, float("-inf")
    if resume: # continue an earlier run instead of starting fresh
        step, best_score = load_checkpoint(Path(resume), agent, device)

    if not quiet:
        print(f"device   : {describe(device)}")
        print(f"envs     : {n_envs} parallel  |  replay {agent.buffer.nbytes / 1e6:.1f} MB on {device.type}")
        print(f"run dir  : {run_dir}")
        print(f"steps    : {cfg.train.total_steps:,}" + (f"  (resuming from {step:,})" if resume else ""))

    metrics_path = run_dir / "metrics.csv"
    # A 0-byte file is left behind by a run that died before its first write and still needs
    # a header.
    new_file = not metrics_path.exists() or metrics_path.stat().st_size == 0
    metrics_file = metrics_path.open("a", newline="")
    writer = csv.writer(metrics_file)
    if new_file:
        writer.writerow(["step", "episodes", "epsilon", "loss", "return_mean",
                         "score_mean", "score_max", "best_score", "steps_per_sec", "wall_s"])

    # How to spend the learning we owe: a few big gradient steps rather than many small ones,
    # which is much faster on a GPU and adds up to the same amount of learning.
    def update_plan(owed: int) -> tuple[int, int]:
        n_up = min(cfg.agent.max_updates_per_iter, owed)
        return n_up, max(cfg.agent.batch_size, int(round(owed * cfg.agent.batch_size / n_up)))

    planned_updates, planned_batch = update_plan(max(1, n_envs // cfg.agent.train_every))
    if planned_batch > cfg.agent.buffer_capacity:
        raise ValueError(
            f"effective batch {planned_batch} (batch_size={cfg.agent.batch_size} scaled for "
            f"num_envs={n_envs}) exceeds agent.buffer_capacity={cfg.agent.buffer_capacity}"
        )
    if not quiet:
        print(f"updates  : {planned_updates} x batch {planned_batch} per iteration "
              f"({cfg.agent.batch_size / cfg.agent.train_every:.0f} gradient samples per transition)")

    loss_w = Window(200) # loss is per update, so a window still makes sense here
    ret_w = Window(200) # episode returns, only updated when a game ends
    reasons: Counter[str] = Counter()
    episodes = 0
    since_update = 0
    # When the next log / eval / save is due, counted from where this run actually starts.
    def next_due(interval: int) -> int:
        return ((step // interval) + 1) * interval # first multiple strictly after `step`

    next_log, next_eval = next_due(cfg.train.log_every), next_due(cfg.train.eval_every)
    next_save = next_due(cfg.train.save_every)
    start_step = step
    started = time.perf_counter()
    last_report_step, last_report_time = step, started

    while step < cfg.train.total_steps:
        epsilon = epsilon_at(step, cfg.agent) # how random this step's actions are

        # 1. Remember what every game looks like right now. copy=True because vec.step() below
        #    overwrites current_obs in place and replay needs the BEFORE picture.
        obs_t = torch.from_numpy(vec.current_obs).to(device, copy=True)

        # 2. Ask the network for a move for every game at once, in one batched forward pass.
        actions = agent.act(vec.current_obs, epsilon)

        # 3. Play those moves. Every game advances exactly one cell.
        next_obs, rewards, terminated, truncated, finished = vec.step(actions)

        # 4. File all n_envs moves away as (before, action, reward, after, died) to learn from
        #    later. `terminated` only, never `truncated`: running out of hunger clock is us giving
        #    up on the episode, not the snake dying, so that future still has value.
        agent.buffer.add_batch(
            obs_t,
            torch.from_numpy(actions).to(device),
            torch.from_numpy(rewards).to(device),
            torch.from_numpy(next_obs).to(device, copy=True),
            torch.from_numpy(terminated).to(device),
        )

        step += n_envs # one transition per env per iteration

        for info in finished: # episodes that ended on this step
            episodes += 1
            ret_w.add(info["return"])
            reasons[info["reason"]] += 1 # wall / self / starved / won histogram

        # The record has to include games that are still running. A good agent plays very long
        # episodes, so waiting for one to end means the best games are never counted at all.
        top_live = vec.best_live_score()
        if top_live > best_score:
            best_score = top_live # new record, snapshot the weights
            save_checkpoint(run_dir / "best.pt", agent, cfg, step, best_score, include_buffer=False)

        # 5. Learn. We owe one update's worth of learning per `train_every` moves collected and
        #    n_envs moves just arrived, so the debt builds fast. update_plan() decides how to spend
        #    it: a few big gradient steps instead of many small ones.
        since_update += n_envs
        owed = since_update // cfg.agent.train_every # how much learning we now owe
        if owed > 0:
            n_up, eff_batch = update_plan(int(owed)) # cap the count, grow the batch instead
            for _ in range(n_up):
                loss = agent.learn(batch_size=eff_batch)
                if loss is not None:
                    loss_w.add(loss) # None until the buffer has enough to sample
            since_update -= owed * cfg.agent.train_every # keep the remainder, never drop it

        if step >= next_log:
            now = time.perf_counter()
            sps = (step - last_report_step) / max(now - last_report_time, 1e-9) # since last log, not since start
            live_mean, live_max = vec.live_scores() # all n_envs games as they stand now
            writer.writerow([step, episodes, round(epsilon, 5), round(loss_w.mean, 6),
                             round(ret_w.mean, 4), round(live_mean, 4), round(live_max, 1),
                             round(best_score, 1), round(sps, 1), round(now - started, 2)])
            metrics_file.flush()

            if not quiet:
                print(f"  step {step:>9,}  eps {epsilon:.3f}  loss {loss_w.mean:8.5f}  "
                      f"score {live_mean:6.1f}  top {live_max:5.0f}  best {best_score:5.0f}  "
                      f"eps_done {episodes:>6,}  {sps:>9,.0f} steps/s")
            last_report_step, last_report_time = step, now
            while next_log <= step:  # skip ahead; never crawl one interval per iteration
                next_log += cfg.train.log_every

        if step >= next_eval:
            result = evaluate(agent, cfg)
            (run_dir / "eval.jsonl").open("a").write(json.dumps({"step": step, **result}) + "\n")
            if not quiet:
                print(f"  [eval @ {step:,}] score {result['score_mean']:.2f} "
                      f"(max {result['score_max']:.0f})  {result['reasons']}")
            while next_eval <= step:  # skip ahead; never crawl one interval per iteration
                next_eval += cfg.train.eval_every

        if step >= next_save:
            save_checkpoint(run_dir / "ckpt.pt", agent, cfg, step, best_score,
                            include_buffer=cfg.train.save_buffer)
            while next_save <= step:  # skip ahead; never crawl one interval per iteration
                next_save += cfg.train.save_every

    save_checkpoint(run_dir / "ckpt.pt", agent, cfg, step, best_score, include_buffer=cfg.train.save_buffer)
    metrics_file.close()
    elapsed = time.perf_counter() - started

    summary = {
        "steps": step,
        "episodes": episodes,
        "best_score": best_score,
        "score_mean_live": vec.live_scores()[0],
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round((step - start_step) / max(elapsed, 1e-9), 1),
        "reasons": dict(reasons),
        "run_dir": str(run_dir),
        "device": device.type,
        "n_envs": n_envs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not quiet:
        print(f"\ndone: {step:,} steps ({step - start_step:,} this session), {episodes:,} episodes, "
              f"best {best_score:.0f}, {summary['steps_per_sec']:,.0f} steps/s over {elapsed:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train SnakeAI (headless).")
    add_config_args(parser)
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ConfigError as exc:
        parser.error(str(exc))
    run_training(cfg, resume=args.resume, quiet=args.quiet)


if __name__ == "__main__":
    main()
