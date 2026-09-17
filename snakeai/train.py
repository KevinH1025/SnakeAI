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
from typing import Any, TextIO

import numpy as np
import torch

from .agent import DQNAgent, epsilon_at
from .config import Config, ConfigError, add_config_args, config_from_args, save_json, to_dict
from .device import assert_kernels_available, describe, resolve_device, resolve_num_envs
from .evaluate import evaluate
from .vecenv import VecSnakeEnv

# One row of metrics.csv, in order. plot.py keeps its own copy of this list because importing
# it from here would drag torch into a plotting script. A test checks the two stay identical.
METRIC_COLUMNS = ["step", "episodes", "epsilon", "loss", "return_mean",
                  "score_mean", "score_max", "best_score", "steps_per_sec", "wall_s"]


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


class Schedule:
    """A repeating due step: due(step) is True the first time `step` reaches each multiple."""

    def __init__(self, every: int, start_step: int) -> None:
        self.every = every # moves played between one due step and the next
        self.next = ((start_step // every) + 1) * every # first multiple after the start

    def due(self, step: int) -> bool:
        if step < self.next:
            return False # not there yet, nothing to do this iteration
        while self.next <= step: # skip ahead; never crawl one interval per iteration
            self.next += self.every
        return True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_metrics_file(path: Path, resume: bool, quiet: bool) -> bool:
    """Decide whether metrics.csv needs a fresh header, moving an unusable one aside first.

    Two ways an existing file cannot be appended to. A fresh run would stack its rows on top of
    the previous run's, so the step column restarts partway down. An older run's header is one
    column narrower than the rows written today. csv.DictReader then silently reads every
    later column shifted by one. Either way the old file is renamed rather than deleted.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True # nothing there yet, or a run that died before its first write

    with path.open(newline="") as f:
        header = next(csv.reader(f), [])

    if resume and header == METRIC_COLUMNS:
        return False # same layout, same run, so carry on appending

    spare = _free_path(path)
    path.rename(spare)
    if not quiet:
        why = "header is from an older version" if resume else "belongs to an earlier run"
        print(f"metrics  : {path.name} {why}, moved to {spare.name}")

    return True


def _open_metrics_csv(path: Path, resume: bool, quiet: bool) -> tuple[TextIO, Any]:
    """Open metrics.csv for appending, writing the header row when the file starts empty."""
    needs_header = _prepare_metrics_file(path, resume=resume, quiet=quiet)
    metrics_file = path.open("a", newline="") # stays open for the rest of the run
    writer = csv.writer(metrics_file)
    if needs_header:
        writer.writerow(METRIC_COLUMNS) # first row names the columns
    return metrics_file, writer


def _free_path(path: Path) -> Path:
    """`path` with .1, .2, ... appended, picking the first name nothing is using."""
    n = 1
    while True:
        spare = path.with_name(f"{path.name}.{n}")
        if not spare.exists():
            return spare
        n += 1


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
        saved = torch.load(replay, map_location=device, weights_only=False) # the buffer file
        agent.buffer.load_state_dict(saved["buffer"])

    return int(state.get("step", 0)), float(state.get("best_score", float("-inf")))


def _print_header(cfg: Config, device: torch.device, n_envs: int, agent: DQNAgent,
                  run_dir: Path, step: int, resume: str | None) -> None:
    """The four lines printed before the first move: hardware, sizes and where output lands."""
    replay_mb = agent.buffer.nbytes / 1e6 # replay buffer size on the device
    resumed = f"  (resuming from {step:,})" if resume else "" # blank for a fresh run

    print(f"device   : {describe(device)}")
    print(f"envs     : {n_envs} parallel  |  replay {replay_mb:.1f} MB on {device.type}")
    print(f"run dir  : {run_dir}")
    print(f"steps    : {cfg.train.total_steps:,}{resumed}")


def run_training(cfg: Config, resume: str | None = None, quiet: bool = False) -> dict:
    device = resolve_device(cfg.train.device)
    assert_kernels_available(device)
    n_envs = resolve_num_envs(cfg.train.num_envs, device)
    seed_everything(cfg.train.seed)

    run_dir = Path(cfg.train.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config.json")

    # Everything this run writes, named once so the loop below stays short.
    ckpt_path = run_dir / "ckpt.pt" # the periodic checkpoint, overwritten in place
    best_path = run_dir / "best.pt" # weights from the best scoring game so far
    eval_log = run_dir / "eval.jsonl" # one json object appended per evaluation
    metrics_path = run_dir / "metrics.csv" # one row per log interval

    vec = VecSnakeEnv(cfg.env, n_envs=n_envs, seed=cfg.train.seed) # the parallel games
    agent = DQNAgent(cfg.agent, device, seed=cfg.train.seed) # nets + replay, all on `device`

    step = 0 # moves played so far, carried over by a resume
    best_score = float("-inf") # highest score seen, live games included
    if resume: # continue an earlier run instead of starting fresh
        step, best_score = load_checkpoint(Path(resume), agent, device)

    if not quiet:
        _print_header(cfg, device, n_envs, agent, run_dir, step, resume)

    metrics_file, writer = _open_metrics_csv(metrics_path, bool(resume), quiet)

    if not quiet:
        print(f"updates  : {cfg.agent.updates_per_iter} x batch {cfg.agent.batch_size:,} "
              f"after every {n_envs} moves played")

    loss_window = Window(200) # loss is per update, so a window still makes sense here
    return_window = Window(200) # episode returns, only updated when a game ends
    reasons: Counter[str] = Counter() # how each finished episode ended
    episodes = 0 # games that have ended since this call started

    # When the next log, eval, save and target sync fall due, counted from this run's start.
    log_schedule = Schedule(cfg.train.log_every, step) # writes a metrics.csv row
    eval_schedule = Schedule(cfg.train.eval_every, step) # runs a greedy evaluation
    save_schedule = Schedule(cfg.train.save_every, step) # writes ckpt.pt
    sync_schedule = Schedule(cfg.agent.target_sync_steps, step) # refreshes the target net

    start_step = step # the step this call began at, so a resume is not counted twice
    started = time.perf_counter()
    last_report_step = step # where the current steps per second window starts
    last_report_time = started

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
        #    later. `terminated` only, never `truncated`: running out of hunger clock is us
        #    giving up on the episode, not the snake dying, so that future still has value.
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
            return_window.add(info["return"])
            reasons[info["reason"]] += 1 # wall / self / starved / won histogram

        # The record has to include games that are still running. A good agent plays very long
        # episodes, so waiting for one to end means the best games are never counted at all.
        top_live = vec.best_live_score()
        if top_live > best_score:
            best_score = top_live # new record, snapshot the weights
            save_checkpoint(best_path, agent, cfg, step, best_score, include_buffer=False)

        # 5. Learn. A fixed number of gradient steps after every round of moves.
        for _ in range(cfg.agent.updates_per_iter):
            loss = agent.learn()
            if loss is not None:
                loss_window.add(loss) # None until the buffer has enough to sample

        # 6. Refresh the frozen target network every target_sync_steps moves played. Counted in
        #    moves, not updates, so changing num_envs cannot quietly change how stale it gets.
        if sync_schedule.due(step):
            agent.sync_target()

        if log_schedule.due(step):
            now = time.perf_counter()
            steps_done = step - last_report_step # moves played since the previous log
            seconds = max(now - last_report_time, 1e-9) # never divide by a zero interval
            sps = steps_done / seconds # rate since the last log, not since the start
            live_mean, live_max = vec.live_scores() # all n_envs games as they stand now

            writer.writerow([ # one value per METRIC_COLUMNS entry, same order
                step,
                episodes,
                round(epsilon, 5),
                round(loss_window.mean, 6),
                round(return_window.mean, 4),
                round(live_mean, 4),
                round(live_max, 1),
                round(best_score, 1),
                round(sps, 1),
                round(now - started, 2),
            ])
            metrics_file.flush() # each row hits disk as it is written

            if not quiet:
                print(f"  step {step:>9,}  eps {epsilon:.3f}  loss {loss_window.mean:8.5f}  "
                      f"score {live_mean:6.1f}  top {live_max:5.0f}  best {best_score:5.0f}  "
                      f"eps_done {episodes:>6,}  {sps:>9,.0f} steps/s")

            last_report_step = step # baseline for the next interval's rate
            last_report_time = now

        if eval_schedule.due(step):
            result = evaluate(agent, cfg)
            record = {"step": step, **result} # the evaluation, tagged with when it ran
            eval_log.open("a").write(json.dumps(record) + "\n")
            if not quiet:
                print(f"  [eval @ {step:,}] score {result['score_mean']:.2f} "
                      f"(max {result['score_max']:.0f})  {result['reasons']}")

        if save_schedule.due(step):
            save_checkpoint(ckpt_path, agent, cfg, step, best_score,
                            include_buffer=cfg.train.save_buffer)

    save_checkpoint(ckpt_path, agent, cfg, step, best_score, # final save on the way out
                    include_buffer=cfg.train.save_buffer)
    metrics_file.close()
    elapsed = time.perf_counter() - started

    session_steps = step - start_step # moves played by this call, ignoring a resume
    final_live_mean, _ = vec.live_scores() # mean over the games still in progress

    summary = {
        "steps": step,
        "episodes": episodes,
        "best_score": best_score,
        "score_mean_live": final_live_mean,
        "elapsed_s": round(elapsed, 2),
        "steps_per_sec": round(session_steps / max(elapsed, 1e-9), 1),
        "reasons": dict(reasons),
        "run_dir": str(run_dir),
        "device": device.type,
        "n_envs": n_envs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if not quiet:
        print(f"\ndone: {step:,} steps ({session_steps:,} this session), "
              f"{episodes:,} episodes, best {best_score:.0f}, "
              f"{summary['steps_per_sec']:,.0f} steps/s over {elapsed:.1f}s")
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
