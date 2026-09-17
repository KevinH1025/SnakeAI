"""Plot a finished (or running) training run from its CSV.

Offline and out-of-process on purpose. v1 called ``plt.pause(0.1)`` from inside the game loop on
every death, which blocked training for 100 ms each episode and made matplotlib a hard dependency
of learning anything at all.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Must stay identical to snakeai.train.METRIC_COLUMNS. Copied rather than imported so that
# plotting a run never has to load torch. A test compares the two lists.
METRIC_COLUMNS = ["step", "episodes", "epsilon", "loss", "return_mean",
                  "score_mean", "score_max", "best_score", "steps_per_sec", "wall_s"]


def read_metrics(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.csv in {run_dir}")

    with path.open(newline="") as f:
        raw = list(csv.reader(f))
    if not raw:
        raise ValueError(f"{path} has no data rows yet")
    header, rows = raw[0], raw[1:]

    # Older runs wrote one column fewer, and a restart could append today's wider rows under
    # that narrower header. Matching every row to the header would then shift each later column
    # by one without complaining, so rows are matched on their own width instead. Today's layout
    # wins when any row has it, and the file's header is the fallback for everything else.
    keep = [row for row in rows if len(row) == len(METRIC_COLUMNS)]
    names = METRIC_COLUMNS
    if not keep:
        names = header
        keep = [row for row in rows if len(row) == len(header)]

    if not keep:
        raise ValueError(f"{path} has no data rows yet")

    # One file can still hold more than one run. A restart begins again from step 0, and a
    # resume replays the steps between its checkpoint and wherever the previous run stopped.
    # Reading backwards and keeping only steps that keep falling leaves exactly the rows
    # leading up to the newest one, and drops the superseded ones either way they overlap.
    latest = []
    limit = float("inf")
    for row in reversed(keep):
        step = float(row[0])
        if step < limit:
            latest.append(row)
            limit = step
    keep = latest[::-1]

    skipped = len(rows) - len(keep)
    if skipped:
        print(f"note: skipped {skipped} row(s) in {path.name}, left over from an earlier run")

    columns: dict[str, list[float]] = {}
    for i, key in enumerate(names):
        columns[key] = [float(row[i]) for row in keep]

    return columns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot a SnakeAI training run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", type=Path, default=None, help="write a PNG instead of showing a window")
    args = parser.parse_args(argv)

    import matplotlib # imported HERE, not at module scope, so train.py stays GUI-free
    if args.save:
        matplotlib.use("Agg") # headless backend when writing a file
    import matplotlib.pyplot as plt

    m = read_metrics(args.run_dir)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"SnakeAI - {args.run_dir}")

    axes[0][0].plot(m["step"], m["score_mean"], label="mean score, live games")
    if "score_max" in m:
        axes[0][0].plot(m["step"], m["score_max"], alpha=0.4, label="best live game")
    if "best_score" in m:
        axes[0][0].plot(m["step"], m["best_score"], alpha=0.4, label="best ever")
    axes[0][0].set_xlabel("environment steps")
    axes[0][0].set_ylabel("score")
    axes[0][0].legend()

    axes[0][1].plot(m["step"], m["loss"], color="tab:orange")
    axes[0][1].set_xlabel("environment steps")
    axes[0][1].set_ylabel("loss (rolling mean of last 200 updates)")
    axes[0][1].set_yscale("log")

    axes[1][0].plot(m["step"], m["epsilon"], color="tab:green")
    axes[1][0].set_xlabel("environment steps")
    axes[1][0].set_ylabel("epsilon")

    axes[1][1].plot(m["step"], m["steps_per_sec"], color="tab:purple")
    axes[1][1].set_xlabel("environment steps")
    axes[1][1].set_ylabel("throughput (env steps / sec)")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
