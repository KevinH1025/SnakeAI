"""Plot a finished (or running) training run from its CSV.

Offline and out-of-process on purpose. v1 called ``plt.pause(0.1)`` from inside the game loop on
every death, which blocked training for 100 ms each episode and made matplotlib a hard dependency
of learning anything at all.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metrics(run_dir: Path) -> dict[str, list[float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.csv in {run_dir}")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows yet")
    columns: dict[str, list[float]] = {}
    for key in rows[0]:
        values = []
        for row in rows:
            values.append(float(row[key]))
        columns[key] = values

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

    axes[0][0].plot(m["step"], m["score_mean"], label="mean score (last 200 eps)")
    if "score_max" in m:
        axes[0][0].plot(m["step"], m["score_max"], alpha=0.4, label="best in window")
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
