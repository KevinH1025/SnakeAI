"""Plot a finished (or running) training run from its CSV.

Plotting happens offline and out of process. matplotlib is imported inside main(), so
importing this module, or training, never loads a GUI stack.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Must stay identical to snakeai.train.METRIC_COLUMNS. Copied rather than imported so that
# plotting a run never has to load torch. A test compares the two lists.
METRIC_COLUMNS = ["step", "episodes", "epsilon", "loss", "return_mean",
                  "score_mean", "score_max", "best_score", "steps_per_sec", "wall_s"]


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """The header and the data rows of metrics.csv, as raw strings."""
    if not path.exists():
        raise FileNotFoundError(f"no metrics.csv in {path.parent}")

    with path.open(newline="") as f:
        raw = list(csv.reader(f)) # every line, header included
    if not raw:
        raise ValueError(f"{path} has no data rows yet")

    header = raw[0] # the column names this file was started with
    rows = raw[1:] # everything written after it
    return header, rows


def _rows_of_one_layout(
    header: list[str], rows: list[list[str]]
) -> tuple[list[str], list[list[str]]]:
    """The column names to read by, plus only the rows holding exactly that many fields.

    A restart can append today's wider rows under an older, narrower header, so rows are
    matched on their own width. Today's layout wins when any row has it, otherwise the
    file's own header is used.
    """
    todays = [row for row in rows if len(row) == len(METRIC_COLUMNS)] # rows in today's layout
    if todays:
        return METRIC_COLUMNS, todays

    matching = [row for row in rows if len(row) == len(header)] # rows matching this header
    return header, matching


def _newest_run(rows: list[list[str]]) -> list[list[str]]:
    """Only the rows belonging to the most recent run, in file order.

    One file can hold several runs: a restart begins again from step 0 and a resume replays
    the steps between its checkpoint and wherever the previous run stopped. Reading backwards
    and keeping only steps that keep falling leaves the rows leading up to the newest one.
    """
    newest = []
    limit = float("inf") # every kept row must start below the one after it
    for row in reversed(rows):
        step = float(row[0]) # the step column, always first
        if step < limit:
            newest.append(row) # part of the newest run
            limit = step # the next row back has to fall below this one
    return newest[::-1] # back into file order


def _to_columns(names: list[str], rows: list[list[str]]) -> dict[str, list[float]]:
    """The rows transposed into one list of floats per column name."""
    columns: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        columns[name] = [float(row[i]) for row in rows] # column i of every row
    return columns


def read_metrics(run_dir: Path) -> dict[str, list[float]]:
    """metrics.csv from a run directory, as one list of floats per column."""
    path = run_dir / "metrics.csv"
    header, rows = _read_csv(path)

    names, matching = _rows_of_one_layout(header, rows) # drop rows of another width
    if not matching:
        raise ValueError(f"{path} has no data rows yet")

    kept = _newest_run(matching) # drop rows from runs that were superseded

    skipped = len(rows) - len(kept) # every row dropped by either stage
    if skipped:
        print(f"note: skipped {skipped} row(s) in {path.name}, left over from an earlier run")

    return _to_columns(names, kept)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line: a run directory and optionally a PNG to write."""
    parser = argparse.ArgumentParser(description="Plot a SnakeAI training run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--save", type=Path, default=None,
                        help="write a PNG instead of showing a window")
    return parser.parse_args(argv)


def _load_pyplot(headless: bool):
    """pyplot, imported late so importing this module stays GUI free."""
    import matplotlib # imported HERE, not at module scope, so train.py stays GUI free
    if headless:
        matplotlib.use("Agg") # headless backend when writing a file
    import matplotlib.pyplot as plt # only safe once the backend above is set
    return plt


def _draw_scores(ax, metrics: dict[str, list[float]]) -> None:
    """Mean score, plus the best live game and the best ever when the file has them."""
    steps = metrics["step"]
    ax.plot(steps, metrics["score_mean"], label="mean score, live games")
    if "score_max" in metrics: # older files stop short of this column
        ax.plot(steps, metrics["score_max"], alpha=0.4, label="best live game")
    if "best_score" in metrics: # and short of this one
        ax.plot(steps, metrics["best_score"], alpha=0.4, label="best ever")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("score")
    ax.legend()


def _draw_line(ax, metrics: dict[str, list[float]],
               column: str, ylabel: str, color: str) -> None:
    """One column against the step count."""
    ax.plot(metrics["step"], metrics[column], color=color)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    plt = _load_pyplot(bool(args.save)) # a file to write means no window to show
    metrics = read_metrics(args.run_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"SnakeAI - {args.run_dir}")

    scores = axes[0][0] # top left panel
    loss = axes[0][1] # top right panel
    epsilon = axes[1][0] # bottom left panel
    throughput = axes[1][1] # bottom right panel

    _draw_scores(scores, metrics)

    _draw_line(loss, metrics, "loss", "loss (rolling mean of last 200 updates)", "tab:orange")
    loss.set_yscale("log") # losses span orders of magnitude

    _draw_line(epsilon, metrics, "epsilon", "epsilon", "tab:green")
    _draw_line(throughput, metrics, "steps_per_sec",
               "throughput (env steps / sec)", "tab:purple")

    for ax in axes.flat:
        ax.grid(alpha=0.25) # faint grid on every panel
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=120)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
