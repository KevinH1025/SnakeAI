"""Training-loop, metric, config and headless tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from snakeai.config import Config, apply_overrides, from_dict, preset, to_dict
from snakeai.train import METRIC_COLUMNS, Window, run_training


# -- metrics ----------------------------------------------------------------------------


def test_window_mean_uses_window_length_not_iteration_count():
    """v1 divided a 1000-element deque's sum by the total iteration count, so its 'average loss'
    fell toward zero forever regardless of the actual loss."""
    w = Window(3)
    for value in (1.0, 1.0, 1.0):
        w.add(value)
    assert w.mean == pytest.approx(1.0)
    for _ in range(1000):
        w.add(1.0)
    assert w.mean == pytest.approx(1.0), "mean decayed as more values arrived"
    assert len(w) == 3
    assert Window(5).mean == 0.0


# -- the loop ---------------------------------------------------------------------------


def short_cfg(tmp_path, **extra):
    overrides = [
        "train.device=cpu", "train.num_envs=4", "train.total_steps=3000",
        "train.log_every=1000", "train.eval_every=2000", "train.save_every=2000",
        "train.eval_episodes=4", "train.eval_max_steps=200",
        "agent.batch_size=64", "agent.learning_starts=128", "agent.buffer_capacity=5000",
        "agent.hidden=32,32", "agent.target_sync_steps=50",
        f"train.run_dir={tmp_path}",
    ]
    overrides += [f"{k}={v}" for k, v in extra.items()]
    return apply_overrides(preset("small"), overrides)


def test_short_run_writes_all_artifacts(tmp_path):
    summary = run_training(short_cfg(tmp_path), quiet=True)
    for name in ("config.json", "metrics.csv", "summary.json", "ckpt.pt", "eval.jsonl"):
        assert (tmp_path / name).exists(), f"missing {name}"
    assert summary["steps"] >= 3000
    assert summary["episodes"] > 0
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert header.startswith("step,episodes,epsilon,loss")
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["device"] == "cpu" and saved["n_envs"] == 4


def test_same_seed_same_metrics(tmp_path):
    a = run_training(short_cfg(tmp_path / "a", **{"train.seed": 7}), quiet=True)
    b = run_training(short_cfg(tmp_path / "b", **{"train.seed": 7}), quiet=True)
    assert a["episodes"] == b["episodes"]
    assert a["best_score"] == b["best_score"]
    assert a["reasons"] == b["reasons"]


def test_different_seed_differs(tmp_path):
    a = run_training(short_cfg(tmp_path / "a", **{"train.seed": 1}), quiet=True)
    b = run_training(short_cfg(tmp_path / "b", **{"train.seed": 2}), quiet=True)
    assert (a["episodes"], a["reasons"]) != (b["episodes"], b["reasons"])


def test_resume_continues_from_checkpoint(tmp_path):
    first = run_training(short_cfg(tmp_path / "run"), quiet=True)
    resumed = run_training(
        short_cfg(tmp_path / "run", **{"train.total_steps": 6000}),
        resume=str(tmp_path / "run" / "ckpt.pt"), quiet=True,
    )
    assert resumed["steps"] >= 6000
    assert resumed["best_score"] >= first["best_score"]


def test_checkpoint_is_written_atomically(tmp_path):
    run_training(short_cfg(tmp_path), quiet=True)
    assert (tmp_path / "ckpt.pt").exists()
    assert not list(tmp_path.glob("*.tmp")), "a .tmp checkpoint was left behind"


def test_evaluate_is_deterministic_and_reports_reasons(tmp_path):
    import torch
    from snakeai.agent import DQNAgent
    from snakeai.evaluate import evaluate

    cfg = short_cfg(tmp_path)
    run_training(cfg, quiet=True)
    state = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=False)
    agent = DQNAgent(cfg.agent, torch.device("cpu"), seed=0)
    agent.load_state_dict(state["agent"])

    first = evaluate(agent, cfg, episodes=8, seed=42, max_steps=300)
    second = evaluate(agent, cfg, episodes=8, seed=42, max_steps=300)
    assert first == second, "same seeds must give identical evaluation"
    assert set(first["reasons"]) <= {"hit_wall", "hit_self", "starved", "won", "unfinished"}


# -- config -----------------------------------------------------------------------------


def test_config_roundtrip_and_presets():
    for name in ("default", "big", "small"):
        cfg = preset(name)
        assert from_dict(to_dict(cfg)) == cfg
    assert preset("big").env.grid_w == 40 and preset("big").env.grid_h == 30


def test_unknown_key_suggests_a_correction():
    with pytest.raises(KeyError) as excinfo:
        apply_overrides(Config(), ["agent.gama=0.9"])
    message = str(excinfo.value)
    assert "agent.gama" in message and "agent.gamma" in message


@pytest.mark.parametrize("override,fragment", [
    ("env.rewards.step=-0.5", "starving would be cheaper"),
    ("agent.gamma=0.5", "shaping_gamma"),
    ("agent.batch_size=999999", "learning_starts"),
    ("env.max_steps_without_food=5", "grid_w + env.grid_h"),
    ("train.device=tpu", "must be one of"),
])
def test_invariants_fire(override, fragment):
    with pytest.raises(ValueError, match=fragment.replace("+", r"\+")):
        apply_overrides(Config(), [override])


def test_overrides_coerce_types():
    cfg = apply_overrides(Config(), ["agent.hidden=256,256", "train.save_buffer=false",
                                     "agent.lr=3e-4", "agent.batch_size=64"])
    assert cfg.agent.hidden == (256, 256)
    assert cfg.train.save_buffer is False
    assert cfg.agent.lr == pytest.approx(3e-4)
    assert isinstance(cfg.agent.batch_size, int)


def test_presets_pass_their_own_invariants():
    for name in ("default", "big", "small"):
        preset(name).validate()


# -- headless -------------------------------------------------------------------------


def test_training_never_imports_a_gui_library():
    """Training must not pull in pygame or matplotlib. v1's game loop WAS the training loop."""
    code = (
        "import sys; import snakeai.train, snakeai.agent, snakeai.env, snakeai.vecenv, snakeai.evaluate; "
        "bad=[m for m in ('pygame','matplotlib','matplotlib.pyplot') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent.parent)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"training imported GUI modules: {out.stdout.strip()}"


def test_replay_buffer_is_not_bundled_into_ckpt(tmp_path):
    """ckpt.pt must stay small enough for `play --follow` to reread it cheaply.

    Bundled together, a buffer_capacity=1,000,000 run wrote a 128 MB ckpt.pt every save and the
    viewer had to read all of it to get ~300 KB of weights, while competing with the trainer
    writing the next copy.
    """
    from snakeai.train import replay_path

    cfg = short_cfg(tmp_path, **{"train.save_buffer": "true", "agent.buffer_capacity": 20000})
    run_training(cfg, quiet=True)

    ckpt = tmp_path / "ckpt.pt"
    assert ckpt.exists()
    assert ckpt.stat().st_size < 2_000_000, "ckpt.pt is carrying the replay buffer again"
    assert replay_path(ckpt).exists(), "the buffer should be saved, just in its own file"
    assert not any("buffer" in k for k in
                   torch.load(ckpt, map_location="cpu", weights_only=False)["agent"])


def test_save_buffer_false_writes_no_replay_file(tmp_path):
    from snakeai.train import replay_path

    run_training(short_cfg(tmp_path, **{"train.save_buffer": "false"}), quiet=True)
    assert (tmp_path / "ckpt.pt").exists()
    assert not replay_path(tmp_path / "ckpt.pt").exists()


def test_live_scores_count_games_still_running(cfg):
    """The record must include games in progress, not only the ones that died.

    A strong agent plays very long games, so counting only finished episodes samples the games
    that died early, which are exactly the bad ones.
    """
    from snakeai.vecenv import VecSnakeEnv

    vec = VecSnakeEnv(cfg.env, n_envs=4, seed=0)
    vec.envs[0].score = 90 # a game doing well, nowhere near finishing
    vec.envs[1].score = 10
    vec.envs[2].score = 0
    vec.envs[3].score = 0

    mean, best = vec.live_scores()
    assert best == 90.0, "the best live game must be visible without waiting for it to end"
    assert mean == 25.0, "the mean is over all games, not just the finished ones"
    assert vec.best_live_score() == 90.0


def test_metrics_header_carries_the_live_columns(tmp_path):
    run_training(short_cfg(tmp_path), quiet=True)
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    for column in ("step", "episodes", "score_mean", "score_max", "best_score"):
        assert column in header, f"missing {column}"


def test_a_fresh_run_does_not_append_to_the_previous_run_s_metrics(tmp_path):
    """Two runs in one dir used to share a csv, so the step column restarted partway down."""
    run_training(short_cfg(tmp_path), quiet=True)
    first = (tmp_path / "metrics.csv").read_text()
    run_training(short_cfg(tmp_path), quiet=True)

    assert (tmp_path / "metrics.csv.1").read_text() == first, "old rows were not kept aside"
    steps = [int(row["step"]) for row in _rows(tmp_path / "metrics.csv")]
    assert steps == sorted(steps), "step column goes backwards, so two runs got mixed together"


def test_a_stale_header_is_moved_aside_instead_of_appended_under(tmp_path):
    """A narrower header over today's rows shifts every later column and says nothing."""
    run_training(short_cfg(tmp_path), quiet=True)
    path = tmp_path / "metrics.csv"
    old = [c for c in METRIC_COLUMNS if c != "best_score"] # what runs before best_score wrote
    lines = path.read_text().splitlines()
    path.write_text("\n".join([",".join(old)] + lines[1:]) + "\n")

    run_training(short_cfg(tmp_path, **{"train.total_steps": 6000}), quiet=True,
                 resume=str(tmp_path / "ckpt.pt"))

    assert path.read_text().splitlines()[0].split(",") == METRIC_COLUMNS
    assert (tmp_path / "metrics.csv.1").exists(), "the stale file was dropped, not kept"


def test_plot_reads_the_same_columns_training_writes():
    """The two lists are separate copies so plotting stays free of torch. They must agree."""
    from snakeai.plot import METRIC_COLUMNS as plot_columns

    assert plot_columns == METRIC_COLUMNS


def test_plot_ignores_rows_left_over_from_an_older_format(tmp_path):
    from snakeai.plot import read_metrics

    old = [c for c in METRIC_COLUMNS if c != "best_score"]
    path = tmp_path / "metrics.csv"
    path.write_text(
        ",".join(old) + "\n"
        + "1,0,0.9,0.0,0.0,0.0,0,100.0,1.0\n"       # an older run, one column short
        + "1,0,0.9,0.0,0.0,5.0,7,7,100.0,1.0\n"     # today's layout, appended underneath
        + "2,1,0.8,0.1,1.0,6.0,9,9,100.0,2.0\n"
    )
    m = read_metrics(tmp_path)

    assert m["step"] == [1.0, 2.0], "the narrower row was read as if it had today's layout"
    assert m["best_score"] == [7.0, 9.0]
    assert m["steps_per_sec"] == [100.0, 100.0], "columns are shifted"


def _rows(path):
    import csv as _csv

    with path.open(newline="") as f:
        return list(_csv.DictReader(f))


def test_plot_keeps_only_the_newest_run_in_a_shared_file(tmp_path):
    """A run dir written before the guard above can hold a restart and a resume in one file."""
    from snakeai.plot import read_metrics

    def row(step, best):
        return f"{step},0,0.5,0.0,0.0,0.0,{best},{best},100.0,1.0"

    path = tmp_path / "metrics.csv"
    path.write_text("\n".join([
        ",".join(METRIC_COLUMNS),
        row(100, 1), row(200, 2), row(300, 3),  # a first run, later started over
        row(100, 5), row(200, 6), row(300, 7),  # the run that matters, then resumed from 200
        row(250, 8), row(400, 9),
    ]) + "\n")
    m = read_metrics(tmp_path)

    assert m["step"] == [100.0, 200.0, 250.0, 400.0], "an abandoned run was plotted too"
    assert m["best_score"] == [5.0, 6.0, 8.0, 9.0], "kept the wrong copy of an overlapping step"
