"""Every setting in the project, in one place.

Reach any of them from the command line with a dotted key, e.g. --set agent.lr=3e-4.
validate() then checks the ones that have to agree with each other.
"""

import argparse
import contextlib
import dataclasses
import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: Cross-field invariants are checked on construction, except while a batch of overrides is being
#: applied. See Config.__post_init__.
_VALIDATE = True


@contextlib.contextmanager
def _deferred_validation():
    global _VALIDATE
    previous, _VALIDATE = _VALIDATE, False
    try:
        yield
    finally:
        _VALIDATE = previous


# --------------------------------------------------------------------------------------
# the dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """How much each outcome is worth. Nothing here scales with the snake's length."""

    food: float = 1.0 # eating
    win: float = 2.0 # filling the whole board
    death: float = -1.0 # wall or self collision
    truncation: float = 0.0 # hunger clock ran out (NOT a death)
    step: float = -0.01 # paid every step, so dithering is never free
    shaping_scale: float = 0.1 # size of the distance-to-food nudge
    shaping_gamma: float = 0.99 # must equal agent.gamma, see validate()


@dataclass(frozen=True)
class EnvConfig:
    grid_w: int = 20 # board width in cells (not pixels)
    grid_h: int = 15 # board height in cells
    init_length: int = 3 # snake length at reset
    max_steps_without_food: int = 100 # the hunger clock -> truncation
    rewards: RewardConfig = field(default_factory=RewardConfig)


@dataclass(frozen=True)
class AgentConfig:
    hidden: tuple[int, ...] = (128, 128) # one Linear+ReLU per entry
    lr: float = 1e-3
    gamma: float = 0.99 # discount factor
    # Every iteration the loop plays num_envs moves, then does updates_per_iter gradient steps
    # of batch_size samples each. Nothing here is scaled or derived, what you set is what runs.
    updates_per_iter: int = 8 # gradient steps after each round of moves
    batch_size: int = 2_048 # past moves each gradient step learns from
    buffer_capacity: int = 1_000_000 # replay size, ~125 MB at this obs width on the GPU
    learning_starts: int = 20_000 # collect this many moves before training starts
    target_sync_steps: int = 4_000 # copy the online net into the target net every N moves played
    grad_clip: float = 10.0 # max gradient norm
    epsilon_start: float = 1.0 # fully random at the start
    epsilon_final: float = 0.01 # floor. 0.05 killed long snakes: 1 random move per 20 steps
    epsilon_decay_steps: int = 50_000 # env steps to go from start to final
    # When exploring, only pick among moves that are not instantly fatal. A random move kills a
    # long snake about one time in five and those deaths teach nothing. The danger flag already
    # said so. Turn it off to compare.
    mask_fatal_exploration: bool = True


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 500_000 # env transitions, not iterations
    log_every: int = 2_000 # write a metrics.csv row
    eval_every: int = 25_000 # run a greedy evaluation
    eval_episodes: int = 20 # episodes per evaluation
    eval_seed: int = 12_345 # fixed, so checkpoints face identical food sequences
    eval_max_steps: int = 5_000 # give up on an episode after this many steps
    save_every: int = 25_000 # write ckpt.pt
    save_buffer: bool = True # include replay in ckpt.pt (needed for an exact resume)
    run_dir: str = "runs/dev"
    seed: int = 0
    device: str = "auto" # auto | cpu | cuda
    num_envs: int = 0 # games to run in parallel; 0 means "pick a sensible one for the device"


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # Skipped while a group of overrides is being applied, because a preset can pass through a
        # temporarily invalid state on the way. apply_overrides() calls validate() at the end.
        if _VALIDATE:
            self.validate()

    def validate(self) -> None:
        """Check the settings that have to agree with each other."""
        r, e, a = self.env.rewards, self.env, self.agent

        # The nudge towards food has to discount at the same rate the learner does.
        if r.shaping_gamma != a.gamma:
            raise ValueError(
                f"env.rewards.shaping_gamma={r.shaping_gamma} must equal agent.gamma={a.gamma}; "
                "potential-based shaping is only policy-invariant at the learner's discount"
            )

        # Never try to sample more rows than the buffer will hold.
        if a.learning_starts < a.batch_size:
            raise ValueError(
                f"agent.learning_starts={a.learning_starts} must be >= agent.batch_size={a.batch_size}"
            )
        if a.batch_size > a.buffer_capacity:
            raise ValueError(
                f"agent.batch_size={a.batch_size} must be <= agent.buffer_capacity={a.buffer_capacity}"
            )

        # Running the hunger clock all the way down must cost less than one food is worth,
        # otherwise wandering around is cheaper than eating.
        if abs(r.step) * e.max_steps_without_food > r.food + 1e-9:
            raise ValueError(
                f"|env.rewards.step|={abs(r.step)} * env.max_steps_without_food="
                f"{e.max_steps_without_food} = {abs(r.step) * e.max_steps_without_food} exceeds "
                f"env.rewards.food={r.food}; starving would be cheaper than eating"
            )

        # There must be time to cross the board before starving.
        if e.max_steps_without_food < e.grid_w + e.grid_h:
            raise ValueError(
                f"env.max_steps_without_food={e.max_steps_without_food} must be >= "
                f"env.grid_w + env.grid_h = {e.grid_w + e.grid_h} or food can be unreachable in time"
            )

        # The starting snake has to fit. reset() lays it out backwards from the middle, so what
        # matters is the distance from the centre to the nearer edge, not the full board size.
        if min(e.grid_w, e.grid_h) < 5:
            raise ValueError(
                f"min(env.grid_w={e.grid_w}, env.grid_h={e.grid_h}) must be >= 5"
            )

        room = e.grid_w * e.grid_h # start high, then take the tightest direction
        for size in (e.grid_w, e.grid_h):
            room = min(room, min(size // 2, size - 1 - size // 2) + 1)

        if e.init_length > room:
            raise ValueError(
                f"env.init_length={e.init_length} does not fit: reset() lays the body backward "
                f"from the centre of a {e.grid_w}x{e.grid_h} board, which allows at most {room}"
            )

        # The nudge towards food must stay well below what food itself pays.
        if 2.0 * r.shaping_scale > 0.5 * r.food:
            raise ValueError(
                f"env.rewards.shaping_scale={r.shaping_scale} is too large relative to "
                f"env.rewards.food={r.food}; shaping would rival the food signal"
            )

        if a.updates_per_iter < 1:
            raise ValueError(f"agent.updates_per_iter={a.updates_per_iter} must be >= 1")
        if self.train.num_envs < 0:
            raise ValueError(f"train.num_envs={self.train.num_envs} must be >= 0 (0 means auto)")
        if self.train.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"train.device={self.train.device!r} must be one of auto, cpu, cuda")


# --------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},

    # The ORIGINAL v1 board: 800x600 at GRID_SIZE 20 == 40x30 cells. Use this to compare against
    # the v1 agent's best score of 94 (see docs/ and model/Best_DQN_Model_64_512_94_776k/).
    "big": {
        "env.grid_w": 40,
        "env.grid_h": 30,
        # At length 100+ the snake legitimately needs many steps to route around its own body to
        # reach food; a 200-step clock was truncating ~30% of good evaluation episodes. The step
        # cost drops to keep |step| * clock <= food (invariant 3 in validate()).
        "env.max_steps_without_food": 400,
        "env.rewards.step": -0.0025,
        "agent.epsilon_decay_steps": 300_000,
        "train.total_steps": 5_000_000,
        # Good episodes now run well past 5,000 steps and an unfinished episode is a measurement
        # we simply do not get.
        "train.eval_max_steps": 30_000,
    },

    # A full learning curve in a few minutes. Used by the tests and for hand iteration.
    "small": {
        "env.grid_w": 8,
        "env.grid_h": 8,
        "env.max_steps_without_food": 40,
        "env.rewards.step": -0.02,
        "agent.epsilon_decay_steps": 5_000,
        "agent.learning_starts": 2_000,
        "agent.batch_size": 512,
        "agent.buffer_capacity": 20_000,
        "agent.hidden": (64, 64),
        "train.total_steps": 60_000,
        "train.log_every": 500,
        "train.eval_every": 10_000,
        "train.save_every": 20_000,
    },
}


def default_config() -> Config:
    return Config()


def preset(name: str) -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(sorted(PRESETS))}")
    overrides = []
    for key, value in PRESETS[name].items():
        overrides.append(f"{key}={_unparse(value)}")

    return apply_overrides(Config(), overrides)


# --------------------------------------------------------------------------------------
# dotted-path access
# --------------------------------------------------------------------------------------


def _is_config(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def all_keys(cfg: Any = None, _prefix: str = "") -> list[str]:
    """Every dotted leaf path, e.g. ``env.rewards.food``."""
    if cfg is None:
        cfg = Config()
    out: list[str] = []
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        path = f"{_prefix}{f.name}"
        if _is_config(value):
            out.extend(all_keys(value, path + "."))
        else:
            out.append(path)
    return out


def _unparse(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    return str(value)


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _coerce(raw: str, tp: Any, key: str) -> Any:
    """Coerce a command-line string against the field's declared type."""
    origin = getattr(tp, "__origin__", None)
    if origin is tuple or tp is tuple or (isinstance(tp, str) and tp.startswith("tuple")):
        numbers = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                numbers.append(int(part))

        return tuple(numbers)
    # bool must be tested before int: bool is a subclass of int
    if tp is bool or tp == "bool":
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{key}: cannot read {raw!r} as a bool; use one of {sorted(_TRUE | _FALSE)}")
    if tp is int or tp == "int":
        return int(raw)
    if tp is float or tp == "float":
        return float(raw)
    return raw


def _set_path(obj: Any, path: str, raw: str, root: Any, full: str | None = None) -> Any:
    """Rebuild ``obj`` with one dotted leaf replaced.

    ``full`` carries the original key down the recursion so errors name the whole path.
    """
    full = full if full is not None else path
    head, _, rest = path.partition(".")
    fields = {f.name: f for f in dataclasses.fields(obj)}
    if head not in fields:
        _raise_unknown_key(full, root)
    current = getattr(obj, head)
    if rest:
        if not _is_config(current):
            _raise_unknown_key(full, root)
        return dataclasses.replace(obj, **{head: _set_path(current, rest, raw, root, full)})
    if _is_config(current):
        _raise_unknown_key(full, root)
    return dataclasses.replace(obj, **{head: _coerce(raw, fields[head].type, full)})


def _raise_unknown_key(key: str, root: Any) -> None:
    close = difflib.get_close_matches(key, all_keys(root), n=3)
    hint = f"; did you mean: {', '.join(close)}" if close else ""
    raise KeyError(f"unknown config key {key!r}{hint}")


def apply_overrides(cfg: Config, items: Sequence[str] | None) -> Config:
    """Apply a sequence of ``dotted.key=value`` strings, re-validating after each one.

    A mistyped key raises rather than being silently ignored. A typo that quietly does nothing
    is the most expensive possible failure when you are tuning.
    """
    with _deferred_validation():
        for item in items or ():
            if "=" not in item:
                raise ValueError(f"malformed override {item!r}; expected dotted.key=value")
            key, _, raw = item.partition("=")
            # Unknown keys and bad values still raise immediately, per override.
            cfg = _set_path(cfg, key.strip(), raw.strip(), cfg)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def _from_dict(cls: type, d: dict) -> Any:
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)
        elif getattr(f.type, "__origin__", None) is tuple and isinstance(value, list):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def from_dict(d: dict) -> Config:
    return _from_dict(Config, d)


def save_json(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(cfg), indent=2, sort_keys=True) + "\n")


def load_json(path: str | Path) -> Config:
    return from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------------------
# argparse plumbing
# --------------------------------------------------------------------------------------


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--config", type=str, default=None, help="load a saved config.json")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override any config key, e.g. --set agent.lr=3e-4 (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--print-config", action="store_true", help="print the resolved config and exit")


class ConfigError(Exception):
    """A user-facing configuration problem: bad key, bad value or a violated invariant."""


def config_from_args(args: argparse.Namespace) -> Config:
    """Precedence: preset < --config file < --set overrides < explicit flags."""
    try:
        cfg = preset(getattr(args, "preset", "default") or "default")
        if getattr(args, "config", None):
            cfg = load_json(args.config)
        cfg = apply_overrides(cfg, getattr(args, "overrides", None))
    except (KeyError, ValueError) as exc:
        # KeyError stringifies with quotes; strip them so the message reads cleanly.
        raise ConfigError(str(exc).strip('"')) from exc

    explicit: list[str] = []
    if getattr(args, "seed", None) is not None:
        explicit.append(f"train.seed={args.seed}")
    if getattr(args, "run_dir", None) is not None:
        explicit.append(f"train.run_dir={args.run_dir}")
    if getattr(args, "device", None) is not None:
        explicit.append(f"train.device={args.device}")
    try:
        cfg = apply_overrides(cfg, explicit)
    except (KeyError, ValueError) as exc:
        raise ConfigError(str(exc).strip('"')) from exc

    if getattr(args, "print_config", False):
        print(json.dumps(to_dict(cfg), indent=2, sort_keys=True))
        raise SystemExit(0)
    return cfg
