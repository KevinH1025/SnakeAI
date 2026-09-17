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
from typing import Any, NoReturn, Sequence

#: Cross field invariants are checked on construction, except while a batch of overrides
#: is being applied. See Config.__post_init__.
_VALIDATE = True

_DEVICES = ("auto", "cpu", "cuda") # accepted values for train.device


@contextlib.contextmanager
def _deferred_validation():
    """Turn the cross field checks off for the duration of the block."""
    global _VALIDATE
    previous = _VALIDATE # so nested uses restore rather than force it back on
    _VALIDATE = False
    try:
        yield
    finally:
        _VALIDATE = previous # put it back exactly as it was


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
    shaping_scale: float = 0.1 # size of the nudge towards food
    shaping_gamma: float = 0.99 # must equal agent.gamma, see validate()


@dataclass(frozen=True)
class EnvConfig:
    grid_w: int = 20 # board width in cells (not pixels)
    grid_h: int = 15 # board height in cells
    init_length: int = 3 # snake length at reset
    max_steps_without_food: int = 100 # the hunger clock -> truncation
    rewards: RewardConfig = field(default_factory=RewardConfig) # see RewardConfig above


@dataclass(frozen=True)
class AgentConfig:
    hidden: tuple[int, ...] = (128, 128) # one Linear+ReLU per entry
    lr: float = 1e-3 # Adam step size
    gamma: float = 0.99 # discount factor
    # Every iteration the loop plays num_envs moves, then does updates_per_iter gradient steps
    # of batch_size samples each. Nothing here is scaled or derived, what you set is what runs.
    updates_per_iter: int = 8 # gradient steps after each round of moves
    batch_size: int = 2_048 # past moves each gradient step learns from
    buffer_capacity: int = 1_000_000 # replay size, ~125 MB at this obs width on the GPU
    learning_starts: int = 20_000 # collect this many moves before training starts
    target_sync_steps: int = 4_000 # copy online net into target net every N moves
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
    save_buffer: bool = False # include replay in ckpt.pt (needed for an exact resume)
    run_dir: str = "runs/dev" # where checkpoints, metrics and config.json land
    seed: int = 0 # seeds the env, the agent and numpy
    device: str = "auto" # auto | cpu | cuda
    num_envs: int = 0 # games to run in parallel; 0 means "pick a sensible one for the device"


@dataclass(frozen=True)
class Config:
    env: EnvConfig = field(default_factory=EnvConfig) # board, hunger clock and rewards
    agent: AgentConfig = field(default_factory=AgentConfig) # network, replay and exploration
    train: TrainConfig = field(default_factory=TrainConfig) # run length, logging and device

    def __post_init__(self) -> None:
        # Skipped while a group of overrides is being applied, because a preset can pass
        # through a temporarily invalid state on the way. apply_overrides() calls validate()
        # at the end.
        if _VALIDATE:
            self.validate()

    def validate(self) -> None:
        """Check the settings that have to agree with each other."""
        _check_shaping_discount(self)      # shaping gamma vs the learner's gamma
        _check_sample_sizes(self)          # batch vs learning_starts and buffer
        _check_hunger_clock_cost(self)     # a full clock must cost less than one food
        _check_clock_allows_crossing(self) # time to walk across the board
        _check_board_fits_the_snake(self)  # board minimum and the starting body
        _check_shaping_stays_small(self)   # nudge vs the food signal
        _check_plain_ranges(self)          # single field bounds


# --------------------------------------------------------------------------------------
# invariants checked by Config.validate()
# --------------------------------------------------------------------------------------


def _check_shaping_discount(cfg: Config) -> None:
    """The nudge towards food has to discount at the same rate the learner does."""
    rewards = cfg.env.rewards
    agent = cfg.agent

    if rewards.shaping_gamma != agent.gamma:
        raise ValueError(
            f"env.rewards.shaping_gamma={rewards.shaping_gamma} "
            f"must equal agent.gamma={agent.gamma}; "
            "potential-based shaping is only policy-invariant at the learner's discount"
        )


def _check_sample_sizes(cfg: Config) -> None:
    """Never try to sample more rows than the buffer will hold."""
    agent = cfg.agent

    if agent.learning_starts < agent.batch_size:
        raise ValueError(
            f"agent.learning_starts={agent.learning_starts} "
            f"must be >= agent.batch_size={agent.batch_size}"
        )

    if agent.batch_size > agent.buffer_capacity:
        raise ValueError(
            f"agent.batch_size={agent.batch_size} "
            f"must be <= agent.buffer_capacity={agent.buffer_capacity}"
        )


def _check_hunger_clock_cost(cfg: Config) -> None:
    """Running the hunger clock all the way down must cost less than one food is worth."""
    rewards = cfg.env.rewards
    env = cfg.env

    clock_cost = abs(rewards.step) * env.max_steps_without_food # paid for a full clock

    if clock_cost > rewards.food + 1e-9:
        raise ValueError(
            f"|env.rewards.step|={abs(rewards.step)} * env.max_steps_without_food="
            f"{env.max_steps_without_food} = {clock_cost} exceeds "
            f"env.rewards.food={rewards.food}; starving would be cheaper than eating"
        )


def _check_clock_allows_crossing(cfg: Config) -> None:
    """There must be time to cross the board before starving."""
    env = cfg.env

    if env.max_steps_without_food < env.grid_w + env.grid_h:
        raise ValueError(
            f"env.max_steps_without_food={env.max_steps_without_food} must be >= "
            f"env.grid_w + env.grid_h = {env.grid_w + env.grid_h} "
            "or food can be unreachable in time"
        )


def _room_for_body(env: EnvConfig) -> int:
    """Longest body a centre start can hold, taking the tightest of the two directions."""
    room = env.grid_w * env.grid_h # start high, then take the tightest direction
    for size in (env.grid_w, env.grid_h):
        centre = size // 2 # where reset() puts the head
        behind = min(centre, size - 1 - centre) # cells to the nearer edge
        room = min(room, behind + 1) # plus the head cell itself
    return room


def _check_board_fits_the_snake(cfg: Config) -> None:
    """The board minimum plus the room reset() has for the starting body.

    reset() lays the body out backwards from the middle, so what matters is the distance
    from the centre to the nearer edge, not the full board size.
    """
    env = cfg.env

    if min(env.grid_w, env.grid_h) < 5:
        raise ValueError(
            f"min(env.grid_w={env.grid_w}, env.grid_h={env.grid_h}) must be >= 5"
        )

    room = _room_for_body(env) # cells reset() can lay the body into

    if env.init_length > room:
        raise ValueError(
            f"env.init_length={env.init_length} does not fit: reset() lays the body backward "
            f"from the centre of a {env.grid_w}x{env.grid_h} board, "
            f"which allows at most {room}"
        )


def _check_shaping_stays_small(cfg: Config) -> None:
    """The nudge towards food must stay well below what food itself pays."""
    rewards = cfg.env.rewards

    if 2.0 * rewards.shaping_scale > 0.5 * rewards.food:
        raise ValueError(
            f"env.rewards.shaping_scale={rewards.shaping_scale} is too large relative to "
            f"env.rewards.food={rewards.food}; shaping would rival the food signal"
        )


def _check_plain_ranges(cfg: Config) -> None:
    """Single field bounds that do not depend on anything else."""
    agent = cfg.agent
    train = cfg.train

    if agent.updates_per_iter < 1:
        raise ValueError(f"agent.updates_per_iter={agent.updates_per_iter} must be >= 1")

    if train.num_envs < 0:
        raise ValueError(f"train.num_envs={train.num_envs} must be >= 0 (0 means auto)")

    if train.device not in _DEVICES:
        raise ValueError(f"train.device={train.device!r} must be one of auto, cpu, cuda")


# --------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},

    # The ORIGINAL v1 board: 800x600 at GRID_SIZE 20 == 40x30 cells. Use this to compare
    # against the v1 agent's best score of 94 (see docs/ and
    # model/Best_DQN_Model_64_512_94_776k/).
    "big": {
        "env.grid_w": 40,
        "env.grid_h": 30,
        # A long snake needs many moves to route around its own body to reach food, so the
        # clock has to be generous. The step cost then has to come down to match, because a
        # full clock must never cost more than one food is worth. validate() checks that pair.
        "env.max_steps_without_food": 800,
        "env.rewards.step": -0.00125,
        "agent.epsilon_decay_steps": 300_000,
        "train.total_steps": 5_000_000,
        # Good episodes run for many thousands of steps. One that hits the cap is a
        # measurement we simply do not get.
        "train.eval_max_steps": 30_000,
        "train.save_every": 10_000,
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

    # one dotted key=value string per entry in the preset
    overrides = [f"{key}={_unparse(value)}" for key, value in PRESETS[name].items()]

    return apply_overrides(Config(), overrides)


# --------------------------------------------------------------------------------------
# dotted path access
# --------------------------------------------------------------------------------------


def _is_config(obj: Any) -> bool:
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def all_keys(cfg: Any = None, _prefix: str = "") -> list[str]:
    """Every dotted leaf path, e.g. ``env.rewards.food``."""
    if cfg is None:
        cfg = Config()
    out: list[str] = []
    for field_def in dataclasses.fields(cfg):
        value = getattr(cfg, field_def.name)
        path = f"{_prefix}{field_def.name}"
        if _is_config(value):
            out.extend(all_keys(value, path + ".")) # a nested config, so walk into it
        else:
            out.append(path) # a leaf, so record the dotted path
    return out


def _unparse(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    return str(value)


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _is_tuple_type(tp: Any) -> bool:
    """True for tuple, tuple[int, ...] and the string form of either."""
    if getattr(tp, "__origin__", None) is tuple: # tuple[int, ...]
        return True
    if tp is tuple: # a bare tuple annotation
        return True
    return isinstance(tp, str) and tp.startswith("tuple") # annotation left as a string


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    """Read "64,64" as (64, 64), skipping blank entries."""
    numbers = []
    for part in raw.split(","):
        part = part.strip() # tolerate "64, 64"
        if part:
            numbers.append(int(part))
    return tuple(numbers)


def _parse_bool(raw: str, key: str) -> bool:
    """Read one of the accepted spellings of true or false."""
    low = raw.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise ValueError(
        f"{key}: cannot read {raw!r} as a bool; use one of {sorted(_TRUE | _FALSE)}"
    )


def _coerce(raw: str, tp: Any, key: str) -> Any:
    """Coerce a command line string against the field's declared type."""
    if _is_tuple_type(tp):
        return _parse_int_tuple(raw)
    # bool must be tested before int: bool is a subclass of int
    if tp is bool or tp == "bool":
        return _parse_bool(raw, key)
    if tp is int or tp == "int":
        return int(raw)
    if tp is float or tp == "float":
        return float(raw)
    return raw # str fields and anything unrecognised pass through


def _set_path(obj: Any, path: str, raw: str, root: Any, full: str | None = None) -> Any:
    """Rebuild ``obj`` with one dotted leaf replaced.

    ``full`` carries the original key down the recursion so errors name the whole path.
    """
    if full is None:
        full = path # the outermost call names the key it was given

    head, _, rest = path.partition(".") # "env.rewards.food" -> "env", "rewards.food"
    fields = {f.name: f for f in dataclasses.fields(obj)}
    if head not in fields:
        _raise_unknown_key(full, root)

    current = getattr(obj, head) # the value this step of the path selects

    if rest: # more path to walk, so this step has to be a nested config
        if not _is_config(current):
            _raise_unknown_key(full, root)
        replaced = _set_path(current, rest, raw, root, full) # rebuild the nested config
        return dataclasses.replace(obj, **{head: replaced})

    if _is_config(current): # the path stopped on a group, not on a leaf
        _raise_unknown_key(full, root)

    value = _coerce(raw, fields[head].type, full) # CLI string -> the field's type
    return dataclasses.replace(obj, **{head: value})


def _raise_unknown_key(key: str, root: Any) -> NoReturn:
    close = difflib.get_close_matches(key, all_keys(root), n=3) # nearest real keys
    hint = f"; did you mean: {', '.join(close)}" if close else "" # empty when nothing is close
    raise KeyError(f"unknown config key {key!r}{hint}")


def _split_override(item: str) -> tuple[str, str]:
    """Split a ``dotted.key=value`` string into its two halves."""
    if "=" not in item:
        raise ValueError(f"malformed override {item!r}; expected dotted.key=value")
    key, _, raw = item.partition("=") # only the first "=" separates
    return key.strip(), raw.strip()


def apply_overrides(cfg: Config, items: Sequence[str] | None) -> Config:
    """Apply a sequence of ``dotted.key=value`` strings, validating once at the end.

    A mistyped key raises rather than being silently ignored. A typo that quietly does nothing
    is the most expensive possible failure when you are tuning.
    """
    with _deferred_validation(): # a preset can pass through an invalid state part way
        for item in items or ():
            key, raw = _split_override(item)
            cfg = _set_path(cfg, key, raw, cfg) # unknown keys and bad values raise here
    cfg.validate() # the whole batch is checked once, at the end
    return cfg


# --------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def _from_dict(cls: type, data: dict) -> Any:
    kwargs = {}
    for field_def in dataclasses.fields(cls):
        if field_def.name not in data:
            continue # missing keys keep the dataclass default
        value = data[field_def.name]

        is_nested = dataclasses.is_dataclass(field_def.type) and isinstance(value, dict)
        is_tuple_field = getattr(field_def.type, "__origin__", None) is tuple

        if is_nested:
            kwargs[field_def.name] = _from_dict(field_def.type, value)
        elif is_tuple_field and isinstance(value, list):
            kwargs[field_def.name] = tuple(value) # json has no tuples
        else:
            kwargs[field_def.name] = value
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
    # what the config starts from plus the overrides applied on top of it
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--config", type=str, default=None, help="load a saved config.json")
    parser.add_argument(
        "--set", dest="overrides", action="append", metavar="KEY=VALUE",
        help="override any config key, e.g. --set agent.lr=3e-4 (repeatable)",
    )

    # the settings that get a shorthand flag of their own
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", choices=_DEVICES, default=None)
    parser.add_argument(
        "--print-config", action="store_true",
        help="print the resolved config and exit",
    )


class ConfigError(Exception):
    """A configuration problem to show the user: bad key, bad value or a broken invariant."""


def _as_config_error(exc: Exception) -> ConfigError:
    """Turn a lookup or value failure into the error the CLI prints."""
    # KeyError stringifies with quotes; strip them so the message reads cleanly.
    return ConfigError(str(exc).strip('"'))


def _base_config(args: argparse.Namespace) -> Config:
    """The starting point: a preset, or a saved config.json when --config is given."""
    cfg = preset(getattr(args, "preset", "default") or "default")
    if getattr(args, "config", None):
        cfg = load_json(args.config) # the file replaces the preset outright
    return cfg


def _flag_overrides(args: argparse.Namespace) -> list[str]:
    """The flags with a shorthand of their own, written as dotted overrides."""
    overrides: list[str] = []
    if getattr(args, "seed", None) is not None:
        overrides.append(f"train.seed={args.seed}")
    if getattr(args, "run_dir", None) is not None:
        overrides.append(f"train.run_dir={args.run_dir}")
    if getattr(args, "device", None) is not None:
        overrides.append(f"train.device={args.device}")
    return overrides


def config_from_args(args: argparse.Namespace) -> Config:
    """Precedence: preset < --config file < --set overrides < explicit flags."""
    try:
        cfg = _base_config(args)
        cfg = apply_overrides(cfg, getattr(args, "overrides", None)) # the --set strings
    except (KeyError, ValueError) as exc:
        raise _as_config_error(exc) from exc

    explicit = _flag_overrides(args)
    try:
        cfg = apply_overrides(cfg, explicit) # the flags win over everything above
    except (KeyError, ValueError) as exc:
        raise _as_config_error(exc) from exc

    if getattr(args, "print_config", False):
        print(json.dumps(to_dict(cfg), indent=2, sort_keys=True))
        raise SystemExit(0)
    return cfg
