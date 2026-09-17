"""The Snake game itself: movement, what the snake sees and what it gets paid.

No torch and no pygame in here, so this is just Python and numpy. Drawing happens in play.py.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .config import EnvConfig, RewardConfig
from .regions import label_regions, reach_counts


# --------------------------------------------------------------------------- geometry

# clockwise, so turning right is +1 on the index and turning left is -1
DIRECTIONS: tuple[tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0)) # y grows downward
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

N_ACTIONS = 3
ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2

_TURN = (0, -1, +1) # how much each action rotates the heading


def turn(heading: int, action: int) -> int:
    """Turn the heading by an action: 0 = straight, 1 = left, 2 = right."""
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action must be 0 (straight), 1 (left) or 2 (right); got {action!r}")

    return (heading + _TURN[action]) % 4 # wrap around the 4 directions


class Event(enum.Enum):
    """What happened on a step."""

    MOVE = "move"
    ATE = "ate"
    HIT_WALL = "hit_wall"
    HIT_SELF = "hit_self"
    STARVED = "starved"
    WON = "won"


# the game is over and there is no future left to value
TERMINAL_EVENTS = frozenset({Event.HIT_WALL, Event.HIT_SELF, Event.WON})

# we stopped the episode ourselves, but the snake was still alive
TRUNCATING_EVENTS = frozenset({Event.STARVED})


# --------------------------------------------------------------------------- what the snake sees

I_DANGER_STRAIGHT = 0
I_DANGER_LEFT = 1
I_DANGER_RIGHT = 2
I_FOOD_AHEAD = 3
I_FOOD_BEHIND = 4
I_FOOD_LEFT = 5
I_FOOD_RIGHT = 6
I_FOOD_FORWARD = 7
I_FOOD_LATERAL = 8
I_LENGTH_FRAC = 9
I_HUNGER_FRAC = 10
I_FREE_STRAIGHT = 11
I_FREE_LEFT = 12
I_FREE_RIGHT = 13
I_TAIL_STRAIGHT = 14
I_TAIL_LEFT = 15
I_TAIL_RIGHT = 16
I_REACH_STRAIGHT = 17
I_REACH_LEFT = 18
I_REACH_RIGHT = 19

OBS_NAMES: tuple[str, ...] = (
    "danger_straight", # 1.0 if going straight kills me
    "danger_left", # 1.0 if turning left kills me
    "danger_right", # 1.0 if turning right kills me

    "food_ahead", # 1.0 if the food is in front of me
    "food_behind", # 1.0 if it is behind me
    "food_left", # 1.0 if it is to my left
    "food_right", # 1.0 if it is to my right
    "food_forward", # how far ahead/behind, scaled to [-1, 1]
    "food_lateral", # how far left/right, scaled to [-1, 1]

    "length_frac", # how much of the board my body covers
    "hunger_frac", # how close I am to starving

    "free_straight", # how much room going straight leads into
    "free_left", # how much room turning left leads into
    "free_right", # how much room turning right leads into

    "tail_straight", # 1.0 if I could still reach my own tail after going straight
    "tail_left", # 1.0 if I could still reach it after turning left
    "tail_right", # 1.0 if I could still reach it after turning right

    "reach_straight", # how much room there is near me if I go straight
    "reach_left", # the same for turning left
    "reach_right", # the same for turning right
)

OBS_DIM = len(OBS_NAMES)

# How far the reach count looks. free_* and tail_* both read off one whole board labelling, so
# they hand back the same number for every move whenever those moves open into the same region.
# Measured on a trained net that is 95% of steps once the snake passes length 200, which is
# exactly where it starts dying to its own body. Counting cells within a short walk instead
# stays different between moves in about 9 out of 10 of those states.
REACH_DEPTH = 10
REACH_MAX = 2 * REACH_DEPTH * (REACH_DEPTH + 1) + 1 # most cells a walk that long can cover


# --------------------------------------------------------------------------- reward


@dataclass(frozen=True)
class StepOutcome:
    """The only three things the reward is allowed to look at."""

    event: Event # what happened
    prev_dist_norm: float # distance to the food before the move
    next_dist_norm: float # distance to the food after the move


def compute_reward(outcome: StepOutcome, cfg: RewardConfig) -> float:
    """Turn one step's outcome into a number."""
    reward = cfg.step # every step costs a little

    if outcome.event is Event.ATE:
        reward += cfg.food
    elif outcome.event is Event.WON:
        reward += cfg.win
    elif outcome.event in (Event.HIT_WALL, Event.HIT_SELF):
        reward += cfg.death
    elif outcome.event is Event.STARVED:
        reward += cfg.truncation

    # A small nudge towards the food, in the potential based form gamma*phi(s') - phi(s).
    # Written this way so it speeds learning up without changing which policy is best.
    phi_prev = -cfg.shaping_scale * outcome.prev_dist_norm

    if outcome.event in TERMINAL_EVENTS:
        phi_next = 0.0 # nothing comes after a terminal state
    else:
        phi_next = -cfg.shaping_scale * outcome.next_dist_norm

    return float(reward + cfg.shaping_gamma * phi_next - phi_prev)


# --------------------------------------------------------------------------- the game


class SnakeEnv:
    """One game of Snake.

        obs = env.reset(seed=None)                                  -> (OBS_DIM,) float32
        obs, reward, terminated, truncated, info = env.step(action) -> action is 0, 1 or 2

    step() never resets by itself. When the game ends, the caller calls reset().
    """

    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed) # every random choice comes from here
        self._needs_reset = True

        # Scratch space for the region labelling, allocated once and reused every step.
        cells = cfg.grid_w * cfg.grid_h
        self._blocked = np.zeros(cells, np.uint8)
        self._label = np.zeros(cells, np.int32)
        self._sizes = np.zeros(cells + 2, np.int32)
        self._stack = np.zeros(cells + 8, np.int32)
        self._seen = np.zeros(cells, np.uint8) # reach_counts leaves this all zero again
        self._queue = np.zeros(cells, np.int32)
        self._entries = np.full(3, -1, np.int32)

        self.reset()

    # -- starting and finishing ---------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new game and return the first observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.cfg

        self.heading = int(self.rng.integers(4)) # face a random direction
        head_x = cfg.grid_w // 2 # start in the middle
        head_y = cfg.grid_h // 2
        back = DIRECTIONS[(self.heading + 2) % 4] # the body trails out behind the head

        self.snake: list[tuple[int, int]] = []
        for i in range(cfg.init_length):
            self.snake.append((head_x + back[0] * i, head_y + back[1] * i))

        self.occupied: set[tuple[int, int]] = set(self.snake) # for fast collision checks

        self.score = 0 # food eaten this game
        self.steps = 0 # moves made this game
        self.steps_since_food = 0 # the hunger clock

        self.terminated = False # died
        self.truncated = False # ran out of hunger clock
        self.last_event = Event.MOVE
        self.crash_cell: tuple[int, int] | None = None # the cell we died trying to enter

        self.food = self._spawn_food()
        self._needs_reset = False

        return self.observe()

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    # -- the board ----------------------------------------------------------

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < self.cfg.grid_w and 0 <= cell[1] < self.cfg.grid_h

    def _spawn_food(self) -> tuple[int, int] | None:
        """Put food on a random empty cell. Returns None if the board is full."""
        width = self.cfg.grid_w
        height = self.cfg.grid_h

        if len(self.occupied) >= width * height:
            return None # no empty cells left, the snake has won

        # Guessing is fast while the board is mostly empty.
        for _ in range(100):
            cell = (int(self.rng.integers(width)), int(self.rng.integers(height)))
            if cell not in self.occupied:
                return cell

        # Nearly full, so just list what is left.
        free = []
        for x in range(width):
            for y in range(height):
                if (x, y) not in self.occupied:
                    free.append((x, y))

        return free[int(self.rng.integers(len(free)))]

    def _fatal(self, cell: tuple[int, int]) -> tuple[bool, bool]:
        """Would moving into this cell end the game? Returns (hit_wall, hit_self)."""
        if not self._in_bounds(cell):
            return True, False # off the board

        if cell not in self.occupied:
            return False, False # empty cell, safe

        # The tail moves out of the way as the head arrives, unless we are about to eat and grow.
        growing = self.food is not None and cell == self.food
        if cell == self.snake[-1] and not growing:
            return False, False

        return False, True # some other part of the body

    def _would_die(self, head: tuple[int, int], heading: int) -> bool:
        """Would stepping one cell from `head` along `heading` end the game?"""
        step = DIRECTIONS[heading]
        nxt = (head[0] + step[0], head[1] + step[1])

        hit_wall, hit_self = self._fatal(nxt)

        return hit_wall or hit_self

    # The three methods below are the original flood implementation. observe() no longer calls
    # them, _regions() does the same job in one labelling pass. They are kept because the tests
    # check the labelling against them, so a change to one that breaks the other fails loudly.

    def _blocked_cells(self) -> bytearray:
        """A flat width*height map: 1 where the snake's body is, 0 where it is free.

        Flat integer indices (x + y*width) are much faster to walk than (x, y) tuples in a set,
        and every flood below uses this same map.
        """
        width = self.cfg.grid_w
        blocked = bytearray(width * self.cfg.grid_h)

        for body_x, body_y in self.snake:
            blocked[body_x + body_y * width] = 1

        # The tail steps out of the way as the head arrives, so it is not really an obstacle.
        # _fatal() already treats it that way and these have to agree with each other.
        tail_x, tail_y = self.snake[-1]
        blocked[tail_x + tail_y * width] = 0

        return blocked

    def _tail_reachable(self, head: tuple[int, int], blocked: bytearray) -> tuple[float, float, float]:
        """After each move, could the snake still reach its own tail?

        If it can, the tail keeps retreating ahead of it and the space opens up. If it cannot, it
        is sealed in and will die once the pocket fills, however roomy that pocket looks now.
        """
        width = self.cfg.grid_w
        height = self.cfg.grid_h

        # Which cell would each move put the head on? Skip any that is a wall or a body cell.
        wanted: dict[int, list[int]] = {}
        for slot, action in enumerate((ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT)):
            step = DIRECTIONS[turn(self.heading, action)]
            cell_x = head[0] + step[0]
            cell_y = head[1] + step[1]

            if not (0 <= cell_x < width and 0 <= cell_y < height):
                continue # off the board, stays 0.0
            if blocked[cell_x + cell_y * width]:
                continue # into the body, stays 0.0

            wanted.setdefault(cell_x + cell_y * width, []).append(slot)

        reachable = [0.0, 0.0, 0.0]
        if not wanted:
            return 0.0, 0.0, 0.0 # every move is fatal anyway

        # Spread out from the tail. Every cell we touch is one the tail is connected to.
        tail_x, tail_y = self.snake[-1]
        start = tail_x + tail_y * width

        visited = bytearray(width * height)
        visited[start] = 1
        stack = [start]
        still_looking = len(wanted)

        while stack and still_looking:
            here = stack.pop()

            if here in wanted: # this cell is one of our three, mark it and tick it off
                for slot in wanted.pop(here):
                    reachable[slot] = 1.0
                    still_looking -= 1
                if not still_looking:
                    break # found all three, no reason to keep walking

            here_x = here % width
            here_y = here // width

            if here_x > 0:
                west = here - 1
                if not blocked[west] and not visited[west]:
                    visited[west] = 1
                    stack.append(west)

            if here_x < width - 1:
                east = here + 1
                if not blocked[east] and not visited[east]:
                    visited[east] = 1
                    stack.append(east)

            if here_y > 0:
                north = here - width
                if not blocked[north] and not visited[north]:
                    visited[north] = 1
                    stack.append(north)

            if here_y < height - 1:
                south = here + width
                if not blocked[south] and not visited[south]:
                    visited[south] = 1
                    stack.append(south)

        return reachable[0], reachable[1], reachable[2]

    def _free_spaces(self, head: tuple[int, int], budget: int,
                     blocked: bytearray) -> tuple[int, int, int]:
        """How many empty cells each move leads into, as (straight, left, right).

        Spreads out from the cell each move lands on, counting as it goes, stopping at `budget`.
        """
        width = self.cfg.grid_w
        height = self.cfg.grid_h

        # Marks which cells this flood has already been to, so it never counts one twice.
        # We reuse ONE array across all three floods, stamping it with a different `token` each
        # time (1, then 2, then 3) instead of clearing it. Clearing is slower than stamping.
        visited = bytearray(width * height)

        # STEP 1: work out which cell each of the three moves would put the head on.
        # Store them as flat indices (x + y*width). None means "that move is not even possible".
        entries = []
        for action in (ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT):
            step = DIRECTIONS[turn(self.heading, action)]
            cell_x = head[0] + step[0]
            cell_y = head[1] + step[1]

            if not (0 <= cell_x < width and 0 <= cell_y < height):
                entries.append(None) # that move goes off the board
            elif blocked[cell_x + cell_y * width]:
                entries.append(None) # that move goes into our own body
            else:
                entries.append(cell_x + cell_y * width)

        # STEP 2: moves that are not possible have no room behind them, so their answer is 0.
        # Everything still set to None is a move we actually have to go and measure.
        counts: list[int | None] = [None, None, None]
        for slot, entry in enumerate(entries):
            if entry is None:
                counts[slot] = 0

        # STEP 3: measure each remaining move by flooding out from the cell it lands on.
        token = 0
        for slot, entry in enumerate(entries):
            if counts[slot] is not None:
                continue # already answered: either impossible or shared with an earlier flood

            token += 1 # a fresh stamp, so this flood cannot be confused with the previous one

            # `stack` is the list of cells we have found but not yet looked around from.
            # Start with just the entry cell and mark it as visited immediately.
            visited[entry] = token
            stack = [entry]
            count = 0

            while stack and count < budget:
                here = stack.pop() # take any cell we have not looked around from yet
                count += 1 # we are standing on it, so it counts as reachable

                # Turn the flat index back into x, y so we can find the neighbours.
                here_x = here % width
                here_y = here // width

                # Look at all four neighbours. For each one: if it is on the board, is not body,
                # and we have not already seen it this flood, then it is reachable too. Mark it
                # and add it to the pile to look around from later.
                if here_x > 0: # there is a cell to the west
                    west = here - 1
                    if not blocked[west] and visited[west] != token:
                        visited[west] = token
                        stack.append(west)

                if here_x < width - 1: # to the east
                    east = here + 1
                    if not blocked[east] and visited[east] != token:
                        visited[east] = token
                        stack.append(east)

                if here_y > 0: # to the north (y grows downward, so north is -width)
                    north = here - width
                    if not blocked[north] and visited[north] != token:
                        visited[north] = token
                        stack.append(north)

                if here_y < height - 1: # to the south
                    south = here + width
                    if not blocked[south] and visited[south] != token:
                        visited[south] = token
                        stack.append(south)

            # The loop ended either because we ran out of cells (a small pocket) or because we
            # reached the budget (plenty of room). Either way, `count` is the answer.
            counts[slot] = count

            # If this flood walked over another move's entry cell, that move is in the same
            # region and would count the same cells, so copy the answer instead of re-walking.
            # Only when positively visited: a flood that stopped at the budget may just not
            # have reached it yet.
            for other in range(slot + 1, 3):
                if counts[other] is None and visited[entries[other]] == token:
                    counts[other] = count

        return counts[0], counts[1], counts[2]

    # -- taking a step ------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Move once. Returns (obs, reward, terminated, truncated, info)."""
        if self._needs_reset:
            raise RuntimeError("step() on a finished episode - call reset()")

        cfg = self.cfg
        head = self.snake[0]
        prev_dist = self._dist_norm(head, self.food) # measured before anything moves

        self.heading = turn(self.heading, action)
        step = DIRECTIONS[self.heading]
        nxt = (head[0] + step[0], head[1] + step[1])

        growing = self.food is not None and nxt == self.food
        hit_wall, hit_self = self._fatal(nxt)

        if hit_wall or hit_self:
            # The bad move is not applied: the snake stays exactly where it died.
            self.crash_cell = nxt
            self.last_event = Event.HIT_WALL if hit_wall else Event.HIT_SELF
            self.terminated = True

        elif growing:
            self.snake.insert(0, nxt) # new head and no tail removed, so the snake grows
            self.occupied.add(nxt)

            self.score += 1
            self.steps_since_food = 0 # fed, so reset the hunger clock
            self.food = self._spawn_food()

            if self.food is None:
                self.last_event = Event.WON # filled the whole board
                self.terminated = True
            else:
                self.last_event = Event.ATE

        else:
            tail = self.snake.pop() # drop the tail
            self.snake.insert(0, nxt) # and add the new head
            self.occupied.discard(tail)
            self.occupied.add(nxt)

            self.steps_since_food += 1

            if self.steps_since_food >= cfg.max_steps_without_food:
                self.last_event = Event.STARVED
                self.truncated = True # out of patience, but not dead
            else:
                self.last_event = Event.MOVE

        self.steps += 1

        outcome = StepOutcome(
            event=self.last_event,
            prev_dist_norm=prev_dist,
            next_dist_norm=self._dist_norm(self.snake[0], self.food),
        )
        reward = compute_reward(outcome, cfg.rewards)

        if self.done:
            self._needs_reset = True # the next step() will raise until reset() is called

        return self.observe(), reward, self.terminated, self.truncated, self._info()

    def _info(self) -> dict:
        return {
            "event": self.last_event,
            "reason": self.last_event.value,
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake),
            "crash_cell": self.crash_cell,
        }

    def _regions(self, head: tuple[int, int]):
        """Room available and tail reachability for all three moves, in one pass.

        Returns (counts, tails, reach), each three long, in the order straight, left, right.
        """
        width = self.cfg.grid_w
        height = self.cfg.grid_h

        # Mark the body. The tail is left clear because it steps out of the way as the head
        # arrives, which is what _fatal() already assumes and these two have to agree.
        self._blocked[:] = 0
        for body_x, body_y in self.snake:
            self._blocked[body_x + body_y * width] = 1

        tail_x, tail_y = self.snake[-1]
        tail = tail_x + tail_y * width
        self._blocked[tail] = 0

        # Where each move would put the head, or -1 if that move cannot be made at all.
        for slot, action in enumerate((ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT)):
            step = DIRECTIONS[turn(self.heading, action)]
            cell_x = head[0] + step[0]
            cell_y = head[1] + step[1]

            if not (0 <= cell_x < width and 0 <= cell_y < height):
                self._entries[slot] = -1 # off the board
            elif self._blocked[cell_x + cell_y * width]:
                self._entries[slot] = -1 # into the body
            else:
                self._entries[slot] = cell_x + cell_y * width

        counts, tails = label_regions(self._blocked, self._entries, tail, len(self.snake) + 1,
                                      width, height, self._label, self._sizes, self._stack)
        reach = reach_counts(self._blocked, self._entries, REACH_DEPTH,
                             width, height, self._seen, self._queue)

        return counts, tails, reach

    # -- the observation ----------------------------------------------------

    def _dist_norm(self, head: tuple[int, int], food: tuple[int, int] | None) -> float:
        """Manhattan distance from head to food, scaled roughly into [0, 1]."""
        if food is None:
            return 0.0 # no food left

        distance = abs(food[0] - head[0]) + abs(food[1] - head[1])

        return distance / (self.cfg.grid_w + self.cfg.grid_h)

    def observe(self) -> np.ndarray:
        """Build the 14 numbers the network sees. Everything is relative to the heading."""
        cfg = self.cfg
        obs = np.zeros(OBS_DIM, dtype=np.float32) # a fresh array every call, never reused
        head = self.snake[0]

        # Is each of the three moves immediately fatal?
        obs[I_DANGER_STRAIGHT] = float(self._would_die(head, turn(self.heading, ACTION_STRAIGHT)))
        obs[I_DANGER_LEFT] = float(self._would_die(head, turn(self.heading, ACTION_LEFT)))
        obs[I_DANGER_RIGHT] = float(self._would_die(head, turn(self.heading, ACTION_RIGHT)))

        # Where is the food, from the snake's point of view?
        if self.food is not None:
            ahead = DIRECTIONS[self.heading] # unit vector pointing where we face
            rightward = DIRECTIONS[(self.heading + 1) % 4] # and 90 degrees clockwise of that

            dx = self.food[0] - head[0]
            dy = self.food[1] - head[1]

            forward = dx * ahead[0] + dy * ahead[1] # how far ahead the food is
            lateral = dx * rightward[0] + dy * rightward[1] # how far to the right it is
            scale = max(cfg.grid_w, cfg.grid_h)

            obs[I_FOOD_AHEAD] = float(forward > 0)
            obs[I_FOOD_BEHIND] = float(forward < 0)
            obs[I_FOOD_LEFT] = float(lateral < 0)
            obs[I_FOOD_RIGHT] = float(lateral > 0)
            obs[I_FOOD_FORWARD] = float(np.clip(forward / scale, -1.0, 1.0))
            obs[I_FOOD_LATERAL] = float(np.clip(lateral / scale, -1.0, 1.0))

        # How big am I and how hungry?
        obs[I_LENGTH_FRAC] = len(self.snake) / (cfg.grid_w * cfg.grid_h)
        obs[I_HUNGER_FRAC] = min(self.steps_since_food / cfg.max_steps_without_food, 1.0)

        # Room available down each move, and whether the tail is still reachable after it. One
        # pass splits the empty cells into regions, then both answers are lookups. 1.0 room means
        # "enough for my whole body".
        counts, tails, reach = self._regions(head)
        budget = len(self.snake) + 1

        obs[I_FREE_STRAIGHT] = counts[0] / budget
        obs[I_FREE_LEFT] = counts[1] / budget
        obs[I_FREE_RIGHT] = counts[2] / budget

        obs[I_TAIL_STRAIGHT] = tails[0]
        obs[I_TAIL_LEFT] = tails[1]
        obs[I_TAIL_RIGHT] = tails[2]

        # How open each move is close up. free_* above goes flat as soon as the three moves
        # share a region, so this is what separates them for most of a long game.
        obs[I_REACH_STRAIGHT] = reach[0] / REACH_MAX
        obs[I_REACH_LEFT] = reach[1] / REACH_MAX
        obs[I_REACH_RIGHT] = reach[2] / REACH_MAX

        return obs
