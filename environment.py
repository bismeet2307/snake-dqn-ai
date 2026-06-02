"""
environment.py
==============
Custom Snake game environment following the OpenAI Gym interface pattern.

The Snake agent observes an 11-dimensional state vector describing:
  - Danger directly ahead, to the left, and to the right of its current heading
  - Current movement direction (4 binary flags)
  - Relative position of food (4 binary flags)

Actions:
  0 = turn left   (relative to current heading)
  1 = go straight (no turn)
  2 = turn right  (relative to current heading)

Rewards:
  +10  : eating food
  -10  : dying (wall or self-collision)
   -1  : each step without eating (discourages stalling)
"""

import numpy as np
import random
from collections import namedtuple
from enum import Enum


# ─── Direction enum ────────────────────────────────────────────────────────────

class Direction(Enum):
    RIGHT = 0
    DOWN  = 1
    LEFT  = 2
    UP    = 3


# ─── Point helper ──────────────────────────────────────────────────────────────

Point = namedtuple("Point", ["x", "y"])


# ─── Constants ─────────────────────────────────────────────────────────────────

GRID_W   = 20          # grid columns
GRID_H   = 20          # grid rows
MAX_STEPS_NO_FOOD = GRID_W * GRID_H  # kill loop if no food eaten in this many steps


class SnakeEnv:
    """
    Lightweight Snake environment.  No rendering dependencies by default.
    Call render() only if you want ASCII output in the terminal.
    """

    def __init__(self, width: int = GRID_W, height: int = GRID_H, seed: int = None):
        self.w = width
        self.h = height
        self.rng = random.Random(seed)
        self.reset()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Start a new episode.  Returns the initial state vector."""
        # Snake starts in the middle, heading right, length 3
        cx, cy = self.w // 2, self.h // 2
        self.direction = Direction.RIGHT
        self.head = Point(cx, cy)
        self.snake = [
            self.head,
            Point(cx - 1, cy),
            Point(cx - 2, cy),
        ]
        self.score = 0
        self.steps_since_food = 0
        self._place_food()
        return self._get_state()

    def step(self, action: int):
        """
        Execute one action.

        Parameters
        ----------
        action : int  — 0=left, 1=straight, 2=right  (relative to heading)

        Returns
        -------
        state      : np.ndarray shape (11,)
        reward     : float
        done       : bool
        info       : dict with 'score'
        """
        self.steps_since_food += 1
        self._move(action)
        self.snake.insert(0, self.head)

        reward = -1           # small time penalty every step
        done   = False

        # Check death
        if self._is_collision() or self.steps_since_food > MAX_STEPS_NO_FOOD:
            reward = -10
            done   = True
            return self._get_state(), reward, done, {"score": self.score}

        # Check food
        if self.head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10
            self._place_food()
        else:
            self.snake.pop()    # move tail forward (no growth)

        return self._get_state(), reward, done, {"score": self.score}

    def render(self):
        """Print an ASCII grid to stdout (useful for debugging)."""
        grid = [["." for _ in range(self.w)] for _ in range(self.h)]
        for p in self.snake[1:]:
            if 0 <= p.x < self.w and 0 <= p.y < self.h:
                grid[p.y][p.x] = "o"
        h = self.head
        if 0 <= h.x < self.w and 0 <= h.y < self.h:
            grid[h.y][h.x] = "H"
        f = self.food
        if 0 <= f.x < self.w and 0 <= f.y < self.h:
            grid[f.y][f.x] = "F"
        border = "+" + "-" * self.w + "+"
        print(border)
        for row in grid:
            print("|" + "".join(row) + "|")
        print(border)
        print(f"Score: {self.score}  Steps: {self.steps_since_food}")

    # ── State representation ────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Encode the game into an 11-dimensional boolean vector.

        Indices 0-2 : danger straight, right, left  (relative to heading)
        Indices 3-6 : direction flags  (left, right, up, down)
        Indices 7-10: food flags       (left, right, up, down from head)
        """
        head = self.head
        d    = self.direction

        # One step ahead in each absolute direction
        pt_l = Point(head.x - 1, head.y)
        pt_r = Point(head.x + 1, head.y)
        pt_u = Point(head.x,     head.y - 1)
        pt_d = Point(head.x,     head.y + 1)

        # Map relative directions to absolute given current heading
        dir_map = {
            Direction.RIGHT: (pt_u, pt_d, pt_l, pt_r),   # (left, right, back, straight)
            Direction.LEFT:  (pt_d, pt_u, pt_r, pt_l),
            Direction.UP:    (pt_l, pt_r, pt_d, pt_u),
            Direction.DOWN:  (pt_r, pt_l, pt_u, pt_d),
        }
        turn_left, turn_right, _, straight = dir_map[d]

        state = np.array([
            # Danger: straight / right / left  (relative)
            self._is_collision(straight),
            self._is_collision(turn_right),
            self._is_collision(turn_left),

            # Current direction (one-hot)
            d == Direction.LEFT,
            d == Direction.RIGHT,
            d == Direction.UP,
            d == Direction.DOWN,

            # Food location relative to head
            self.food.x < head.x,   # food is left
            self.food.x > head.x,   # food is right
            self.food.y < head.y,   # food is above
            self.food.y > head.y,   # food is below
        ], dtype=np.float32)
        return state

    # ── Internals ──────────────────────────────────────────────────────────────

    def _place_food(self):
        """Randomly place food on an empty cell."""
        while True:
            pos = Point(
                self.rng.randint(0, self.w - 1),
                self.rng.randint(0, self.h - 1),
            )
            if pos not in self.snake:
                self.food = pos
                return

    def _is_collision(self, point: Point = None) -> bool:
        """Return True if `point` (defaults to head) is a wall or body cell."""
        p = point if point is not None else self.head
        if p.x < 0 or p.x >= self.w or p.y < 0 or p.y >= self.h:
            return True
        if p in self.snake[1:]:
            return True
        return False

    def _move(self, action: int):
        """
        Translate relative action (0=left, 1=straight, 2=right) to absolute
        direction, then advance the head by one cell.
        """
        # Clockwise order of directions
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx   = clock.index(self.direction)

        if action == 0:           # turn left  → counter-clockwise
            self.direction = clock[(idx - 1) % 4]
        elif action == 2:         # turn right → clockwise
            self.direction = clock[(idx + 1) % 4]
        # action == 1 → straight, direction unchanged

        x, y = self.head
        if   self.direction == Direction.RIGHT: x += 1
        elif self.direction == Direction.LEFT:  x -= 1
        elif self.direction == Direction.UP:    y -= 1
        elif self.direction == Direction.DOWN:  y += 1

        self.head = Point(x, y)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def state_size(self) -> int:
        return 11

    @property
    def action_size(self) -> int:
        return 3
