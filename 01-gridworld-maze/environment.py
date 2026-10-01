"""
environment.py — THE WORLD (step 1 of our RL project)
======================================================

In reinforcement learning there are two characters in a loop:

    AGENT  --acts-->  ENVIRONMENT  --(next state, reward)-->  AGENT

This file IS the environment: the maze, its walls, its rules, and its
scoring system. The agent (written in a later step) will only ever talk
to this file through two methods, following the standard "Gym API"
contract that every RL environment in the world uses:

    state        = env.reset()                      # start a new episode
    state, reward, done, info = env.step(action)    # take one action

Try it yourself (random agent demo):

    python environment.py
"""

import random  # only used by the demo at the bottom — the environment
               # itself is fully deterministic, no randomness needed


# ==============================================================
# 1. WHAT THE MAZE IS MADE OF  (the "cells")
# ==============================================================
# The maze is a grid of cells. Each cell holds one of these codes.
# We use numbers (not words) because number grids are what ML code
# and numpy work with best — but the names keep the code readable.
EMPTY = 0   # open floor — the agent can walk here
WALL  = 1   # solid wall  — blocks movement
GOAL  = 2   # the target  — reaching it wins the episode  (+10)
PIT   = 3   # a trap      — falling in loses the episode   (-10)

# The maze itself, written as text so it's easy to SEE and edit.
#   . = empty   # = wall   S = start   G = goal   P = pit
# Row 0 is the TOP row. Column 0 is the LEFT column.
MAZE_MAP = [
    "S.#....",   # row 0
    "..#.##.",   # row 1
    ".......",   # row 2
    "###.###",   # row 3   <- the only door between top and bottom
    ".......",   # row 4
    ".#####.",   # row 5   <- only the left and right edges connect
    ".....PG",   # row 6   <- the pit sits in front of the goal!
]

# One-character symbol for each cell type — used by render() so you
# can watch episodes play out as ASCII art (no window needed).
CELL_CHARS = {EMPTY: ".", WALL: "#", GOAL: "G", PIT: "P"}
START_CHAR = "S"   # shown only while the agent is still standing on it

MAZE_CHARS = {".": EMPTY, "#": WALL, "G": GOAL, "P": PIT, "S": EMPTY}


# ==============================================================
# 2. WHAT THE AGENT CAN DO  (the "action space")
# ==============================================================
# An RL agent picks an action as an INTEGER (0, 1, 2, 3) — numbers are
# what algorithms manipulate. The deltas below translate each action
# into a change of (row, col). Note the sign convention:
#   rows grow DOWNWARD  (+1 row = one step down)
#   cols grow RIGHTWARD (+1 col = one step right)
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3

ACTION_DELTAS = {
    UP:    (-1,  0),
    DOWN:  (+1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0, +1),
}

# A friendly name for every action — purely cosmetic, used in logs
# so we can print "step 3: DOWN" instead of "step 3: 1".
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}

N_ACTIONS = len(ACTION_DELTAS)   # = 4, handy for the agent later


# ==============================================================
# 3. HOW THE WORLD SCORES  (the "reward function")
# ==============================================================
# This is the environment's judging system — the numbers the agent
# will chase (or avoid) when it learns. Changing these later is a
# one-line experiment: reward design is THE lever you pull when the
# agent learns something you didn't want.
GOAL_REWARD   = +10.0   # reached the goal   → win
PIT_REWARD    = -10.0   # fell into the pit  → loss
STEP_PENALTY  =  -0.1   # every move costs a little, so the agent
                        # learns to reach the goal FAST, not wander

MAX_STEPS = 100         # episode time limit. Without it, an agent
                        # that never reaches anything would loop
                        # forever and never get feedback.


# ==============================================================
# 4. THE ENVIRONMENT CLASS — the world in code
# ==============================================================
class GridWorld:
    """
    A 7x7 maze where an agent must reach the goal without falling
    into the pit, following the Gym API contract:

        state             = reset()
        state, reward, done, info = step(action)
    """

    def __init__(self):
        """Build the static parts of the world (walls, goal, pit...).

        __init__ runs ONCE when the object is created — the maze
        layout never changes, so it's parsed here, not on every reset.
        """
        # Parse the text map into a 2D grid: grid[row][col] = cell code.
        # This lookup table is how we ask "what's at (row, col)?" fast.
        self.grid = [
            [MAZE_CHARS[ch] for ch in row] for row in MAZE_MAP
        ]
        self.rows = len(self.grid)         # 7
        self.cols = len(self.grid[0])      # 7

        # Remember where the agent starts — reset() needs it every
        # episode, and render() needs it to draw the "S" marker.
        self.start_pos = (0, 0)

        # Dynamic state — these change as the episode plays out,
        # so they start unset and are (re)initialized by reset().
        self.agent_pos = None   # where the agent currently stands
        self.steps = 0          # how many actions taken this episode
        self.done = False       # has the episode ended?

    # ----------------------------------------------------------
    # 4a. reset() — start a fresh episode
    # ----------------------------------------------------------
    def reset(self):
        """
        Put everything back to square one and return the initial state.

        In RL an "episode" = one full attempt (start → win/lose).
        The agent calls this AFTER every finished episode.

        Returns: state — the agent's current position as (row, col).
                 For our maze the state IS the position: the agent
                 knows exactly where it is (fully observable).
        """
        self.agent_pos = self.start_pos
        self.steps = 0
        self.done = False
        return self.agent_pos

    # ----------------------------------------------------------
    # 4b. step(action) — the heart of the environment
    # ----------------------------------------------------------
    def step(self, action):
        """
        Apply one action and see what the world does in return.

        Args:    action — integer 0..3 (UP/DOWN/LEFT/RIGHT)
        Returns: (state, reward, done, info)

            state  — the agent's new position after the action
            reward — a single float: immediate judgement of that
                     action (+10 win / -10 loss / -0.1 ordinary step)
            done   — True if the episode is over (win, loss, or timeout)
            info   — a dict of extra facts for humans/logging;
                     the agent does NOT learn from it
        """
        # --- Guard rails -------------------------------------------------
        # Calling step() after the episode ended is a classic bug that
        # produces confusing errors later, so we fail loudly and early.
        if self.done:
            raise RuntimeError(
                "Episode is over — call reset() before step()."
            )
        if action not in ACTION_DELTAS:
            raise ValueError(
                f"Invalid action {action!r}: use 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT"
            )

        # --- 1. Move the agent (transition dynamics) ---------------------
        # Compute where the action WOULD take us...
        dr, dc = ACTION_DELTAS[action]
        row, col = self.agent_pos
        new_pos = (row + dr, col + dc)

        # ...then check the world's rules. Deterministic movement means:
        # same action + same state = same result, every single time.
        bumped = False
        if not self._inside(new_pos) or self._cell(new_pos) == WALL:
            # Hit a wall or the edge → the agent stays put.
            # It still "spent" the turn (and pays the step penalty),
            # so bumping into walls is discouraged, not free.
            new_pos = self.agent_pos
            bumped = True

        self.agent_pos = new_pos
        self.steps += 1

        # --- 2. Score the outcome (reward function) ----------------------
        cell = self._cell(self.agent_pos)
        reward = STEP_PENALTY        # default: every step costs a little
        done = False
        reason = None                # explains WHY the episode ended

        if cell == GOAL:
            reward = GOAL_REWARD     # win!
            done = True
            reason = "goal"
        elif cell == PIT:
            reward = PIT_REWARD      # loss!
            done = True
            reason = "pit"
        elif self.steps >= MAX_STEPS:
            # Ran out of time without winning — still -0.1 for the
            # step, but the episode is forcibly closed ("truncated"
            # in Gym terminology). Counted as a failure.
            done = True
            reason = "timeout"

        # --- 3. Close the episode ----------------------------------------
        if done:
            self.done = True

        # info is a side-channel for humans: why it ended, whether we
        # bumped a wall, how many steps we've taken. The agent ignores it.
        info = {"reason": reason, "bumped": bumped, "steps": self.steps}

        return self.agent_pos, reward, done, info

    # ----------------------------------------------------------
    # 4c. render() — watch the world without a window
    # ----------------------------------------------------------
    def render(self):
        """
        Print the maze as ASCII art, with 'A' marking the agent.

        A quick sanity-check tool: runs anywhere, no GUI. The pretty
        pygame window comes in part C — this is the debugger version.
        """
        lines = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                if (r, c) == self.agent_pos:
                    row_cells.append("A")           # the agent wins
                elif (r, c) == self.start_pos and self.grid[r][c] == EMPTY:
                    row_cells.append(START_CHAR)    # remember the start
                else:
                    row_cells.append(CELL_CHARS[self.grid[r][c]])
            lines.append(" ".join(row_cells))
        print("\n".join(lines))

    # ----------------------------------------------------------
    # 4d. Small helpers (used by step/render)
    # ----------------------------------------------------------
    def _inside(self, pos):
        """True if (row, col) is within the grid borders."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _cell(self, pos):
        """What kind of cell lives at (row, col)?"""
        r, c = pos
        return self.grid[r][c]


# ==============================================================
# 5. DEMO — a RANDOM agent playing one episode
# ==============================================================
# The environment is useless until we can run episodes through it.
# This demo plays badly on purpose (random moves) — its job is to
# prove the rules work: walls block, scoring fires, episodes end.
if __name__ == "__main__":
    env = GridWorld()

    state = env.reset()          # fresh episode
    total_reward = 0.0           # cumulative score, like a game total
    print("=== Random agent episode ===")
    env.render()

    while True:
        # Random policy: pick one of the 4 actions, no thinking involved.
        # (Later, the Q-learning agent will replace this line.)
        action = random.randint(0, N_ACTIONS - 1)

        state, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"\naction={ACTION_NAMES[action]:<5}  "
            f"reward={reward:+.1f}  "
            f"state={state}  "
            f"{'WALL BUMP - stayed put' if info['bumped'] else ''}"
        )
        env.render()

        if done:
            break

    # ------------------------------------------------------------ #
    print("\n=== Episode finished ===")
    print(f"outcome : {info['reason']}")        # goal / pit / timeout
    print(f"steps   : {info['steps']} / {MAX_STEPS}")
    print(f"reward  : {total_reward:+.1f}")
