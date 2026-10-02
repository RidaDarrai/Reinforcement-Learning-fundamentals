"""
agent.py — THE BRAIN (step 2 of our cart-pole project)
=======================================================

Good news: the brain from project 1 carries over almost unchanged —
epsilon-greedy Q-learning, same math, same three knobs (alpha,
gamma, epsilon; see 01-gridworld-maze/agent.py for the full sermon).

Bad news: this brain needs EYES.

THE PROBLEM THIS FILE SOLVES FIRST
----------------------------------
The maze knew WHERE it was: (row, col) — two small integers, ready
to use as a table index. Cart-pole reports four CONTINUOUS floats:

    (x, x_dot, theta, theta_dot) = (0.0044, 0.2208, -0.0171, -0.8571)

A table cannot have a row for every possible number — there are
infinitely many (0.0044 and 0.00441 are "the same moment" for our
purposes). So we BUCKET them:

    StateBinner: 4 floats  ->  4 bucket numbers (a "zip code")

    (0.0044, 0.2208, -0.0171, -0.8571)  ->  (5, 4, 6, 4)
          |        |        |       |
         mid      mid      mid     mid    = "centered, upright,
                                             at rest" — THE home
                                             bucket of a balanced pole

Two moments that land in the same bucket are the SAME STATE to the
brain. That's how infinite becomes finite:

    10 x 8 x 12 x 8 = 7,680 rows instead of infinitely many.

THE TRADE (say it out loud so you remember it)
----------------------------------------------
  buckets too FEW  -> different moments blur together ("leaning 1°
                      or 11° is all the same" — disaster)
  buckets too MANY -> table too sparse; each row learns from a
                      handful of visits (memorizing, not learning)

Our split (DEFAULT_RANGES, tuned to OUR environment):
    x        10 buckets  over ±2.4 m    -> 0.48 m each
    x_dot     8 buckets  over ±3 m/s    -> 0.75 m/s each
    theta    12 buckets  over ±12 deg   -> 2° each (finest
                                           resolution near the
                                           cliff where episodes die)
    theta_dot 8 buckets  over ±4 rad/s  -> 0.5 rad/s each

Values OUTSIDE a range (a pole at 20° during its death thump, a
cart at 9 m/s) clip into the nearest edge bucket — the world may
leave the lines we drew, the table must not crash.

OPTIMISM AS EXPLORATION (the bug this fixes — remember it)
-----------------------------------------------------------
Our first trained agents kept dying THE EXACT SAME WAY, because
the Q-value of an action nobody had tried yet stayed at 0.0, and
0.0 beats any value in a death zone (where everything is bad).
The untried RIGHT button never got a chance.

The fix is a one-line change with a big idea: fresh rows start at
q_init = 50.0 (half of "perfect play" for gamma=0.99), NOT 0.0 —
every unvisited possibility is TOLD it's great, and reality knocks
it down on contact. An action you've never tried now looks
ATTRACTIVE instead of worthless, so it gets tried. Optimism is
exploration you don't have to pay for with random moves.

API UNCHANGED FROM PROJECT 1:

    choose_action(state)              -> LEFT (0) or RIGHT (1)
    update(state, action, reward, next_state, done)  -> nudge Q

...where `state` is the raw four floats: the binner lives INSIDE
the brain (self.binner), so callers never think about buckets.

ONE MORE THING SAVED ALONGSIDE THE SCORECARD
---------------------------------------------
save() also writes the binner's config AND the q_init optimism
level. A brain without its eyes (or its attitude) is useless:
loading must bucket identically, or today's zip codes point at
yesterday's wrong memories.
"""

import json
import math
import random

# Friendly defaults (same knobs as project 1).
DEFAULT_ALPHA = 0.1     # learning rate: trust experience gradually
DEFAULT_GAMMA = 0.99    # discount: future rewards still count
DEFAULT_EPSILON = 1.0   # start as a pure explorer

# The starting value of a brand-new row (see "OPTIMISM AS
# EXPLORATION" at the top): half of perfect play for gamma=0.99,
# where even a flawless run can't score much past ~100.
DEFAULT_Q_INIT = 50.0

# How many buckets per dimension -> table size 10*8*12*8 = 7,680.
DEFAULT_N_BINS = (10, 8, 12, 8)

# The lines we draw on the world. (These must MATCH our
# environment's limits: ±2.4 m track, ±12° pole — test_agent.py
# imports environment.py and asserts exactly that, so nobody can
# change one file and forget the other.)
DEFAULT_RANGES = (
    (-2.4, 2.4),                          # x: the whole track (m)
    (-3.0, 3.0),                          # x_dot: cart speed (m/s)
    (-math.radians(12), math.radians(12)),  # theta: pole tilt (rad)
    (-4.0, 4.0),                          # theta_dot: spin (rad/s)
)


class StateBinner:
    """Turn four continuous floats into a bucket tuple (the zip code).

    Each dimension is cut into n equal slices over [low, high]:

        bucket = round((value - low) / (high - low) * n)
        bucket = clamp(bucket, 0, n - 1)      # out-of-range -> edge

    Why round() and not int()? Binary dust: +6° is mathematically
    the exact boundary 9.0, but floats compute 8.999999999999998 —
    int() would floor it into bucket 8 and your hand-math would
    never match the code. round() heals the dust. (The old
    project-1 maze never hit this: its state was integers.)

    So the result is always n small ints — exactly what a table
    needs as a key.
    """

    def __init__(self, n_bins=DEFAULT_N_BINS, ranges=DEFAULT_RANGES):
        if len(n_bins) != len(ranges):
            raise ValueError("n_bins and ranges must have the same length")
        self.n_bins = tuple(int(n) for n in n_bins)
        for n in self.n_bins:
            if n < 1:
                raise ValueError("every dimension needs at least 1 bucket")
        self.ranges = tuple((float(lo), float(hi)) for lo, hi in ranges)
        for lo, hi in self.ranges:
            if hi <= lo:
                raise ValueError(f"empty range [{lo}, {hi}]")

    @property
    def n_states(self):
        """How many rows the table COULD hold (we still only store
        the ones we actually visit — dict, not giant array)."""
        n = 1
        for b in self.n_bins:
            n *= b
        return n

    def __call__(self, state):
        """4 floats -> 4 bucket ints, e.g. (0,0,0,0) -> (5,4,6,4)."""
        key = []
        for value, n, (lo, hi) in zip(state, self.n_bins, self.ranges):
            bucket = round((float(value) - lo) / (hi - lo) * n)
            bucket = min(max(bucket, 0), n - 1)   # clip to the map
            key.append(bucket)
        return tuple(key)

    def config(self):
        """Serializable recipe — written into the save file so a
        loaded brain buckets exactly like the brain that trained."""
        return {
            "n_bins": list(self.n_bins),
            "ranges": [[lo, hi] for lo, hi in self.ranges],
        }


class QAgent:
    """Tabular Q-learning agent: scorecard + update rule + eyes."""

    def __init__(self, n_actions, alpha=DEFAULT_ALPHA,
                 gamma=DEFAULT_GAMMA, epsilon=DEFAULT_EPSILON,
                 binner=None, q_init=DEFAULT_Q_INIT):
        """
        Args:
            n_actions: how many distinct moves exist (2 here:
                       LEFT=0, RIGHT=1 — the agent never learns them,
                       it only counts them)
            alpha:   learning rate   (how hard we re-learn per step)
            gamma:   discount factor (how much future counts now)
            epsilon: exploration rate (probability of a random move)
            binner:  custom StateBinner (default: the standard
                     ±2.4 m / ±12° split described at the top)
            q_init:  starting value of brand-new rows (default 50.0
                     = strategic optimism; pass 0.0 for textbook
                     "ignorance starts at zero")
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_init = q_init
        self.binner = binner if binner is not None else StateBinner()

        # THE SCORECARD: bucket-key -> list of n_actions floats.
        # Dict (not a pre-filled array) because most of the 7,680
        # buckets are never visited. Missing rows auto-create in
        # get_q() filled with q_init — optimism, not ignorance
        # (see the module docstring for why that saved us).
        self.q_table = {}

    # ------------------------------------------------------------------
    # 1. THE SCORECARD — get (and auto-create) the row for a state
    # ------------------------------------------------------------------
    def get_q(self, state):
        """Return this state's Q-values, creating the row if new.

        `state` is the RAW four floats — the binner converts them
        to a bucket key first. New rows start [q_init, q_init]:
        50.0 of pure optimism, so unseen possibilities look worth
        trying (reality lowers the number on the first visit).
        """
        key = self.binner(state)
        if key not in self.q_table:
            self.q_table[key] = [self.q_init] * self.n_actions
        return self.q_table[key]

    # ------------------------------------------------------------------
    # 2. THE POLICY — decide what to do
    # ------------------------------------------------------------------
    def choose_action(self, state, explore=True):
        """Pick an action: sometimes random, otherwise best guess.

        Epsilon-greedy, the classic explore/exploit compromise:

            with probability epsilon -> random  (EXPLORE: wander,
                try dumb things, discover what actually balances)
            with probability 1-eps   -> current BEST action
                (EXPLOIT: cash in what we think we know)

        explore=False forces pure exploitation — the mode we use to
        WATCH a trained agent perform (no more mistakes).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # exploration

        q = self.get_q(state)
        # Exploit: highest Q. Ties break to the lowest index —
        # deterministic and testable (project 1 lesson, reused).
        return max(range(self.n_actions), key=lambda a: q[a])

    # ------------------------------------------------------------------
    # 3. THE LEARNING RULE — file one experience away
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """Nudge Q(state,action) toward what that experience suggests.

        Args:
            state:     where we were (raw 4 floats)
            action:    what we did there (0 or 1)
            reward:    what the environment said it was worth
            next_state:where we ended up (raw 4 floats)
            done:      did the episode end? (crucial — see below)

        Two flavors of target:

            done=True : the future is EMPTY. A fall is the whole
                        truth: target = reward. (No daydreaming
                        about rewards that can never arrive —
                        in cart-pole the "fall" reward is still +1,
                        the last tick of survival.)

            done=False: we also inherited the best future available
                        from next_state:
                        target = reward + gamma * max Q(next_state).
                        This chaining is how "+1 forever" teaches the
                        nudges from fifty ticks ago.

        The learning step:

            Q(s,a) += alpha * (target - Q(s,a))
                      |_____|      |________|
                       learning   how wrong      -> the ERROR we fix
                         rate      we were
        """
        q = self.get_q(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_q(next_state))

        error = target - q[action]          # how wrong were we?
        q[action] += self.alpha * error     # fix it gradually

    # ------------------------------------------------------------------
    # 4. MEMORY OUTSIDE THE BRAIN — save / load eyes + scorecard
    # ------------------------------------------------------------------
    def save(self, path):
        """Write the scorecard AND the binner recipe to JSON —
        train once, watch forever."""
        data = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            # Travel with the brain: the optimism level decides what
            # ANY future unseen row gets created with.
            "q_init": self.q_init,
            # Without these, a loaded brain would bucket the world
            # differently and read the WRONG rows. Eyes travel
            # with the scorecard.
            "binner": self.binner.config(),
            # JSON can't use tuples as keys: (5,4,6,4) becomes
            # the string "5,4,6,4" — parsed back in load().
            "q_table": {
                ",".join(str(b) for b in key): values
                for key, values in self.q_table.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path):
        """Rebuild an agent (eyes included) from save()."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Files saved before binner existed fall back to the
        # default eyes (an old brain still loads).
        binner = None
        binner_cfg = data.get("binner")
        if binner_cfg:
            binner = StateBinner(
                n_bins=binner_cfg["n_bins"],
                ranges=[tuple(r) for r in binner_cfg["ranges"]],
            )

        agent = cls(
            n_actions=data["n_actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            binner=binner,
            # Files saved before this field existed get plain
            # zero-ignorance (they were trained that way anyway).
            q_init=data.get("q_init", 0.0),
        )
        for key, values in data["q_table"].items():
            bucket_key = tuple(int(x) for x in key.split(","))
            agent.q_table[bucket_key] = [float(v) for v in values]
        return agent


# ------------------------------------------------------------------
# Tiny demo: the eyes first, then one Q-value learning twice.
# (The real training loop arrives in step 3 — train.py.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    binner = StateBinner()
    print("bucket( 0, 0, 0deg, 0)  ->", binner((0.0, 0.0, 0.0, 0.0)))
    print("  hand: x=(0+2.4)/4.8*10=5, spin=4, tilt=(0+12)/2=6, ..."
          "  = (5, 4, 6, 4)  = the balanced home bucket")
    print("bucket( 0, 0,+6deg, 0)  ->",
          binner((0.0, 0.0, math.radians(6), 0.0)))
    print("  hand: tilt=(6+12)/2=9 -> (5, 4, 9, 4)")
    print("bucket( 0, 0,+20deg, 0) ->",
          binner((0.0, 0.0, math.radians(20), 0.0)))
    print("  past the cliff -> clipped to the edge bucket 11")
    print(f"table could hold {binner.n_states:,} rows; "
          f"we only ever store visited ones\n")

    # The optimism story first — a brand-new brain's first look at
    # ANY state: promising, not empty.
    fresh = QAgent(n_actions=2)
    print("\ndefault new row       :", fresh.get_q((0.0, 0.0, 0.0, 0.0)),
          "  <- q_init optimism")

    # Now the same hand-checks as project 1 — state is the raw float
    # tuple, the binner works behind the scenes. q_init=0.0 keeps
    # the arithmetic textbook-clean (like project 1's ignorance).
    agent = QAgent(n_actions=2, alpha=0.5, gamma=0.9, q_init=0.0)
    state = (0.0, 0.0, 0.0, 0.0)
    balanced = (0.1, 0.0, math.radians(1), 0.0)

    print("q_init=0 start        :", agent.get_q(state))
    # Terminal tick: reward +1 (last survival point), episode over.
    # Hand check: 0 + 0.5 * (1 - 0) = 0.5
    agent.update(state, action=1, reward=+1.0,
                 next_state=balanced, done=True)
    print("after one fatal step  :", agent.get_q(state))

    # Continuing tick: reward +1, and the future (best next = 2.0)
    # counts too. Hand check: 0.5 + 0.5 * ((1 + 0.9*2.0) - 0.5)
    #                       = 0.5 + 0.5 * (2.8 - 0.5) = 1.65
    agent.get_q(balanced)[0] = 2.0
    agent.update(state, action=1, reward=+1.0,
                 next_state=balanced, done=False)
    print("after one alive step  :", agent.get_q(state))
