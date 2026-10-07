"""
environment.py — THE WORLD (step 1 of the cart-pole MLP project)
=================================================================

In reinforcement learning there are two characters in a loop:

    AGENT  --acts-->  ENVIRONMENT  --(next state, reward)-->  AGENT

This file IS the environment: the track, the cart, the pole, the
physics that move them, and the scoring system. The agent (a DQN
brain built in step 2) will only ever talk to this file through two
methods, following the standard "Gym API" contract:

    state                  = env.reset()                    # new episode
    state, reward, done, info = env.step(action)            # one push

WHAT CHANGED vs PROJECT 02 (the Q-table cart-pole)?
----------------------------------------------------
Nothing in the physics. Same masses, same track (±2.4 m), same pole
limit (±12°), same +1-per-step reward, same 500-step cap. That is
deliberate: keeping the world identical means the two brains
(Q-table vs MLP) can be compared fairly — same game, different brain.

What changed is what the brain DOES with the state:

    02 Q-table brain:  4 floats -> binner -> bucket "zip code"
                       -> one row of a 7,680-row table.
    03 DQN brain:      4 floats -> straight into an MLP
                       (4 inputs -> hidden layers -> 2 Q-values).

A neural net eats continuous numbers natively, so the StateBinner
from 02 is GONE. No buckets, no clipping, no round()-vs-int()
boundary dust. The raw floats ARE the network input (later
normalized inside agent.py, not here — the world stays honest).

The state is still four continuous numbers (real decimals):

    (x, x_dot, theta, theta_dot)
      |     |       |        └── how fast the pole is tipping (rad/s)
      |     |       └── pole angle: 0 = perfectly upright (radians)
      |     └── cart speed (m/s)
      └── cart position on the track (meters)

Try it yourself (random agent demo — watch it faceplant):

    python environment.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import math    # sin/cos for the pendulum equations, radians<->degrees
import random  # reset() start-noise AND the demo's random pushes below


# ==============================================================
# 1. WHAT THE WORLD IS MADE OF  (the physics constants)
# ==============================================================
# Every number below is a physical fact about our little universe.
# The agent can NEVER change them — it only picks a direction, and
# the equations decide the rest.
#
# Units are SI (meters, kilograms, seconds) and match gym's CartPole,
# so Rosa's results stay comparable across projects 02 and 03.
GRAVITY = 9.8           # m/s² — pulls the pole's tip down. When the
                        # pole leans, that pull tips it FURTHER: an
                        # upright pole is a knife-edge equilibrium.
MASS_CART = 1.0         # kg — the box the pole stands on
MASS_POLE = 0.1         # kg — light pole, so the cart can boss it around
POLE_HALF_LENGTH = 0.5  # m — hinge to the pole's MIDDLE. The pole is
                        # 1 m long; its weight acts at the center, so
                        # the equations care about this half-distance.
FORCE_MAG = 10.0        # N — how hard ONE push shoves the cart. Fixed
                        # on purpose: the agent chooses a DIRECTION,
                        # not a strength (only 2 actions, forever).
DT = 0.02               # s — one tick of the physics clock. 50 ticks
                        # per second; 500 steps = 10 simulated seconds.

X_LIMIT = 2.4                   # m — track half-width. |x| past this
                                # and the cart has left the track.
ANGLE_LIMIT_DEG = 12.0          # degrees the pole may lean...
ANGLE_LIMIT = math.radians(ANGLE_LIMIT_DEG)  # ...stored in radians,
                                # because math.sin/cos speak radians.
                                # Degrees are ONLY for printing to us.
MAX_STEPS = 500         # episode cap — and here the GOOD ending: still
                        # upright at step 500 means you won.

START_NOISE = 0.05      # reset() scatters each of the 4 state numbers
                        # across ±0.05 — a perfectly balanced start
                        # would be a lie: real episodes begin mid-wobble.

# Combinations the equations reuse — computed ONCE here instead of
# on every single step (50 times per simulated second):
TOTAL_MASS = MASS_CART + MASS_POLE              # 1.1 kg
POLE_MASS_LENGTH = MASS_POLE * POLE_HALF_LENGTH  # 0.05 kg·m


# ==============================================================
# 2. WHAT THE AGENT CAN DO  (the "action space")
# ==============================================================
# Just two buttons. Each one applies FORCE_MAG for one tick:
# action -> force -> acceleration -> (speed changes) -> (position
# changes). The agent steers ACCELERATION here, not position — which
# is why momentum and overshooting exist in this project.
LEFT = 0    # shove the cart LEFT   (force = -10 N)
RIGHT = 1   # shove the cart RIGHT  (force = +10 N)

# Friendly names for logs ("step 3: RIGHT" reads better than "1").
ACTION_NAMES = {LEFT: "LEFT", RIGHT: "RIGHT"}

N_ACTIONS = 2   # handy for the agent later: the MLP's output layer
                # will have exactly this many neurons (one Q per action).


# ==============================================================
# 3. HOW THE WORLD SCORES  (the "reward function")
# ==============================================================
# The simplest reward we have ever written — and deliberate:
#
#     +1 for EVERY step the pole is still up. Episode over = stop.
#
# So an episode's total return = how many steps it survived, and
# 500 is the maximum score. Why not one big +500 for surviving all
# 500? Because that is a SPARSE reward: until the agent is already
# great, every attempt scores 0 and there is nothing to learn from.
# With +1/step, even a hopeless 17-step episode teaches "17 > 8".
#
# Note there is no penalty for falling — we do not need one. Falling
# stops the +1 stream, and gamma turns "rewards stop soon" into
# "low future value". (Step 3's train.py may still SHAPE what the
# brain is TAUGHT on the fatal tick — but environment.py always
# reports the honest +1/tick, so charts never lie.)
REWARD_PER_STEP = 1.0


# ==============================================================
# 4. THE ENVIRONMENT CLASS — the world in code
# ==============================================================
class CartPole:
    """
    A pole hinged on a cart: keep the pole upright by pushing the
    cart left/right, following the Gym API contract:

        state               = reset()
        state, reward, done, info = step(action)

    The DQN agent in step 2 will feed `state` (raw floats) straight
    into its MLP — no bucketing step in between.
    """

    def __init__(self, max_steps=MAX_STEPS):
        """Set up the (unchanging) physics; reset() starts episodes.

        Args: max_steps — episode time cap (500 by default; tests
              shrink it to end episodes faster).
        """
        self.max_steps = max_steps

        # Dynamic state — these change as the episode plays out, so
        # they start unset and are (re)initialized by reset().
        # Names match the physics: _dot means "the time derivative"
        # (x_dot = dx/dt = speed — Newton's notation).
        self.x = None          # cart position      (m)
        self.x_dot = None      # cart speed         (m/s)
        self.theta = None      # pole angle          (rad, 0 = upright)
        self.theta_dot = None  # tipping speed       (rad/s)
        self.steps = 0         # actions taken this episode
        self.done = False      # has the episode ended?

    @property
    def state(self):
        """The agent's view of the world: exactly four floats.

        In 02 this tuple went into a binner. In 03 it goes into
        the MLP as a 4-element input vector (see agent.py, which
        converts it with torch.tensor(state, dtype=torch.float32)).
        """
        return (self.x, self.x_dot, self.theta, self.theta_dot)

    # ----------------------------------------------------------
    # 4a. reset() — start a fresh episode
    # ----------------------------------------------------------
    def reset(self):
        """
        Put everything back to square one and return the initial state.

        In RL an "episode" = one full attempt (upright → fell, or
        survived to the cap). The agent calls this AFTER every
        finished episode.

        Returns: state — (x, x_dot, theta, theta_dot), each nudged
                 by a tiny random amount so no two episodes start
                 identical. Close to balanced, never exactly so.
        """
        self.x = random.uniform(-START_NOISE, START_NOISE)
        self.x_dot = random.uniform(-START_NOISE, START_NOISE)
        self.theta = random.uniform(-START_NOISE, START_NOISE)
        self.theta_dot = random.uniform(-START_NOISE, START_NOISE)
        self.steps = 0
        self.done = False
        return self.state

    # ----------------------------------------------------------
    # 4b. step(action) — the heart of the environment
    # ----------------------------------------------------------
    def step(self, action):
        """
        Apply one push and let the physics answer.

        Args:    action — 0 (LEFT) or 1 (RIGHT)
        Returns: (state, reward, done, info)

            state  — the four numbers AFTER the physics ran
            reward — +1.0 for surviving this tick (yes, even the
                     tick that ends the episode — it still happened)
            done   — True if the episode is over (pole fell / cart
                     out / reached the time cap)
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
        if action not in (LEFT, RIGHT):
            raise ValueError(
                f"Invalid action {action!r}: use 0=LEFT, 1=RIGHT"
            )

        # --- 1. THE PHYSICS — accelerate, then integrate -----------------
        # This is the entire "world simulator": three lines of algebra,
        # then we step the clock forward by DT (0.02 s).
        #
        # The closed forms below are the classic cart-pole equations
        # (the benchmark problem from Sutton & Barto's 1983 paper).
        # Intuition before symbols:
        #
        #   x_acc     — sideways acceleration of the CART
        #   theta_acc — how fast the pole's TILT accelerates
        #
        # What feeds them:
        #   * our push (±10 N) accelerates the cart directly,
        #   * gravity pulls the pole's tip down — the further it
        #     leans, the harder it pulls (that is the whole game),
        #   * accelerating the cart rocks the pole, like stepping
        #     on the gas makes a hanging pendulum swing back,
        #   * a spinning pole tugs the cart sideways (the
        #     theta_dot² · sin(theta) term).
        force = FORCE_MAG if action == RIGHT else -FORCE_MAG
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        # temp = the net sideways force per kg on the whole cart+pole
        # system, including that centrifugal side-tug of the spinning
        # pole. Both other accelerations are built from it.
        temp = (force + POLE_MASS_LENGTH * self.theta_dot ** 2 * sin_t) \
            / TOTAL_MASS

        # The pole's rotational acceleration around its hinge:
        #   GRAVITY * sin_t  — gravity's torque: 0 when upright,
        #                      grows as it leans (unstable!)
        #   - cos_t * temp   — the cart's sideways acceleration
        #                      rocking the pole back
        # The denominator is geometry: POLE_HALF_LENGTH scales torque
        # into acceleration, and (4/3 - ...) is a shape constant that
        # comes from the pole being a UNIFORM ROD hinged at one end
        # (its moment of inertia, folded together with the cart's
        # mass). At theta = 0 gravity contributes nothing — exactly
        # balanced, until a push or a nudge breaks the symmetry.
        theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
            POLE_HALF_LENGTH
            * (4.0 / 3.0 - MASS_CART * cos_t ** 2 / TOTAL_MASS)
        )

        # The cart's acceleration: our push (via temp), minus the
        # reaction force of the pole swinging against it (action and
        # reaction — the pole pushes the cart back).
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_t / TOTAL_MASS

        # --- integrate one tick (semi-implicit Euler) ---------------------
        # "Semi-implicit" = update the SPEEDS first, then move using the
        # speeds we just updated. Plain Euler (move first, then speed)
        # systematically pumps energy INTO wobbles; this order stays
        # stable for thousands of ticks. Small trick, big difference.
        self.x_dot += DT * x_acc
        self.x += DT * self.x_dot
        self.theta_dot += DT * theta_acc
        self.theta += DT * self.theta_dot
        self.steps += 1

        # --- 2. Score the outcome (reward function) ----------------------
        reward = REWARD_PER_STEP    # alive = +1, always
        done = False
        reason = None               # explains WHY the episode ended

        if abs(self.theta) > ANGLE_LIMIT:
            done = True
            reason = "pole_fell"
        elif abs(self.x) > X_LIMIT:
            done = True
            reason = "cart_out"
        elif self.steps >= self.max_steps:
            # Still standing at the cap — the GOOD ending. In gym
            # terms this is "truncated"; for us it is winning.
            done = True
            reason = "time_limit"

        # --- 3. Close the episode ----------------------------------------
        if done:
            self.done = True

        # info is a side-channel for humans: why it ended, how many
        # steps we have taken. The agent ignores it (it only learns
        # from state and reward).
        info = {"reason": reason, "steps": self.steps}

        return self.state, reward, done, info

    # ----------------------------------------------------------
    # 4c. render() — watch the world without a window
    # ----------------------------------------------------------
    def render(self):
        """
        Print the track, cart and pole as ASCII art, plus the state.

        A quick sanity-check tool: runs anywhere, no GUI. The pretty
        pygame window comes in step 4 — this is the debugger version.
        """
        track_width = 51   # characters across the full ±2.4 m track
        pole_rows = 5      # how tall the pole is drawn
        lean_cols = 12     # horizontal reach of a fully leaning pole

        # State line first (degrees for humans — math keeps radians).
        lines = [
            f"step {self.steps}/{self.max_steps}  "
            f"x={self.x:+.2f}m  x_dot={self.x_dot:+.2f}  "
            f"angle={math.degrees(self.theta):+.1f}deg  "
            f"spin={self.theta_dot:+.2f}"
        ]

        # Where the cart sits, as a column on the track.
        hinge = round((self.x + X_LIMIT) / (2 * X_LIMIT)
                      * (track_width - 1))
        hinge = max(1, min(track_width - 2, hinge))

        # The pole: a line of '|' leaning with sin(theta), one per
        # row above the cart. At 12° the tip sits ~2 columns off —
        # small, but you SEE which way it is falling.
        reach = math.sin(self.theta) * lean_cols
        for i in range(pole_rows, 0, -1):       # tip first (top row)
            col = hinge + round(reach * i / pole_rows)
            col = max(0, min(track_width - 1, col))
            row = [" "] * track_width
            row[col] = "|"
            lines.append("".join(row))

        # The track with the cart [C] sitting on it. Plain ASCII:
        # Windows' default console (cp1252) cannot print box-drawing
        # characters — '.' taught us to keep render() portable.
        track = ["-"] * track_width
        track[hinge - 1:hinge + 2] = list("[C]")
        lines.append("".join(track))

        print("\n".join(lines))


# ==============================================================
# 5. DEMO — a RANDOM agent trying to balance
# ==============================================================
# The environment is useless until we can run episodes through it.
# This demo plays badly on purpose (random pushes) — its job is to
# prove the rules work: physics run, the pole falls, episodes end.
# Random flailing survives ~20-40 steps; keep that number in mind
# — it is the baseline our DQN brain must crush (500).
if __name__ == "__main__":
    env = CartPole()

    state = env.reset()          # fresh episode
    total_reward = 0.0           # cumulative score, like a game total
    print("=== Random agent episode ===")
    env.render()

    while True:
        # Random policy: left or right, no thinking involved.
        # (In step 2, the DQN's MLP will replace this line.)
        action = random.randint(0, N_ACTIONS - 1)

        state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"\naction={ACTION_NAMES[action]:<5}  reward={reward:+.1f}")
        env.render()

        if done:
            break

    # ------------------------------------------------------------ #
    print("\n=== Episode finished ===")
    print(f"outcome : {info['reason']}")   # pole_fell / cart_out / time_limit
    print(f"steps   : {info['steps']} / {env.max_steps}")
    print(f"reward  : {total_reward:+.1f}")
