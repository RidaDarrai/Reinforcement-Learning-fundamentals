"""
environment.py — THE WORLD (step 1 of the lunar-lander-gym project)
===================================================================

Projects 01-04 hand-built every world. THIS project does something
new on purpose: the world is a REAL Gym environment — the classic
lunar lander as the pros fly it — and our job shrinks to learning
from it.

Under the hood: gymnasium (the maintained fork of OpenAI's Gym —
same API, old name deprecated) with its Box2D physics:

    LunarLander-v3   a lander even simpler than 04's to LOOK at but
                     harder to score: real rocket thrust along the
                     body axis, two legs that really touch things,
                     a 1,000-tick clock, and gym's own fuel-burn
                     shaping. No reward tuning of ours. That is the
                     whole experiment: the world stops being ours.

THE WRAPPER (this file's whole job)
-----------------------------------
Gym's modern step() returns a 5-tuple and TWO done flags:

    obs, reward, terminated, truncated, info

Our projects (01-04's loops) speak the classic 4-tuple with ONE
boolean:

    state, reward, done, info

So the wrapper folds terminated|truncated into done and hands the
state out as a plain 8-float tuple. Everything else it passes
through UNTOUCHED — no shaping on top, no re-tuned spawns. If the
agent fails to solve this, the blame lands on the AGENT, which is
exactly what the last two projects' postmortems made necessary.

THE STATE — gym's 8 floats, fed to the brain RAW
------------------------------------------------
    (x, y, vx, vy, angle, omega, legL, legR)
      x      — position, PRE-NORMALIZED by gym: 0.0 = pad center,
               +-1.0 = the screen edges
      y      — height, same habit: 0.0 = the helipad's touchdown
               line, spawn sits ~+0.9, the ground below is negative
      vx/vy  — speeds, scaled into the same ~[-1, 1] band
      angle  — tilt, RAW radians; 0 = upright, quarter-turns +/-pi/2
      omega  — spin rate, pre-scaled by gym
      legL/legR — 1.0 while that leg's tip is touching the ground

So the preprocessing lesson INVERTS this time: 04 needed a private
normalize_state() because OUR physics printed SI meters; the gym
world prints apartment-sized numbers already. No preprocessing
module ships in this project — stacking a second normalizer on
gym's would be exactly the "must EARN its place" habit turned into
a bug.

THE SCORING (gym's own anatomy — the point of the switch)
---------------------------------------------------------
    landed   EXACTLY +100 on the killer tick (gym's source sets the
             terminal, overriding shaping). The +10-per-touching-leg
             travels INSIDE the shaping instead, so a soft descent
             arrives at a total well above +100 while the terminal
             itself stays the flat classic.
    crashed  exactly -100 (both flavors: hull-death and drifting
             out of the arena pay the same flat bill, overriding
             the tick's shaping — gymnasium's own source, or the
             tests that pin it). The gravity-like shaping makes
             every falling tick cost, so a typical death LIFE
             totals -100..-30 even though the killer tick is flat.
    timeout  the 1,000-tick cap fires: truncated=True, no bonus —
             a stalled life pays its whole burn history and stops.
    Out of the four diary tags, "crashed" and "out_of_bounds" are
    BOTH exactly -100 on the tick, so the wrapper splits the
    flavors by the arena wall: |state[0]| >= 1.0 is gym's own OOB
    rule -> "out_of_bounds", any other -100 -> "crashed".

    (In gym's book a landing counts ANYWHERE on the ground — the
    pad flags are cosmetic. We keep 04's four-tag honesty: a slow
    level arrival off-pad still reads "landed"; the same bill paid
    outside the wall reads "out_of_bounds".)

ACTIONS — 4 discrete buttons, same ARGUMENT as 04, real physics:
    0 = IDLE   engines off — coast
    1 = MAIN   main engine (thrusts along the body's up-axis, so a
               tilted lander pushes SIDEWAYS too — real physics that
               04's turn-buttons had to fake)
    2 = LEFT   left-side orientation engine (tilts and nudges)
    3 = RIGHT  right-side orientation engine

THE SOLVED BAR (the classic Gym one, and this repo's target):
    average return > +200 over any 20-episode window — the very bar
    the reference this project ports reaches within ~1,500 episodes.

Try it (a scripted tour — no brain yet, just the world talking):

    python environment.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import gymnasium as gym   # the environment library itself: env
                          # registry, Box2D lander, pygame renderer

# The world this project trains on, pinned by NAME: "v3" is the
# gymnasium-era formulation of the classic lander (v2's old gym
# package is deprecated and its API dialect is frozen in time).
ENV_ID = "LunarLander-v3"

# The state contract, pinned as constants so agent.py and the tests
# read THEM (never count commas).
OBS_DIM   = 8      # floats per state — see the layout map above
N_ACTIONS = 4      # discrete buttons the brain picks from

ACTION_NAMES = {
    0: "IDLE",     # engines off — coast
    1: "MAIN",     # main engine
    2: "LEFT",     # left orientation engine
    3: "RIGHT",    # right orientation engine
}

# The wrapper decides the death FLAVOR with two named facts, both
# straight from gymnasium's lander source (pinned by our tests):
STATE_X_LIMIT = 1.0   # arena edge in STATE units: gym terminates
                      # when |state[0]| >= 1.0 — the same -100 either
                      # way, so the wrapper names the flavor by it
LANDED_CUT   = 0.0    # a terminated tick paying ABOVE this is a
                      # landing (exactly +100); every death pays
                      # exactly -100. The cut sits in the desert no
                      # real terminal tick visits — gym grades, the
                      # wrapper only splits flavors.

DEATH_REASONS = ("landed", "crashed", "out_of_bounds", "timeout")


def ending_of(reward, state, truncated):
    """Which of the four endings fired (the diary column).

    The truth table, straight from gymnasium's lander source:

        truncated                 -> "timeout"
        terminated, reward > LANDED_CUT
                                  -> "landed"
        terminated, |state[0]| >= STATE_X_LIMIT
                                  -> "out_of_bounds"
        otherwise                 -> "crashed"

    Why the two cuts are exact: gym's terminal override pays
    EXACTLY +100 on a landing and EXACTLY -100 on a death (crash
    or arena escape alike), so reward alone splits landed from
    dead, and gym's OWN arena rule (|x| >= 1.0 terminates) splits
    which flavor of dead. info carries no termination reason —
    the state is the witness, and the wrapper already holds it.
    Module-level for the same reason the other rules are: tests
    import it directly (no world needed for a truth table).
    """
    if truncated:
        return "timeout"
    if abs(state[0]) >= STATE_X_LIMIT:
        return "out_of_bounds"
    if reward > LANDED_CUT:
        return "landed"
    return "crashed"


class LunarLanderEnv:
    """The gym lander, wearing our project's API dialect.

    Composes gym's world (is-a shield: no override of internals).
    Its reset()/step() return exactly the shape 01-04's loops read.

    Seeding: gym's reset(seed=N) resets the world's private RNG.
    The wrapper seeds ONCE (the first reset) and lets the stream
    continue afterwards — a fixed seed replays the SAME sequence of
    episodes (tests pin this), while 'None' keeps every episode
    independently random (nice for demos, banned for tests).
    """

    def __init__(self, seed=None, render_mode=None):
        """Compose the world.

        Args: seed        — int, or None for "random every episode".
                            Same seed -> same episode sequence.
              render_mode — None while training; "rgb_array" makes
                            each step also produce a pixel frame
                            (step 4's window paints it + the HUD);
                            "human" would open gym's own bare
                            window (no HUD) — kept unavailable.
        """
        if render_mode not in (None, "rgb_array"):
            raise ValueError(
                "render_mode must be None or 'rgb_array' "
                "(the human window is gym's bare renderer, "
                "play.py builds the HUD'd one itself)")
        self._seed = seed
        self._seed_planted = False   # first reset() consumes the seed
        self._render_mode = render_mode   # for render()'s honesty
        self._env = gym.make(ENV_ID, render_mode=render_mode)
        self.steps = 0               # ticks in the current episode
        self.done = False            # True once the episode ended

    # ----------------------------------------------------------
    # reset() — a new episode
    # ----------------------------------------------------------
    def reset(self):
        """Start an episode; returns the first state (8-float tuple).

        Raises RuntimeError mid-episode — 04's kept contract: a
        loop that forgets to NOTICE the end must crash loudly, not
        fly a dead world.
        """
        if not self.done and self.steps > 0:
            raise RuntimeError(
                "reset() called mid-episode: read 'done' first, "
                "or fix the loop that forgot it.")
        seed = None
        if self._seed is not None and not self._seed_planted:
            seed = self._seed          # the ONE seeded reset
            self._seed_planted = True
        obs, _info = self._env.reset(seed=seed)
        self.steps = 0
        self.done = False
        return self._freeze(obs)

    # ----------------------------------------------------------
    # step() — one tick
    # ----------------------------------------------------------
    def step(self, action):
        """Advance one tick. Returns OUR classic 4-tuple:

            (state, reward, done, info)

        action: 0..3 (one of the four buttons). Anything else is
        rejected HERE, with a named message, BEFORE the physics
        sees it (tests pin the wording; gym's own would be an
        unhelpful numpy crash deep in Box2D).

        info: gym's own dict (it carries "x"/"y" — the lander's
        dashboard numbers) extended with our "outcome" tag — the
        four DEATH_REASONS, decided by the rule set below.
        """
        if self.done:
            raise RuntimeError(
                "step() called after the episode ended. reset() first.")
        self._check_action(action)
        obs, reward, terminated, truncated, info = self._env.step(action)
        state = self._freeze(obs)
        self.steps += 1
        self.done = bool(terminated or truncated)
        info = dict(info)            # attach our tag, not gym's dict
        info["outcome"] = ending_of(reward, state, bool(truncated))
        return state, float(reward), self.done, info

    # ----------------------------------------------------------
    # internals — small, named, testable
    # ----------------------------------------------------------
    @staticmethod
    def _freeze(obs):
        """gym's float32 array -> a plain 8-float tuple.

        numpy crosses this ONE boundary and never under it: the
        brain in agent.py talks tuples of floats, same as 02-04.
        """
        obs = tuple(float(v) for v in obs)
        if len(obs) != OBS_DIM:
            raise RuntimeError(f"world sent {len(obs)} floats, "
                               f"the contract says {OBS_DIM}")
        return obs

    @staticmethod
    def _check_action(action):
        """Range check with a NAMED message (tests pin the wording)."""
        if not isinstance(action, int) or isinstance(action, bool) \
                or not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be an int in "
                             f"0..{N_ACTIONS - 1}, got {action!r}")

    def render(self):
        """Hand the pixels out (only sensible with render_mode
        "rgb_array"; step 4's window blits them under its HUD).

        In training mode (None) this returns None WITHOUT waking
        gym's warn-prone renderer — the world is not obligated to
        draw itself, and step 4 does that anyway.
        """
        if self._render_mode is None:
            return None
        return self._env.render()

    def close(self):
        """Release the window/physics (tests + demos call this)."""
        self._env.close()


# ==============================================================
# DEMO — the world, seen through our API
# ==============================================================
if __name__ == "__main__":
    print("=== the world through our API ===")
    print("id        :", ENV_ID)
    print("state     :", OBS_DIM, "floats  (x, y, vx, vy, angle, omega, legL, legR)")
    print("actions   :", N_ACTIONS, "->", " | ".join(ACTION_NAMES[a] for a in range(N_ACTIONS)))
    print("endings   :", " | ".join(DEATH_REASONS))

    print()
    print("=== A: do nothing, let gravity speak (until done) ===")
    lander = LunarLanderEnv(seed=0)
    lander.reset()
    ret = 0.0
    for tick in range(200):
        state, reward, done, info = lander.step(0)
        ret += reward
        if done:
            break
    print(f"idle for {lander.steps} ticks -> return {ret:+.2f}")
    print(f"final state : y {state[1]:+.3f}, vy {state[3]:+.3f}")
    print(f"outcome     : {info['outcome']}")
    lander.close()

    print()
    print("=== B: fire each engine once from the same spawn ===")
    for label, action in (("IDLE ", 0), ("MAIN ", 1),
                          ("LEFT ", 2), ("RIGHT", 3)):
        lander = LunarLanderEnv(seed=7)
        lander.reset()
        state, reward, done, info = lander.step(action)
        print(f"{label} tick 1: reward {reward:+8.2f}   state[4] (angle) {state[4]:+.4f}")
        lander.close()

    print()
    print("=== C: a full episode, scripted tumble (all MAIN) ===")
    lander = LunarLanderEnv(seed=3)
    lander.reset()
    ret = 0.0
    while True:
        state, reward, done, info = lander.step(1)
        ret += reward
        if done:
            break
    print(f"fired MAIN every tick -> {lander.steps} ticks, "
          f"return {ret:+.2f}, outcome {info['outcome']}")
    lander.close()
