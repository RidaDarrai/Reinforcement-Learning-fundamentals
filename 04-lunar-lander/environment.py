"""
environment.py — THE WORLD (step 1 of the lunar lander project)
===============================================================

In reinforcement learning there are two characters in a loop:

    AGENT  --acts-->  ENVIRONMENT  --(next state, reward)-->  AGENT

This file IS the environment: the lander, the Moon's gravity, the
engines, the ground with its landing pad, and the scoring system.
The agent (a DQN brain, rewritten in step 2 of THIS project — no
imports from 01-03) only ever talks to this file through two
methods, the standard "Gym API" contract:

    state                     = env.reset()          # new episode
    state, reward, done, info = env.step(action)     # one tick

WHAT KIND OF WORLD IS THIS (vs the cart-pole projects)?
------------------------------------------------------
Cart-pole was ONE degree of freedom: push the cart left/right. This
world is a 3-DOF RIGID BODY — the lander slides (x, y) AND rotates
(theta) independently. That is what "landing" actually means:
arrive at the pad slow, level, and legs-first, all at once.

Contacts are DETECTED, not SOLVED. There is no physics engine
(no Box2D): each tick we ask "did anything poke below the ground
line?" and the rules in section 4 decide land-or-crash from that.
The lander never bounces or rests — it falls THROUGH the ground
unless the episode ends. Every line stays hand-checkable, and it is
plenty for the real decision problem: "how do I fly down gently?"

SIDE EFFECT OF THE DETECTION-ONLY MODEL (a feature, honestly): a
slow-ish touch (the band between the landing limit and the impact
limit) leaves the lander sinking — a quick main-engine burn can
still arrest it and land, legs buried a few centimeters. The agent
can DISCOVER "touch, then settle" behavior. Section 4 numbers it.

THE STATE — EIGHT HONEST FLOATS
-------------------------------
    (x, y, vx, vy, theta, omega, legL, legR)
      |   |    |    |      |      |      └───┴ 1 if that leg's tip
      |   |    |    |      |      └ spin rate (rad/s)     is in the
      |   |    |    |      └ tilt angle, 0 = upright      ground now
      |   |    |    └ vertical speed (+ up)
      |   |    └ horizontal speed
      |   └ height above the ground (m)
      └ sideways offset from the PAD CENTER (m)

Conventions (math space: x right, y UP, angles counterclockwise):
  +theta leans the nose toward -x ("left"); the body's up-axis is
  (-sin theta, cos theta). The pygame window in step 4 maps this
  space to the screen (y flipped once, at the draw boundary only).

The raw floats go straight to the MLP (same lesson as 03: no
binner); normalize_state() below is the ONLY preprocessing.

THE REWARD is no longer "+1 per tick" — this world has a real
reward anatomy (terminal / shaping / fuel), all computed HERE and
broken out per component in info["components"] so the charts in
step 5 can never lie about money.

Try it (a scripted demo — watch the rules fire):

    python environment.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import math    # sin/cos for the thrust direction, hypot for distances,
               # radians for the landing angle limit
import random  # reset()'s seeded spawn scatter (a PRIVATE RNG — see
               # class note: no global-seed side effects)


# ==============================================================
# 1. THE CLOCK AND THE MOON  (physics constants)
# ==============================================================
# Units are SI (meters, seconds). The agent can NEVER change these —
# it only picks one of the four buttons in section 3.
DT = 0.02               # s per tick of the physics clock (50 ticks/s).
                        # 1000 steps = 20 s of flight — a hover stays
                        # under it; a 'never land' policy times out.
MAX_STEPS = 1000        # episode cap (see timeout in section 4).

GRAVITY = 1.62          # m/s² — the real Moon's pull (1/6 of Earth's).
                        # Crime of the classic arcade: fake heavy
                        # gravity. Ours is honest, so thrust numbers
                        # below read like a real lander's.
MAIN_ACC = 4.9          # m/s² the main engine accelerates the whole
                        # lander ALONG ITS BODY UP-AXIS when at full
                        # burn. Thrust-to-weight = 4.9/1.62 ≈ 3.0:
                        # enough to hover AND climb, useless unless
                        # pointed sensibly. Fixed on purpose: like
                        # cart-pole, the agent picks a BUTTON, never
                        # a strength — discrete actions forever.
SIDE_ANG_ACC = 1.5      # rad/s² the side thrusters add to the spin.
                        # Torque only: they rotate the lander but do
                        # NOT move its center of mass. A real RCS
                        # plume would nudge translation too — we skip
                        # that coupling to keep the physics 3 lines.


# ==============================================================
# 2. THE ARENA AND THE LANDER'S BODY  (geometry)
# ==============================================================
WORLD_HALF_WIDTH = 50.0  # m — playfield spans x in [-50, +50]. Flying
                         # past either edge = gone (out_of_bounds).
CEILING_Y   = 80.0       # m — the spawn altitude band tops out at 70;
                         # pushing above the ceiling is out_of_bounds
                         # too (Gym's "left the screen" rule).
PAD_HALF    = 15.0       # m — the pad is 30 m wide, centered on x=0,
                         # marked by two flags in the window later.
                         # A leg touching the ground OFF the pad is a
                         # missed landing, not a near thing.

# The lander is a rigid body: every fixture below is a point in the
# BODY frame (origin = center of mass, y toward the nose, meters).
# World position of point (bx, by) at tilt theta:
#     (x + bx·cos θ − by·sin θ ,  y + bx·sin θ + by·cos θ)
HULL_CORNERS = (         # the crew cabin: a 2.0 m x 1.2 m box.
    (1.0, 0.6), (1.0, -0.6), (-1.0, 0.6), (-1.0, -0.6),
)   # hull corner below ground = the ship belly-flopped (crash).
LEG_JOINTS = (           # where the two legs meet the hull bottom —
    (1.0, -0.6), (-1.0, -0.6),
)   # drawing only (step 4); physics only reads the TIPS.
LEG_TIPS = (             # foot pads: 1 m of splay to either side and
    (1.6, -1.6), (-1.6, -1.6),
)   # 1 m below the hull. UPRIGHT STANCE: tips at y = COM_y - 1.6,
    # hull bottom at COM_y - 0.6. So when the tips kiss the ground
    # the hull still has 1.0 m of clearance — that gap IS the
    # recovery budget (how far you may sink after a touch).

# reset()'s scatter, as named constants: seeds of VARIETY. A fixed
# spawn would teach one memorized dive; these force a POLICY.
SPAWN_ALTITUDE = 70.0    # m — spawn height, fixed (12 m below the
                         # ceiling: a full-burn panic climb has room,
                         # but not for long).
SPAWN_X_RANGE = 30.0     # m — x scattered in [-30, +30]: every spawn
                         # still starts 70..76 m from the pad center.
SPAWN_VX_RANGE = 2.0     # m/s — sideways drift in [-2, +2]
SPAWN_VY_RANGE = 1.0     # m/s — sink/rise drift in [-1, +1]


# ==============================================================
# 3. WHAT THE AGENT CAN DO  (four buttons, discrete forever)
# ==============================================================
IDLE      = 0   # engines off — free fall / spin-down
MAIN      = 1   # main engine: +MAIN_ACC along the body up-axis
ROT_LEFT  = 2   # left thruster: +SIDE_ANG_ACC → nose leans to -x
ROT_RIGHT = 3   # right thruster: -SIDE_ANG_ACC → nose leans to +x

ACTION_NAMES = {IDLE: "IDLE", MAIN: "MAIN",
                ROT_LEFT: "LEFT", ROT_RIGHT: "RIGHT"}
N_ACTIONS = 4    # the MLP in step 2 gets exactly 4 output neurons.


# ==============================================================
# 4. THE RULES OF TOUCHDOWN  (termination thresholds)
# ==============================================================
# All the crash/land decisions compare against THREE speeds/angles.
# Edges: strict for dying (> limit), inclusive for landing (<=), the
# same convention 03 used for its 12° pole limit.
V_IMPACT       = 2.0   # m/s — touching ground FASTER than this is an
                       # instant hard-impact crash. Below it, a touch
                       # is survivable...
V_LAND         = 1.0   # m/s — ...but to be SCORED a landing, BOTH legs
                       # down with |vx| and |vy| at or under this.
                       # The band (1.0, 2.0] m/s is the RECOVERY BAND:
                       # sunk but not dead — burn the main engine and
                       # pull out (see section 8's demo).
ANGLE_LAND_DEG = 10.0  # deg — level-enough to count as landed.
ANGLE_LAND = math.radians(ANGLE_LAND_DEG)

LAND_REWARD = 100.0    # terminal bonus for a good touchdown.

# --- THE DEATH BILL, now GRADED (run 2's lesson) -------------------
# A crash is not one event: it is death AT A SPEED, AT AN ATTITUDE.
# Two official runs taught this the hard way:
#   RUN 1 (flat -100 bill): every episode ended in the bill, the
#     replay corpus never saw a positive terminal, and the Q surface
#     decayed into a smooth "danger map" with a flat action
#     dimension. Medicine: shrink the bill (03's lesson again).
#   RUN 2 (flat -30 bill): 3000 episodes, land rate STILL 0%, exam
#     returns all positive (+40..+58) — this time improved, not
#     fixed. The probe found the graduate arriving at a MEDIAN
#     speed of 14.7 m/s (free-fall from 70 m is ~15.1: it stopped
#     braking) and a MEDIAN wrapped tilt of 2.6 rad (869 deg of
#     spin) — because a flat bill prices a 15 m/s meteor EXACTLY
#     like a 1 m/s graze. No money anywhere on being gentle, and
#     rotation (a 0.005/tick cough vs IDLE's free) drifted to 59%
#     of the graduate's actions near the ground.
# The bill is now priced on the SAME quantities the touchdown rules
# already read:
#     bill = -(BASE + PER_SPEED * speed + PER_TILT * wrapped |tilt|)
# speed = hypot(vx, vy) (the impact rule's own number, so physics
# and price can never split); the tilt folds into [0, pi], so 869
# degrees of spin reads as "a bit past level", not "eight hundred
# degrees worse". Hand values: a level 1 m/s graze -7.5, the meteor
# arrival about -50 wherever it lands — a real gradient toward
# slow, level arrivals, paid ONCE at the fatal tick (never as
# per-tick nagging; 03's hammer-blow lesson). TIMEOUT still costs
# 0 — hovering forever pays no bonus either (section 5's trap).
# Every number here is pinned in test_environment.py.
CRASH_BASE      = 5.0    # flat fee for dying at all, any flavor
CRASH_PER_SPEED = 2.5    # + per m/s of COM speed at the fatal tick
CRASH_PER_TILT  = 4.0    # + per radian of wrapped |tilt| (0..pi)
# Sanity ceiling: the worst legal death (a full-tilt meteor) pays
# about -(5 + 2.5*16 + 4*pi) ~ -68 — always milder than run 1's
# flat -100, and never able to drown landing's +100 prize.

DEATH_REASONS = ("landed", "crashed", "out_of_bounds", "timeout")


def crash_bill(vx, vy, theta):
    """The graded terminal bill for ANY way of dying (section 4).

    Args: vx, vy — COM speeds at the fatal tick; hypot(vx, vy) is
                    the SAME number the impact rule reads, so the
                    physics and the price never split.
          theta  — the tilt at the fatal tick (raw and continuous:
                   a spinning lander wraps through mod-2pi HERE,
                   so step() can keep raw theta and the bill still
                   reads ATTITUDE, not lap count).

    Returns: NEGATIVE money (the step's terminal component). One
    function serves hull / impact / missed_pad AND out-of-bounds —
    the death flavors differ in the coroner's report, not in the
    physics that prices them.
    """
    turn = abs(math.fmod(theta, 2.0 * math.pi))   # fold into [0, 2pi)
    if turn > math.pi:                            # then into [0, pi]
        turn = 2.0 * math.pi - turn
    return -(CRASH_BASE
             + CRASH_PER_SPEED * math.hypot(vx, vy)
             + CRASH_PER_TILT * turn)


# ==============================================================
# 5. THE REWARD ANATOMY  (money, three currencies)
# ==============================================================
# reward(step) = terminal + shaping + fuel, all three reported in
# info["components"] so training logs can chart them separately.
#
# 1) TERMINAL — landed +100; every death pays the GRADED bill of
#    section 4 (run 2's lesson); timeout pays 0.
# 2) SHAPING — potential-based: the agent is paid BACK for the
#    distance it closes toward the pad, charged for opening it:
#
#        Φ(x, y) = -SHAPING_SCALE · dist(x, y) / SHAPING_NORM
#        shaping(step) = Φ(after) − Φ(before)        <- a DIFFERENCE
#
#    dist is center-of-mass distance to the pad center (0, 0). Why
#    differences (not a flat per-taxi-meter fee): they TELESCOPE.
#    Mid-way fees cancel out, so an episode's total shaping is
#    exactly Φ(far end) − Φ(start) — one number, hand-checkable in
#    test_shaping_telescopes. Spawn sits ~70–76 m out → Φ(start) is
#    about -105..-114; landed COM sits ~1.6 m up → Φ(land) ≈ -2.4.
#    So a good landing BANKS ≈ +103..+112 of shaping, and returns
#    (death bills graded, section 4) order out like:
#        land                    ≈ +200
#        hover the clock         ≈ +90   <- the escape hatch
#        gentle graze at the pad ≈ +90   (-8..-14 bill + full bank)
#        meteor arrival          ≈ +30..+55
#    The gradient says TWICE as much now: get down there (the
#    funnel) AND get down gently (the graded bill). Mediocrity
#    banks money; only humility climbs past it.
#
#    KNOWN TRAPS, updated by the official runs:
#    - run 1: the flat -100 bill drowned the funnel (section 4).
#    - run 2: the flat -30 bill made hover and gentle crashing the
#      two best mediocre lives but paid braking NOTHING, so the
#      policy stayed a meteor. The graded bill is that fix. If run
#      3 instead goes timeout-heavy (hover-the-clock wins too
#      often), the next knob is a small timeout bill — tuned with
#      data, not guessed.
#
# 3) FUEL — score cost per ENGINE-BURNING tick. No tank, no dry-out
#    failure: fuel is money, not oxygen (one less death mode to
#    test). Magnitudes are small on purpose: a whole-episode full
#    burn costs 20 — real pressure, never bigger than the terminal
#    bills (03's lesson: shaping via hammer blows breaks DQN
#    gradients).
SHAPING_SCALE = 150.0   # A — full money for closing 100 m.
SHAPING_NORM  = 100.0   # m — distance that counts as "the whole way".
FUEL_MAIN = 0.02        # per tick of main burn
FUEL_SIDE = 0.005       # per tick of either side thruster


# ==============================================================
# 6. THE STATE'S SCALES  (for the MLP in step 2, not for physics)
# ==============================================================
# Raw floats live on wildly different ranges; dividing by the
# "typical max" of each maps everything to roughly ±1 so no input
# dominates the gradients (same rule as 03's normalize_state).
# Legs scale by 1 — they are 0/1 already.
# Angle scale is π, but theta is NOT wrapped (a spinning lander can
# exceed ±1 the way a 20° death thump exceeded 1 in 03 — the net is
# allowed to see "far outside"; clipping would hide the spin).
STATE_SCALE = (
    WORLD_HALF_WIDTH,   # x        ±50 m  (reach a world edge)
    CEILING_Y,          # y        0..80 m (ground to ceiling)
    6.0,                # vx       ±6 m/s (spawn ±2, crashes land ~3)
    8.0,                # vy       ±8 m/s (terminal fall speed ~15)
    math.pi,            # theta    a half-flip counts as ±1
    4.0,                # omega    ±4 rad/s (4 full burns of spin)
    1.0, 1.0,           # legL/legR binary — already normalized
)


def normalize_state(state):
    """Scale 8 raw env floats into MLP-friendly ~[-1, 1] inputs.

    Args: state — the 8-tuple from reset()/step().
    Returns: list of 8 floats (NOT a tensor — keeping it plain makes
             it testable with no torch in the loop).
    """
    return [float(v) / s for v, s in zip(state, STATE_SCALE)]


# ==============================================================
# 7. THE ENVIRONMENT CLASS — the world in code
# ==============================================================
class LunarLander:
    """A lander falling through the Moon's sky toward a flat pad:

        state                     = reset()
        state, reward, done, info = step(action)

    The DQN brain in step 2 feeds `state` (8 raw floats) straight
    into its MLP. `info` is the humans'/charts' side channel — the
    agent never learns from it.
    """

    def __init__(self, seed=None, max_steps=MAX_STEPS):
        """Set up the unchanging rules; reset() starts episodes.

        Args: seed      — int for reproducible spawns (tests pin the
                          first step with one; None = unpredictable).
              max_steps — episode cap; tests shrink it to hit the
                          timeout quickly.

        REPORTR NOTE (independence): the RNG is a PRIVATE
        random.Random(self) instance, NOT the global random module —
        seeding this world seeds nothing else, and nothing else can
        hijack this world's spawns. 01-03 used global seeds; this
        project is self-contained on purpose.
        """
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        # Dynamic state — every episode overwrites them (see reset).
        self.x = None          # COM position, pad-centered   (m)
        self.y = None          # COM height above ground       (m)
        self.vx = None         # horizontal speed              (m/s)
        self.vy = None         # vertical speed (+ up)         (m/s)
        self.theta = None      # tilt, 0 = upright, + = lean -x (rad)
        self.omega = None      # spin rate (+ = CCW)           (rad/s)
        self.leg_l = 0         # 1 while the LEFT tip is in the ground
        self.leg_r = 0         # 1 while the RIGHT tip is in the ground
        self.steps = 0         # ticks taken this episode
        self.done = False      # episode over? (guards step())
        self.fuel_used = 0.0   # cumulative burn, positive money (HUD)
        # The banked-shaping ledger: shaping is a DIFFERENCE of the
        # potential, so step() compares each new Φ with this.
        self.prev_potential = None

    @property
    def state(self):
        """The agent's view of the world: exactly eight floats, in
        the fixed order documented at the top of this file. Tests
        pin the order (index 6 must be the LEFT leg)."""
        return (self.x, self.y, self.vx, self.vy,
                self.theta, self.omega, self.leg_l, self.leg_r)

    # ----------------------------------------------------------
    # 7a. potential() — the shaping ledger's per-state worth
    # ----------------------------------------------------------
    def potential(self, x, y):
        """Shaping potential Φ = −A · (distance to pad) / norm.

        Distance is COM-to-pad-center. Pulling A/norm together:
        Φ = −1.5 · dist(m). Hand values: spawn ≈ −105..−114,
        landed (COM ~1.6 m up) ≈ −2.4, so landing banks ≈ +103..112.
        Public on purpose — the telescoping test and step 5's
        reward-decomposition chart both recompute it.
        """
        return -SHAPING_SCALE * math.hypot(x, y) / SHAPING_NORM

    # ----------------------------------------------------------
    # 7b. reset() — a fresh attempt at the sky
    # ----------------------------------------------------------
    def reset(self):
        """Spawn high over the pad, slow and level; return the state.

        WHY random: a fixed spawn would teach one memorized dive.
        Scattering x/vx/vy (levels fixed) forces a POLICY — like 02's
        START_NOISE, but with real stakes: spawn distance is 70-76 m,
        so every episode starts with ~100 points of unbanked
        potential it can earn or throw away.

        Returns: the 8-float start state (eyes on the pad instantly —
        including flags 0/0, since ground contact is impossible yet).
        """
        self.x = self.rng.uniform(-SPAWN_X_RANGE, SPAWN_X_RANGE)  # m
        self.y = SPAWN_ALTITUDE          # fixed altitude
        self.vx = self.rng.uniform(-SPAWN_VX_RANGE, SPAWN_VX_RANGE)
        self.vy = self.rng.uniform(-SPAWN_VY_RANGE, SPAWN_VY_RANGE)
        self.theta = 0.0                 # upright — drifting, not tipped
        self.omega = 0.0                 # not spinning
        self.leg_l = 0                   # spawn is 68 m above the tips'
        self.leg_r = 0                   # never-touch ground line
        self.steps = 0
        self.done = False
        self.fuel_used = 0.0
        self.prev_potential = self.potential(self.x, self.y)
        return self.state

    # ----------------------------------------------------------
    # 7c. step(action) — the heart of the world
    # ----------------------------------------------------------
    def step(self, action):
        """Apply one engine command and let gravity answer.

        Args:    action — IDLE / MAIN / ROT_LEFT / ROT_RIGHT (0..3)
        Returns: (state, reward, done, info)
            state  — 8 floats AFTER this tick (flags included)
            reward — terminal + shaping + fuel for this tick (the
                     terminal part lands on the tick that ends it)
            done   — episode over? (landed / crashed / OOB / timeout)
            info   — {reason, crash_type, steps, fuel_used,
                      components: {terminal, shaping, fuel}} — for
                      logs and charts; the brain ignores it.
        """
        # --- Guard rails (same culture as 03: fail loudly, early) ---
        if self.done:
            raise RuntimeError("Episode is over — call reset().")
        if action not in (IDLE, MAIN, ROT_LEFT, ROT_RIGHT):
            raise ValueError(
                f"Invalid action {action!r}: use 0=IDLE, 1=MAIN, "
                "2=ROT_LEFT, 3=ROT_RIGHT")

        # --- 1. THE PHYSICS — accelerations, then semi-implicit Euler
        # Speeds update FIRST from this tick's accelerations, then
        # positions move using the FRESH speeds (same 02/03 scheme:
        # plain Euler pumps energy into wobbles; this one stays sane).
        ax = 0.0                      # gravity alone...
        ay = -GRAVITY                 # ...points straight down.
        ang_acc = 0.0
        if action == MAIN:
            # Thrust along the BODY UP-AXIS (-sinθ, cosθ): tilt and
            # the burn steers you sideways — θ=0 is a pure vertical
            # rocket; θ=90° is a horizontal one (and a comedy).
            ax += -MAIN_ACC * math.sin(self.theta)
            ay += MAIN_ACC * math.cos(self.theta)
        elif action == ROT_LEFT:
            ang_acc = SIDE_ANG_ACC    # nose toward -x
        elif action == ROT_RIGHT:
            ang_acc = -SIDE_ANG_ACC   # nose toward +x

        self.vx += ax * DT            # speeds first...
        self.vy += ay * DT
        self.omega += ang_acc * DT
        self.x += self.vx * DT        # ...positions after, with the
        self.y += self.vy * DT        # JUST-UPDATED speeds.
        self.theta += self.omega * DT
        self.steps += 1

        # --- 2. WHO IS TOUCHING WHAT  (detection, not a solver) -----
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        # Body point (bx, by) -> world: rotate by θ, then translate.
        def to_world(bx, by):
            return (self.x + bx * cos_t - by * sin_t,
                    self.y + bx * sin_t + by * cos_t)

        # The ONE graded death bill for this tick (run 2's lesson),
        # computed HERE so every branch below prices death with the
        # same speeds/attitude the physics just produced — no branch
        # can forget the grading.
        bill = crash_bill(self.vx, self.vy, self.theta)

        terminal = 0.0
        reason = None
        crash_type = None             # fine-grained crash flavor for
                                      # the step-5 death chart
        self.leg_l = 0
        self.leg_r = 0

        # 2a. Left the world? (checked first — nothing else matters.
        #     Edges are STRICT: exactly ON the boundary is still in.)
        if (self.x < -WORLD_HALF_WIDTH or self.x > WORLD_HALF_WIDTH
                or self.y > CEILING_Y):
            terminal, reason = bill, "out_of_bounds"

        # 2b. Belly on the ground? Any hull corner below 0 = the ship's
        #     BODY hit dirt — game over no matter how gently.
        elif any(to_world(bx, by)[1] < 0.0 for bx, by in HULL_CORNERS):
            terminal, reason = bill, "crashed"
            crash_type = "hull"

        # 2c. Feet on the ground: each tip poking below 0 is contact.
        #     Off-pad contact and hard impacts kill; slow on-pad ticks
        #     register as leg flags (or leave the recovery band open).
        else:
            speed = math.hypot(self.vx, self.vy)   # COM speed (a docu-
                          # mented simplification: the tip's own
                          # rotational speed is not counted)
            missed = hard = False
            for which, tip in (("l", LEG_TIPS[1]), ("r", LEG_TIPS[0])):
                tx, ty = to_world(tip[0], tip[1])
                if ty < 0.0:
                    if abs(tx) > PAD_HALF:
                        missed = True            # one foot off = fail
                    elif speed > V_IMPACT:
                        hard = True              # too fast to survive
                    else:
                        setattr(self, f"leg_{which}", 1)  # soft tick
            if missed or hard:
                terminal, reason = bill, "crashed"
                crash_type = "missed_pad" if missed else "impact"
            # 2d. BOTH feet ticked, slow and level? That is a landing —
            #     scored THIS tick, legs possibly a bit buried.
            elif (self.leg_l and self.leg_r
                    and abs(self.vx) <= V_LAND and abs(self.vy) <= V_LAND
                    and abs(self.theta) <= ANGLE_LAND):
                terminal, reason = LAND_REWARD, "landed"
            # (single-foot or recovery-band ticks: keep falling — the
            #  agent may arrest and land, or sink into a hull check)

        # 2e. Still flying after 1000 ticks? The clock ends it, but
        #     remembers nothing happened (no bonus, no bill).
        if reason is None and self.steps >= self.max_steps:
            reason = "timeout"

        # --- 3. THE MONEY — terminal + shaping + fuel ---------------
        # Shaping: the potential's CHANGE this tick (telescoping by
        # construction — see section 5 and the shaping tests).
        new_potential = self.potential(self.x, self.y)
        shaping = new_potential - self.prev_potential
        self.prev_potential = new_potential

        fuel = 0.0
        if action == MAIN:
            fuel = -FUEL_MAIN
        elif action in (ROT_LEFT, ROT_RIGHT):
            fuel = -FUEL_SIDE
        self.fuel_used += -fuel       # HUD keeps the BURN (positive)

        reward = terminal + shaping + fuel
        self.done = reason is not None

        info = {
            "reason": reason,         # None while alive
            "crash_type": crash_type, # hull / impact / missed_pad
            "steps": self.steps,
            "fuel_used": self.fuel_used,
            "components": {"terminal": terminal,
                           "shaping": shaping, "fuel": fuel},
        }
        return self.state, reward, self.done, info

    # ----------------------------------------------------------
    # 7d. geometry() — the body's points in world space (draw me)
    # ----------------------------------------------------------
    def geometry(self):
        """Every drawn fixture in WORLD coordinates (meters, y-up).

        Step 4's pygame window maps these to pixels SOLELY at the
        draw boundary, so the world keeps math space and the window
        keeps one flip. Returns:
            hull       — 4 corners (the cabin box outline)
            leg_joints — 2 points (leg roots on the hull bottom)
            leg_tips   — 2 points (foot pads)
        """
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        def to_world(bx, by):
            return (self.x + bx * cos_t - by * sin_t,
                    self.y + bx * sin_t + by * cos_t)

        return {
            "hull": [to_world(bx, by) for bx, by in HULL_CORNERS],
            "leg_joints": [to_world(bx, by) for bx, by in LEG_JOINTS],
            "leg_tips": [to_world(bx, by) for bx, by in LEG_TIPS],
        }

    # ----------------------------------------------------------
    # 7e. render() — the world as ASCII, no window needed
    # ----------------------------------------------------------
    def render(self):
        """Print a side view: lander, ground, pad between two flags.

        The debugger's cheap X-ray of the pygame window (step 4).
        ASCII only — '.' taught 02 to stay cp1252-portable.
        """
        cols, sky_rows = 51, 9   # 51 chars wide; 9 sky rows + ground

        lines = [
            f"step {self.steps}/{self.max_steps}  "
            f"x={self.x:+.2f}m  y={self.y:.2f}m  "
            f"vx={self.vx:+.2f}  vy={self.vy:+.2f}  "
            f"tilt={math.degrees(self.theta):+.1f}deg  "
            f"legs=[{'L' if self.leg_l else '.'}"
            f"{'R' if self.leg_r else '.'}]"
        ]

        # The lander as a 'Y': mast plus two legs, one char. Column
        # from x (-50..50 -> 0..50), row from y (0..80 -> 0..8, 0 is
        # just above the ground row).
        col = round((self.x + WORLD_HALF_WIDTH)
                    / (2 * WORLD_HALF_WIDTH) * (cols - 1))
        col = max(0, min(cols - 1, col))
        row = round(min(max(self.y, 0.0), CEILING_Y)
                    / CEILING_Y * (sky_rows - 1))
        row = max(0, min(sky_rows - 1, row))

        for r in range(sky_rows - 1, -1, -1):   # top row first
            sky = [" "] * cols
            if r == row:
                sky[col] = "Y"                  # the lander itself
            lines.append("".join(sky))

        # Ground: '-' everywhere, '=' pad, '|' the two flags.
        pad = []
        for c in range(cols):
            wx = c / (cols - 1) * 100.0 - WORLD_HALF_WIDTH
            if abs(wx) <= PAD_HALF:
                pad.append("=")                 # between the flags
            elif abs(abs(wx) - PAD_HALF) < 100.0 / (cols - 1):
                pad.append("|")                 # a flagpole cell
            else:
                pad.append("-")
        lines.append("".join(pad))

        print("\n".join(lines))


# ==============================================================
# 8. DEMO — the rules of touchdown, fired by hand
# ==============================================================
# A scripted tour, not a game loop: place the world in four poses and
# watch each rule answer. The demos may set fields directly — the
# same trick test_environment.py uses, advertised so nobody thinks
# it is magic.
if __name__ == "__main__":
    print("=== the numbers of this world ===")
    print(f"gravity {GRAVITY} m/s^2 | main engine {MAIN_ACC} m/s^2 "
          f"(thrust-to-weight {MAIN_ACC / GRAVITY:.2f}) "
          f"| side thrust {SIDE_ANG_ACC} rad/s^2 | tick {DT} s")
    print(f"pad: {2 * PAD_HALF:.0f} m wide | world: "
          f"{2 * WORLD_HALF_WIDTH:.0f} x {CEILING_Y:.0f} m "
          f"| episode cap: {MAX_STEPS} ticks = {MAX_STEPS * DT:.0f} s")
    print(f"speed rules: land <= {V_LAND}, touch survives <= "
          f"{V_IMPACT}, level <= {ANGLE_LAND_DEG} deg | "
          f"bills: land {+LAND_REWARD:+.0f}; crash = -({CRASH_BASE} "
          f"+ {CRASH_PER_SPEED}*speed + {CRASH_PER_TILT}*tilt)")

    # --- pose A: plain free fall (three ticks) -----------------------
    env = LunarLander(seed=0)
    env.reset()
    env.x, env.y, env.vx, env.vy = 0.0, 70.0, 0.0, 0.0
    env.prev_potential = env.potential(env.x, env.y)
    print("\n=== A: engines off, gravity speaks ===")
    env.render()
    for _ in range(3):
        env.step(IDLE)
        env.render()
    print(f"hand check: vy must lose {GRAVITY * DT:.4f} m/s per tick")

    # --- pose B: the RECOVERY BAND — touch too hard-ish, then save it
    env.done = False
    env.x, env.y, env.vx, env.vy, env.theta, env.omega = 0.0, 1.6, 0.0, -1.5, 0.0, 0.0
    env.prev_potential = env.potential(env.x, env.y)
    print("\n=== B: touch at 1.5 m/s (too fast to score, too slow to die) ===")
    state, reward, done, info = env.step(IDLE)
    print(f"touch tick: legs={state[6]}{state[7]}, vy={state[3]:+.3f}, "
          f"still flying: {not done}  <- the band is open")
    while not done:
        state, reward, done, info = env.step(MAIN)
    print(f"burned out of the dive and landed: reason={info['reason']}, "
          f"terminal={info['components']['terminal']:+.0f}, "
          f"COM at y={env.y:.2f} (legs buried a bit)")

    # --- pose C: the three ways to die on this Moon -------------------
    for name, setup, why in (
        ("C1 hard impact", dict(y=1.6, vy=-3.0),
         f"speed > {V_IMPACT} at the tick the legs touch"),
        ("C2 missed the pad", dict(x=16.0, y=1.6, vy=-0.5),
         "a leg touches ground with |x| beyond the flags"),
        ("C3 belly flop", dict(y=0.55), "a hull corner below the ground"),
    ):
        env.done = False
        env.x, env.y = setup.get("x", 0.0), setup["y"]
        env.vx, env.vy = 0.0, setup.get("vy", 0.0)
        env.theta, env.omega = 0.0, 0.0
        env.leg_l = env.leg_r = 0
        env.prev_potential = env.potential(env.x, env.y)
        state, reward, done, info = env.step(IDLE)
        print(f"\n=== {name}: {why} ===")
        print(f"reason={info['reason']!r} ({info['crash_type']}), "
              f"terminal bill {info['components']['terminal']:+.1f}, "
              f"reward {reward:+.3f}")
        env.render()

    print("\nstep 2 (the DQN brain) must learn to fly pose A into pose B"
          "\nand never pose C. Every number above is pinned in"
          "\ntest_environment.py - run it after ANY change here.")
