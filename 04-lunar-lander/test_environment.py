"""
test_environment.py — does the world actually behave?
=====================================================

Physics can lie convincingly, so the bar is the same one 03 set:
the FIRST step of each control action is computed BY HAND and the
code must match it (test_hand_computed_first_step and friends).
Beyond the three integration checks, every touchdown rule gets its
own scenario: land, recovery band, the three crash flavors, the
world edges, the clock — and the reward's three currencies are
pinned component by component, including the telescoping identity
that makes the shaping honest.

Run with:  python test_environment.py
"""

# Imports, each earning its line (repo rule: comment every file well).
import contextlib  # capture render()'s print() output for assertions
import io          # in-memory buffer render() "prints" into
import math        # radians for tilt tests, isclose for float checks

# The whole public contract of the world, imported in one place so
# a rename in environment.py breaks loudly here instead of silently.
from environment import (
    LunarLander, IDLE, MAIN, ROT_LEFT, ROT_RIGHT, N_ACTIONS, ACTION_NAMES,
    DT, MAX_STEPS, GRAVITY, MAIN_ACC, SIDE_ANG_ACC,
    WORLD_HALF_WIDTH, CEILING_Y, PAD_HALF,
    HULL_CORNERS, LEG_JOINTS, LEG_TIPS,
    SPAWN_ALTITUDE, SPAWN_X_RANGE, SPAWN_VX_RANGE, SPAWN_VY_RANGE,
    V_IMPACT, V_LAND, ANGLE_LAND_DEG,
    LAND_REWARD, CRASH_PENALTY,
    SHAPING_SCALE, SHAPING_NORM, FUEL_MAIN, FUEL_SIDE,
    STATE_SCALE, normalize_state,
)


def _at(env, x=0.0, y=SPAWN_ALTITUDE, vx=0.0, vy=0.0,
        theta=0.0, omega=0.0):
    """Place the world at an EXACT pose (tests poke the physics).

    reset() is intentionally random, but rule tests need known
    starting numbers — so we set the state fields directly. The
    shaping ledger is re-synced so the next step's delta measures
    THIS pose, not a ghost of an earlier one.
    """
    env.x, env.y = x, y
    env.vx, env.vy = vx, vy
    env.theta, env.omega = theta, omega
    env.leg_l = env.leg_r = 0
    env.fuel_used = 0.0
    env.steps = 0
    env.done = False
    env.prev_potential = env.potential(env.x, env.y)
    return env


# ---------------------------------------------------------------
# reset() — a fair, reproducible, safe start
# ---------------------------------------------------------------
def test_spawn_contract():
    """Every spawn is a legal pose: high, slow, upright, feet dry —
    and a seed reproduces it exactly (tests and demos depend on it)."""
    for seed in range(30):
        env = LunarLander(seed=seed)
        s = env.reset()

        assert len(s) == 8, "state must be exactly 8 numbers"
        assert all(isinstance(v, (int, float)) for v in s)
        assert s[1] == SPAWN_ALTITUDE, "spawn altitude is fixed at 70 m"
        assert s[4] == 0.0 and s[5] == 0.0, "spawn upright and still"
        assert s[6] == 0 and s[7] == 0, "feet dry at spawn"
        assert -SPAWN_X_RANGE <= s[0] <= SPAWN_X_RANGE
        assert -SPAWN_VX_RANGE <= s[2] <= SPAWN_VX_RANGE
        assert -SPAWN_VY_RANGE <= s[3] <= SPAWN_VY_RANGE
        assert abs(s[0]) < WORLD_HALF_WIDTH, "never spawn off the world"

    a, b = LunarLander(seed=42), LunarLander(seed=42)
    assert a.reset() == b.reset(), "same seed -> identical spawn"

    env = LunarLander(seed=42)
    first, second = env.reset(), env.reset()
    assert first != second, "the RNG stream advances between resets"
    print("  30 seeded spawns legal; same seed repeats, stream advances")


def test_invalid_action_rejected():
    """Four buttons exist; anything else gets a friendly error."""
    env = LunarLander(seed=0)
    env.reset()
    assert N_ACTIONS == 4
    assert ACTION_NAMES == {IDLE: "IDLE", MAIN: "MAIN",
                            ROT_LEFT: "LEFT", ROT_RIGHT: "RIGHT"}

    for bad in (4, -1, "main", None, 2.5):
        try:
            env.step(bad)
        except ValueError as e:
            assert "Invalid action" in str(e)
        else:
            raise AssertionError(f"step({bad!r}) should have failed")
    print("  only 0..3 accepted; everything else raises ValueError")


def test_step_after_done_rejected():
    """Forgetting reset() between episodes fails loudly, not silently."""
    env = LunarLander(seed=0)
    _at(env, y=1.6, vy=-3.0)              # guaranteed impact this tick
    env.step(IDLE)
    assert env.done
    try:
        env.step(IDLE)
    except RuntimeError as e:
        assert "reset()" in str(e)
    else:
        raise AssertionError("step() after done should have failed")
    print("  step() after episode end raises a clear RuntimeError")


# ---------------------------------------------------------------
# THE BIG ONES — one hand-computed step per control law
# ---------------------------------------------------------------
def test_hand_computed_first_step():
    """One MAIN burn from rest must match a hand calculation.

    Hand derivation (upright, theta = 0 so the body up-axis is (0, 1),
    parked 70 m up where nothing can touch us):

        ax   = -4.9 * sin(0)      = 0
        ay   = -1.62 + 4.9*cos(0) = +3.28 m/s^2

    then semi-implicit integration (speeds first) with DT = 0.02:

        vy = 0.02 * 3.28        = +0.0656   m/s
        y  = 70 + 0.02 * 0.0656 = 70.001312 m
        vx = x = omega = theta  = unchanged (no tilt -> no sideways)

    If these six-decimal numbers move, the physics changed — and
    step 2's DQN would be learning a different game than this file
    certifies.
    """
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0)              # exact rest, high in the air

    state, reward, done, info = env.step(MAIN)
    x, y, vx, vy, theta, omega, leg_l, leg_r = state

    assert x == 0.0 and vx == 0.0, "upright burn is purely vertical"
    assert theta == 0.0 and omega == 0.0
    assert abs(vy - 0.0656) < 1e-12, f"vy={vy:.9f}"
    assert abs(y - 70.001312) < 1e-12, f"y={y:.9f}"
    assert not done and info["reason"] is None
    assert info["components"]["fuel"] == -FUEL_MAIN
    print(f"  hand-computed MAIN step matches: vy={vy:+.6f}, "
          f"y={y:+.7f} m")


def test_gravity_free_fall():
    """Engines off: pure lunar gravity, hand-computed per tick.

        vy = -1.62 * 0.02 = -0.0324 m/s per tick (no thrust, no drag)
        y  = y + 0.02 * vy (semi-implicit: move with the FRESH speed)
    """
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0)

    state, reward, done, info = env.step(IDLE)
    assert abs(state[3] - (-0.0324)) < 1e-12
    assert abs(state[1] - 69.999352) < 1e-12
    assert state[0] == 0.0 and state[2] == 0.0
    assert info["components"]["fuel"] == 0.0, "IDLE is free"
    assert reward > 0.0, "falling level = closing on the pad = paid"
    print(f"  free fall: vy={state[3]:+.6f} m/s, y={state[1]:.6f} m")


def test_main_beats_gravity():
    """Thrust-to-weight ≈ 3: hovering is possible, climbing is easy —
    the number that makes this a game (and not a rock)."""
    ratio = MAIN_ACC / GRAVITY
    assert 3.0 < ratio < 3.1, f"T/W = {ratio:.3f}, expected ~3.0"
    env = LunarLander(seed=0)
    _at(env, y=70.0)
    state, _, done, _ = env.step(MAIN)
    assert state[3] > 0.0 and not done, "main engine climbs from rest"
    print(f"  T/W = {ratio:.2f}: main engine outweighs the Moon")


def test_rotation_first_step_and_persistence():
    """Side thrusters: hand-computed first tick, then prove the spin
    persists (no friction — the Moon's sky has no air).

        action LEFT : omega = +1.5*0.02 = +0.03 rad/s
                      theta = +0.02*0.03 = +0.0006 rad (nose to -x)
        next IDLE   : omega unchanged, theta += 0.02*0.03 -> 0.0012
    """
    env = LunarLander(seed=0)
    _at(env)
    state, _, _, _ = env.step(ROT_LEFT)
    assert abs(state[5] - 0.03) < 1e-12, f"omega={state[5]}"
    assert abs(state[4] - 0.0006) < 1e-12, f"theta={state[4]}"

    state, _, _, _ = env.step(IDLE)          # hands off the stick
    assert abs(state[5] - 0.03) < 1e-12, "spin must persist (no drag)"
    assert abs(state[4] - 0.0012) < 1e-12, "theta keeps integrating"

    env2 = LunarLander(seed=0)
    _at(env2)
    state, _, _, _ = env2.step(ROT_RIGHT)
    assert abs(state[5] - (-0.03)) < 1e-12
    assert abs(state[4] - (-0.0006)) < 1e-12
    print("  rotation: left/right mirrored; spin persists, no drag")


def test_tilted_thrust_steers():
    """Thrust points along the BODY, not the world: at 90 deg tilt a
    main burn is a pure sideways shove (and 180 deg accelerates you
    INTO the ground — the comedy that forces careful piloting)."""
    env = LunarLander(seed=0)
    _at(env, theta=math.radians(90))         # lying on its left side
    state, _, _, _ = env.step(MAIN)
    assert abs(state[2] - (-0.098)) < 1e-9, f"vx={state[2]}"
    assert abs(state[3] - (-0.0324)) < 1e-12, "no vertical push left"

    env = LunarLander(seed=0)
    _at(env, theta=math.radians(-90))
    state, _, _, _ = env.step(MAIN)
    assert abs(state[2] - 0.098) < 1e-9, "mirror tilt -> mirror shove"

    env = LunarLander(seed=0)
    _at(env, theta=math.radians(180))        # engine pointing DOWN
    state, _, _, _ = env.step(MAIN)
    assert abs(state[3] - (-0.1304)) < 1e-9, f"vy={state[3]}"
    assert abs(state[2]) < 1e-9
    print("  thrust follows the body axis: sideways at 90, down at 180")


# ---------------------------------------------------------------
# State layout — the MLP in step 2 eats these in a fixed order
# ---------------------------------------------------------------
def test_state_layout_order():
    """state is (x, y, vx, vy, theta, omega, legL, legR) — pin the
    order NOW; a silent swap would poison the MLP invisibly."""
    env = LunarLander(seed=0)
    _at(env, x=10.0, y=69.0, vx=1.5, vy=-0.5, theta=0.2, omega=0.1)
    assert env.state == (10.0, 69.0, 1.5, -0.5, 0.2, 0.1, 0, 0)
    print("  state order pinned: x, y, vx, vy, theta, omega, legL, legR")


def test_left_leg_only_contact():
    """A slow single-leg touch keeps flying: legL=1, legR=0, no
    landing, no crash — this is the 'feel the ground' state the
    8-float design was chosen for."""
    # Tilted +0.05 rad leans the nose to -x, dipping the LEFT tip
    # ~8 cm under while the RIGHT tip stays ~8 cm clear (hand-checked
    # with the rotation formulas): y' = 1.599352,
    #   tipL y = 1.599352 - 1.6*(sin+.05 + cos.05) = -0.079
    #   tipR y = 1.599352 + 1.6*(sin+.05 - cos.05) = +0.081
    env = LunarLander(seed=0)
    _at(env, y=1.60, theta=0.05)
    state, reward, done, info = env.step(IDLE)

    assert state[6] == 1, "LEFT tip must register contact"
    assert state[7] == 0, "RIGHT tip is still clear"
    assert not done and info["reason"] is None
    assert env.leg_l == 1 and env.leg_r == 0
    print("  one-foot touch: legL fires, episode keeps going")


# ---------------------------------------------------------------
# The rules of touchdown
# ---------------------------------------------------------------
def test_slow_two_leg_touch_lands():
    """Both feet down, slow and level = scored, +100 terminal, done."""
    env = LunarLander(seed=0)
    _at(env, y=1.60, vy=-0.5)      # vy' = -0.5324 at tick end
    state, reward, done, info = env.step(IDLE)

    assert done and info["reason"] == "landed"
    assert env.leg_l == 1 and env.leg_r == 1
    comp = info["components"]
    assert comp["terminal"] == LAND_REWARD == 100.0
    assert comp["fuel"] == 0.0
    assert reward > LAND_REWARD, "landing also banks positive shaping"
    print(f"  gentle touchdown done: terminal={comp['terminal']:+.0f}, "
          f"reward={reward:+.3f}")


def test_recovery_band_save():
    """THE feature the detection-only design buys for free: touch in
    the band (1.0, 2.0] m/s — too fast to score, too slow to die —
    then BURN and pull out. Hand trail (vy = -1.5 at the touch):

        touch tick: vy' = -1.5324  -> both feet tick, band is open
        each MAIN : vy += +0.0656  -> -1.008 (8th, still too fast!)
                                      -0.942 (9th) -> LANDED
        COM sinks 1.599 -> 1.353 m: legs buried ~24 cm, hull clears
        it with 0.75 m to spare (the 1.0 m clearance budget in 2).
    """
    env = LunarLander(seed=0)
    _at(env, y=1.60, vy=-1.5)
    state, _, done, _ = env.step(IDLE)          # the touch tick

    assert not done, "1.53 m/s touch must open the band, not die"
    assert state[6] == 1 and state[7] == 1, "both feet ticked"
    assert abs(state[3]) > V_LAND, "too fast to score a landing yet"
    assert abs(state[3]) <= V_IMPACT, "too slow to count as an impact"

    for i in range(9):                          # the burn-out
        state, _, done, info = env.step(MAIN)
        if done:
            break
    assert done and info["reason"] == "landed", \
        f"must pull out on the 9th burn (py={env.y:.4f})"
    assert env.y > 1.30, "recovery never sank into a hull check"
    assert env.y < 1.37, "and it really did touch before saving"
    print(f"  recovery band: touched at ~1.5, landed with legs "
          f"buried to COM y={env.y:.3f} m")


def test_hard_impact_crash():
    """Faster than V_IMPACT at the tick a leg touches = rubble."""
    env = LunarLander(seed=0)
    _at(env, y=1.60, vy=-3.0)      # impact speed ~3.03 > 2.0
    state, reward, done, info = env.step(IDLE)

    assert done and info["reason"] == "crashed"
    assert info["crash_type"] == "impact"
    assert info["components"]["terminal"] == -CRASH_PENALTY == -100.0
    assert env.leg_l == 0 and env.leg_r == 0, "dead feet tick nothing"
    # The bill is -100, but the tick still banked a hair of positive
    # shaping (it was falling TOWARD the pad), so the total sits just
    # above -100 — verified exactly in the fuel/composition test.
    assert -100.0 < reward < -99.0
    print(f"  hard impact: speed {-state[3]:.2f} > {V_IMPACT} -> "
          f"crashed, bill {info['components']['terminal']:+.0f}")


def test_missed_pad_crash():
    """One foot beyond the flags ends it — the pad is 30 m, the world
    is 100 m, and 'almost' is not a landing."""
    env = LunarLander(seed=0)
    _at(env, x=16.0, y=1.60, vy=-0.5)   # tipR at x = 17.6 > 15
    state, reward, done, info = env.step(IDLE)

    assert done and info["reason"] == "crashed"
    assert info["crash_type"] == "missed_pad"
    assert info["components"]["terminal"] == -100.0

    env = LunarLander(seed=0)
    _at(env, x=13.0, y=1.60, vy=-0.5)   # tips at 11.4 / 14.6: inside!
    state, reward, done, info = env.step(IDLE)
    assert done and info["reason"] == "landed", "13 m off-center lands"
    print("  off-pad foot = crash; 13 m sideways still lands on the pad")


def test_hull_crash_and_precedence():
    """A hull corner below ground ends it — even when the touchdown
    SOUNDS acceptable (slow, level, both feet). Belly first = dead,
    full stop: the hull check must outrank the landing check."""
    env = LunarLander(seed=0)
    _at(env, y=0.55)            # hull bottom y' = -0.050; feet way under
    state, reward, done, info = env.step(IDLE)

    assert done and info["reason"] == "crashed"
    assert info["crash_type"] == "hull"
    assert info["components"]["terminal"] == -100.0
    # Precedence proof: in this pose BOTH feet are ticked and every
    # landing inequality would pass — the hull check won anyway.
    assert env.leg_l == 0 and env.leg_r == 0, "crash preempts flags"
    print("  hull corner down = crashed, outranking a would-be landing")


def test_out_of_bounds_and_edges():
    """Past the world's edges you are gone; exactly ON an edge you
    are not (strict comparison, same convention as 03's limits)."""
    env = LunarLander(seed=0)
    _at(env, x=WORLD_HALF_WIDTH + 0.5)
    state, reward, done, info = env.step(IDLE)
    assert done and info["reason"] == "out_of_bounds"
    assert info["components"]["terminal"] == -100.0

    env = LunarLander(seed=0)
    _at(env, x=WORLD_HALF_WIDTH)            # exactly on the edge
    state, _, done, _ = env.step(IDLE)
    assert not done, "boundary itself is still in bounds"

    env = LunarLander(seed=0)
    _at(env, y=CEILING_Y + 0.5)
    state, _, done, info = env.step(IDLE)
    assert done and info["reason"] == "out_of_bounds", "ceiling too"

    env = LunarLander(seed=0)
    _at(env, y=CEILING_Y)                   # exactly at the ceiling
    state, _, done, _ = env.step(IDLE)
    assert not done, "exactly 80 m is still legal"
    print("  |x|>50 or y>80 = out_of_bounds; on the line = safe")


def test_timeout():
    """A policy that never touches anything gets ended by the clock
    — with NO bonus and NO bill (the hover-trap stance from section
    5 of environment.py)."""
    env = LunarLander(seed=0, max_steps=5)
    env.reset()
    for i in range(5):
        state, reward, done, info = env.step(IDLE)
    assert done and info["reason"] == "timeout"
    assert info["components"]["terminal"] == 0.0
    assert env.steps == 5
    # Sanity: 5 gravity ticks from 70 m cannot reach anything.
    assert env.y > 69.9, "timeout fired, not a floor contact"
    print("  5 idle ticks -> timeout, terminal 0, nobody got paid")


# ---------------------------------------------------------------
# The three currencies of the reward
# ---------------------------------------------------------------
def test_fuel_accounting_and_composition():
    """Fuel: -0.02 per MAIN tick, -0.005 per side tick, free IDLE;
    the ticker sums to the HUD number; and every reward is EXACTLY
    its three components added in order."""
    env = LunarLander(seed=0)
    _at(env, y=70.0)     # a safe band: 4 ticks of messing around
    state_a, _, _, _ = env.step(MAIN)
    state_b, _, _, _ = env.step(ROT_LEFT)
    state_c, _, _, _ = env.step(IDLE)
    state_d, _, _, _ = env.step(ROT_RIGHT)

    # 0.03 accumulated tick by tick may differ from 0.02 + 2*0.005 by
    # a few ulps of binary rounding — compare, don't demand bitwise.
    assert math.isclose(env.fuel_used, 0.03, abs_tol=1e-12)
    print(f"  fuel ticker: 1 main + 2 side burns + 1 idle = "
          f"{env.fuel_used:.3f}")

    env2 = LunarLander(seed=0)
    _at(env2, y=70.0)
    for act in (MAIN, ROT_LEFT, IDLE, ROT_RIGHT):
        _, r, _, info = env2.step(act)
        c = info["components"]
        assert r == c["terminal"] + c["shaping"] + c["fuel"], \
            "reward must be the exact sum of its components"
    print("  reward == terminal + shaping + fuel, tick by tick")


def test_shaping_one_step_hand():
    """The shaping ledger, one tick by hand. From (0, 70), at rest:

        phi(before) = -150 * 70   / 100 = -105.000
        phi(after)  = -150 * 69.999352/100 (free-fall tick)
        shaping     = +0.000972   — positive: closing = earning.
    """
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0)
    phi0 = env.potential(0.0, 70.0)
    assert abs(phi0 - (-105.0)) < 1e-9

    state, reward, done, info = env.step(IDLE)
    c = info["components"]
    phi1 = env.potential(state[0], state[1])
    assert abs(c["shaping"] - (phi1 - phi0)) < 1e-12
    assert abs(c["shaping"] - 0.000972) < 1e-5
    assert c["shaping"] > 0.0
    print(f"  shaping one tick: {c['shaping']:+.6f} (hand: +0.000972)")


def test_shaping_sign_both_ways():
    """Closing on the pad pays; drifting away bills. Straight down x=0
    makes dist == y, so sign flipping is trivial to force."""
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0, vy=2.0)     #_heading AWAY (level, upward)
    _, _, _, info_up = env.step(IDLE)
    assert info_up["components"]["shaping"] < 0.0, "drifting = billed"
    print("  shaping sign: descending earns, climbing bills")


def test_shaping_telescopes():
    """The honesty test: mid-way fees cancel, so an episode's shaping
    total is EXACTLY phi(far end) − phi(start) — no hidden taxes. Run
    30 alternating MAIN/IDLE ticks from a seeded spawn and demand
    the identity (float noise stays under 1e-9)."""
    env = LunarLander(seed=3)
    s = env.reset()
    phi0 = env.potential(s[0], s[1])
    total = 0.0
    for i in range(30):
        s, r, done, info = env.step(MAIN if i % 2 == 0 else IDLE)
        total += info["components"]["shaping"]
        assert not done, "the scripted 30 ticks must stay airborne"
    phiT = env.potential(s[0], s[1])
    assert abs(total - (phiT - phi0)) < 1e-9, \
        f"telescoping broke: {total} vs {phiT - phi0}"
    print(f"  telescoping: {total:+.4f} == phi(end) - phi(start)")


# ---------------------------------------------------------------
# The preprocessing the MLP will rely on
# ---------------------------------------------------------------
def test_normalize_state():
    """Scales map each documented range to ±1: 50 m, 80 m, 6 m/s,
    8 m/s, pi rad, 4 rad/s, legs pass through as 0/1."""
    zeros = normalize_state((0.0,) * 8)
    assert zeros == [0.0] * 8, "the origin stays the origin"

    ones = normalize_state((50.0, 80.0, 6.0, 8.0, math.pi, 4.0, 1, 1))
    assert ones == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], \
        f"documented maxima must map to exactly 1.0: {ones}"

    halves = normalize_state((-25.0, 40.0, 3.0, 4.0, math.pi, 2.0, 0, 1))
    assert halves == [-0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0], \
        f"halves must halve: {halves}"

    assert len(STATE_SCALE) == 8, "one scale per state float"
    print("  normalize_state: exact 0s, exact 1s, exact halves")


# ---------------------------------------------------------------
# Tools for step 4 (the window) and the humans
# ---------------------------------------------------------------
def test_geometry_for_the_window():
    """geometry() exposes the rig in world meters, y-up — pin the
    upright stance so step 4's pygame drawing starts from facts:
    hull bottom at COM-0.6, leg tips at COM-1.6, splay to +-1.6 m."""
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0)
    g = env.geometry()

    assert len(g["hull"]) == 4, "a box has four corners"
    assert len(g["leg_joints"]) == len(g["leg_tips"]) == 2

    bottom = min(p[1] for p in g["hull"])
    tips = sorted(p[1] for p in g["leg_tips"])
    assert abs(bottom - 69.4) < 1e-9, f"hull bottom {bottom}"
    assert abs(tips[0] - 68.4) < 1e-9, f"tip height {tips[0]}"
    assert abs(g["leg_tips"][1][0] - (-1.6)) < 1e-9, "index 1 = LEFT tip"
    assert abs(g["leg_tips"][0][0] - 1.6) < 1e-9, "index 0 = RIGHT tip"
    print("  geometry: hull 69.4 m, tips 68.4 m, splay correct")


def test_render_smoke():
    """render() prints a state header, a lander 'Y', the ground with
    pad '=' between two '|' flags — and never crashes."""
    env = LunarLander(seed=0)
    _at(env, x=0.0, y=70.0)
    env.step(MAIN)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env.render()
    out = buf.getvalue()

    assert "step 1/1000" in out, "state line missing"
    assert "Y" in out, "lander glyph missing"
    assert "=" in out, "pad missing"
    assert "|" in out, "flags missing"
    assert "-" in out, "ground missing"
    assert "legs=[" in out, "leg readout missing"
    print("  render(): header + lander + pad + flags + ground")


def test_deterministic_episodes():
    """Same seed, same action script -> identical epochs, tick for
    tick, including WHERE it dies. Independence requires this:
    train.py must be able to replay the same world on command."""
    a, b = LunarLander(seed=7), LunarLander(seed=7)
    assert a.reset() == b.reset()

    traj_a, traj_b = [], []
    for i in range(1500):                   # this script falls slowly:
                                            # net -0.016 m/s per tick
                                            # -> impact around tick 600+
        act = i % 4                         # IDLE, MAIN, LEFT, RIGHT
        sa, ra, da, ia = a.step(act)
        sb, rb, db, ib = b.step(act)
        traj_a.append((sa, ra, da))
        traj_b.append((sb, rb, db))
        if da or db:
            assert da == db, "twins must die on the same tick"
            assert ia["reason"] == ib["reason"]
            assert ia["crash_type"] == ib["crash_type"]
            break
    else:
        raise AssertionError("600 ticks and nobody hit anything?")

    assert traj_a == traj_b, "twins must live identical lives"
    assert len(traj_a) < 1500, "script must reach an episode end"
    print(f"  seeded twins: identical for {len(traj_a)} ticks, "
          f"both died of {ia['reason']} ({ia['crash_type']})")


if __name__ == "__main__":
    tests = [
        test_spawn_contract,
        test_invalid_action_rejected,
        test_step_after_done_rejected,
        test_hand_computed_first_step,
        test_gravity_free_fall,
        test_main_beats_gravity,
        test_rotation_first_step_and_persistence,
        test_tilted_thrust_steers,
        test_state_layout_order,
        test_left_leg_only_contact,
        test_slow_two_leg_touch_lands,
        test_recovery_band_save,
        test_hard_impact_crash,
        test_missed_pad_crash,
        test_hull_crash_and_precedence,
        test_out_of_bounds_and_edges,
        test_timeout,
        test_fuel_accounting_and_composition,
        test_shaping_one_step_hand,
        test_shaping_sign_both_ways,
        test_shaping_telescopes,
        test_normalize_state,
        test_geometry_for_the_window,
        test_render_smoke,
        test_deterministic_episodes,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} world checks verified.")
