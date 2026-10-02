"""
test_environment.py — does the physics actually behave?
========================================================

The maze could be eyeballed; physics can lie convincingly. So the
bar here is higher: one test computes the FIRST step of a push BY
HAND (see test_hand_computed_first_step) and demands the code match
it to six decimals. If that passes, the equations are trustworthy.

Run with:  python test_environment.py
"""

import contextlib
import io
import math
import random

from environment import (
    CartPole, LEFT, RIGHT, N_ACTIONS, ACTION_NAMES, MAX_STEPS,
    X_LIMIT, ANGLE_LIMIT, ANGLE_LIMIT_DEG, START_NOISE, REWARD_PER_STEP,
)


def _at(env, x=0.0, x_dot=0.0, theta=0.0, theta_dot=0.0):
    """Place the world at an EXACT spot (tests poke the physics).

    reset() is random by design, but physics tests need known
    starting numbers — so we set the four state fields directly.
    """
    env.x, env.x_dot = x, x_dot
    env.theta, env.theta_dot = theta, theta_dot
    env.steps = 0
    env.done = False
    return env


# ---------------------------------------------------------------
# reset() — fresh, noisy, within bounds
# ---------------------------------------------------------------
def test_reset_bounds_and_seed():
    """reset() returns 4 small floats, resets counters, and obeys
    random.seed() so tests (and someday demos) are reproducible."""
    env = CartPole()

    random.seed(11)
    state = env.reset()
    assert len(state) == 4, "state must be the four numbers"
    assert all(isinstance(v, float) for v in state), "all floats"
    assert all(abs(v) <= START_NOISE for v in state), \
        f"reset noise must stay within ±{START_NOISE}: {state}"
    assert env.steps == 0 and env.done is False
    assert env.state == state, "state property must match reset"

    random.seed(11)               # same seed -> identical episode start
    assert env.reset() == state, "seeded resets must repeat"

    random.seed(12)               # different seed -> different world
    assert env.reset() != state, "different seeds must differ"
    print(f"  reset: 4 floats in +/-{START_NOISE}, seeded & repeatable")


def test_invalid_action_rejected():
    """Only two buttons exist; anything else gets a friendly error."""
    env = CartPole()
    env.reset()
    assert N_ACTIONS == 2 and LEFT == 0 and RIGHT == 1
    assert ACTION_NAMES == {0: "LEFT", 1: "RIGHT"}

    for bad in (2, -1, "left", None):
        try:
            env.step(bad)
        except ValueError as e:
            assert "Invalid action" in str(e)
        else:
            raise AssertionError(f"step({bad!r}) should have failed")
    print("  only 0=LEFT / 1=RIGHT accepted; others raise ValueError")


def test_step_after_done_rejected():
    """Forgetting reset() between episodes fails loudly, not silently."""
    env = CartPole()
    _at(env, theta=0.5)           # already past the limit: dies step 1
    env.step(RIGHT)
    assert env.done
    try:
        env.step(LEFT)
    except RuntimeError as e:
        assert "reset()" in str(e)
    else:
        raise AssertionError("step() after done should have failed")
    print("  step() after episode end raises a clear RuntimeError")


# ---------------------------------------------------------------
# THE BIG ONE — the first step, computed on paper
# ---------------------------------------------------------------
def test_hand_computed_first_step():
    """One push from perfect rest must match a hand calculation.

    Hand derivation (state all zeros, push RIGHT = +10 N):

        temp      = 10 / 1.1                    = 9.090909...
        theta_acc = (9.8*0 - 1*temp)            = -42.857143
                    / (0.5 * (4/3 - 1/1.1))         rad/s²
        x_acc     = temp - 0.05*theta_acc/1.1    = 11.038961   m/s²

    then semi-implicit integration (speeds first) with DT = 0.02:

        x_dot     = 0.02 * 11.038961             = +0.220779
        x         = 0.02 * 0.220779              = +0.004416
        theta_dot = 0.02 * -42.857143            = -0.857143
        theta     = 0.02 * -0.857143             = -0.017143 rad

    If any of these six-decimal numbers move, the physics changed.
    """
    env = CartPole()
    _at(env)                       # exact rest: all zeros

    state, reward, done, info = env.step(RIGHT)

    x, x_dot, theta, theta_dot = state
    assert abs(x_dot - 0.220779) < 1e-6, f"x_dot={x_dot:.7f}"
    assert abs(x - 0.004416) < 1e-6, f"x={x:.7f}"
    assert abs(theta_dot - -0.857143) < 1e-6, f"theta_dot={theta_dot:.7f}"
    assert abs(theta - -0.017143) < 1e-6, f"theta={theta:.7f}"
    assert reward == REWARD_PER_STEP and not done
    print(f"  hand-computed step matches: x={x:+.6f}, "
          f"theta={theta:+.6f} rad")


def test_push_right_tips_pole_left():
    """The signature move of cart-pole: shove the cart RIGHT and the
    pole lags BEHIND (tilts LEFT) — inertia, like a hanging
    pendulum swinging back when the car accelerates."""
    env = CartPole()
    _at(env)
    env.step(RIGHT)

    assert env.x > 0 and env.x_dot > 0, "cart must accelerate right"
    assert env.theta < 0, "pole must tip LEFT (behind the push)"
    print("  cart right + pole left: inertia shows up in the numbers")


def test_mirror_symmetry():
    """Left and right are equal citizens: mirroring the whole state
    and flipping the action must mirror the result exactly."""
    env1 = CartPole()
    env2 = CartPole()
    _at(env1, x=0.3, x_dot=0.2, theta=0.1, theta_dot=-0.15)
    _at(env2, x=-0.3, x_dot=-0.2, theta=-0.1, theta_dot=0.15)

    env1.step(RIGHT)
    env2.step(LEFT)

    assert math.isclose(env1.x, -env2.x, abs_tol=1e-12)
    assert math.isclose(env1.x_dot, -env2.x_dot, abs_tol=1e-12)
    assert math.isclose(env1.theta, -env2.theta, abs_tol=1e-12)
    assert math.isclose(env1.theta_dot, -env2.theta_dot, abs_tol=1e-12)
    print("  mirrored state + mirrored action = mirrored outcome")


# ---------------------------------------------------------------
# The three ways an episode can end
# ---------------------------------------------------------------
def test_pole_fell_termination():
    """Past 12° means the pole is down — episode over, reason named."""
    env = CartPole()
    _at(env, theta=math.radians(ANGLE_LIMIT_DEG + 5))
    state, reward, done, info = env.step(RIGHT)

    assert done and env.done
    assert info["reason"] == "pole_fell"
    assert info["steps"] == 1
    # Right at the edge: 12° exactly is still alive (the limit is
    # strictly greater), one degree past is not.
    env2 = CartPole()
    _at(env2, theta=ANGLE_LIMIT)          # exactly 12°
    assert env2.step(RIGHT)[2] is False, "exactly 12° must survive"
    print(f"  |angle| > {ANGLE_LIMIT_DEG:.0f}deg -> pole_fell; "
          f"exactly {ANGLE_LIMIT_DEG:.0f}deg -> alive")


def test_cart_out_termination():
    """Past ±2.4 m the cart has left the track."""
    env = CartPole()
    _at(env, x=X_LIMIT + 0.1)
    state, reward, done, info = env.step(RIGHT)

    assert done and info["reason"] == "cart_out"
    print(f"  |x| > {X_LIMIT}m -> cart_out")


def test_time_limit_termination():
    """Survive max_steps -> the GOOD ending. Alternating pushes keep
    the tiny episode safely balanced so only the clock can end it."""
    env = CartPole(max_steps=4)
    env.reset()
    for i in range(4):
        state, reward, done, info = env.step(LEFT if i % 2 == 0 else RIGHT)
        if done:
            break
    assert done and info["reason"] == "time_limit", \
        f"clean episode ended early: {info}"
    assert env.steps == 4
    print("  4 gentle steps -> time_limit (surviving IS winning)")


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------
def test_reward_is_one_per_step():
    """+1 every tick, INCLUDING the terminal tick: episode total
    reward always equals steps survived."""
    env = CartPole(max_steps=5)
    env.reset()
    rewards = []
    for i in range(5):
        _, r, done, _ = env.step(LEFT if i % 2 == 0 else RIGHT)
        rewards.append(r)
        if done:
            break

    assert rewards == [REWARD_PER_STEP] * len(rewards)
    assert sum(rewards) == env.steps, "total reward must equal steps"
    print(f"  {len(rewards)} steps -> total reward {sum(rewards):.1f}")


def test_info_reports_reason():
    """reason is None while alive, a string once done — and steps
    counts every action."""
    env = CartPole()
    _at(env)
    _, _, done, info = env.step(RIGHT)
    assert done is False and info["reason"] is None
    assert info["steps"] == 1

    _at(env, theta=math.radians(20))       # force a fall next step
    _, _, done, info = env.step(LEFT)
    assert done and info["reason"] == "pole_fell"
    print("  info: None while alive, named reason when done")


def test_render_smoke():
    """render() prints a state line, the pole, and the cart — and
    never crashes (it's also what the demo draws)."""
    env = CartPole()
    env.reset()
    env.step(RIGHT)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env.render()
    out = buf.getvalue()

    assert "step 1/" in out, "state line missing"
    assert "[C]" in out, "cart missing"
    assert "|" in out, "pole missing"
    assert "-" in out, "track missing"
    print("  render(): state line + pole + cart + track")


if __name__ == "__main__":
    tests = [
        test_reset_bounds_and_seed,
        test_invalid_action_rejected,
        test_step_after_done_rejected,
        test_hand_computed_first_step,
        test_push_right_tips_pole_left,
        test_mirror_symmetry,
        test_pole_fell_termination,
        test_cart_out_termination,
        test_time_limit_termination,
        test_reward_is_one_per_step,
        test_info_reports_reason,
        test_render_smoke,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} physics checks verified.")
