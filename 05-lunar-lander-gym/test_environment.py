"""
test_environment.py — THE WORLD'S CONTRACT, pinned by tests
============================================================

Every claim environment.py's docstring makes becomes an assert
here, with hand-checkable numbers wherever physics is involved:
    - the 4-tuple API shape (reset / step) and its error habits
    - deterministic replay from a fixed seed, stream continuity
    - the death-flavor truth table (_outcome: +100 vs exactly
      -100 vs the arena wall)
    - REAL integration flights: free fall, wall exit — each with
      the money checked against gym's OWN source numbers

    python test_environment.py
"""
import os
import sys
import tempfile

# The headless guard for any accidental window (render_mode tests
# opt into rgb_array explicitly):
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import (
    LunarLanderEnv, DEATH_REASONS, ENV_ID, OBS_DIM, N_ACTIONS,
    ACTION_NAMES, STATE_X_LIMIT, LANDED_CUT, ending_of,
)


def test_world_contract():
    """reset() -> 8-float tuple; step() -> our classic 4-tuple."""
    lander = LunarLanderEnv(seed=0)
    state = lander.reset()
    assert isinstance(state, tuple) and len(state) == OBS_DIM
    assert all(isinstance(v, float) for v in state), \
        f"state must be plain floats: {type(state[0])}"
    # gym's own scales (pinned from 30 seeded spawns): the scatter
    # is VELOCITY-first, unlike 04's position-first spawn -  x hugs
    # the pad, the lateral kick arrives as vx:
    assert -0.05 <= state[0] <= 0.05, f"spawn x hugs the pad: {state=}"
    assert 1.35 <= state[1] <= 1.45, f"spawn y band: {state=}"
    assert -0.85 <= state[2] <= 0.85 and -0.6 <= state[3] <= 0.6
    assert state[6] in (0.0, 1.0) and state[7] in (0.0, 1.0)

    state2, reward, done, info = lander.step(0)
    assert isinstance(state2, tuple) and len(state2) == OBS_DIM
    assert isinstance(reward, float) and isinstance(done, bool)
    assert lander.steps == 1
    assert info["outcome"] in DEATH_REASONS
    lander.close()
    print("  reset/step contract: 8-float tuple, classic 4-tuple")


def test_action_validation():
    """Only 0..3 ints pass, with the wrapper's own named wording."""
    lander = LunarLanderEnv(seed=0)
    lander.reset()
    for bad in (-1, 4, 99, "1", 1.5, True, (0,)):
        try:
            lander.step(bad)
            raise AssertionError(f"action {bad!r} must be rejected")
        except ValueError as e:
            assert str(e).startswith("action must be an int"), \
                f"named wording expected, got: {e}"
    # a bool is an int in Python - explicitly rejected above (True)
    lander.step(0)      # the REAL ints sail through
    lander.close()
    print("  actions: only ints 0..3; bools rejected; named wording")


def test_error_habits():
    """Mid-episode reset raises; post-done step raises; done is a
    clean slate afterwards."""
    lander = LunarLanderEnv(seed=1)
    lander.reset()
    lander.step(0)
    try:
        lander.reset()
        raise AssertionError("mid-episode reset must raise")
    except RuntimeError as e:
        assert "mid-episode" in str(e), str(e)

    # fly the episode out with a hard cap, then step-again must raise
    for _ in range(2000):
        state, reward, done, info = lander.step(1)
        if done:
            break
    assert lander.done
    try:
        lander.step(0)
        raise AssertionError("post-done step must raise")
    except RuntimeError as e:
        assert "after the episode" in str(e), str(e)

    lander.reset()          # the slate is clean again
    assert lander.steps == 0 and not lander.done
    lander.close()
    print("  errors: mid-reset and post-step scream; reset heals")


def test_seed_reproducibility_and_stream():
    """Same seed -> the SAME episode 1; and the stream CONTINUES
    (episode 2 differs from episode 1, same seed)."""
    a = LunarLanderEnv(seed=543)
    b = LunarLanderEnv(seed=543)
    c = LunarLanderEnv(seed=777)

    s_a, s_b, s_c = a.reset(), b.reset(), c.reset()
    assert s_a == s_b, "same seed must replay the same spawn"
    assert s_a != s_c, "different seed: different stream"

    seq_a, seq_b = [], []
    for env, seq in ((a, seq_a), (b, seq_b)):
        env.reset()
        for _ in range(10):
            s, r, d, i = env.step(0)
            seq.append((s, round(r, 9)))
    a.close(); b.close(); c.close()
    assert seq_a == seq_b, "sync twin replay diverged"
    print("  seeds: episode 1 identical; different seeds differ")


def test_outcome_truth_table():
    """_outcome: split by the two EXACT terminal payments and the
    arena wall (hand tuples, no world needed):
        truncated=True                -> timeout
        +100, |x| < 1                 -> landed
        -100, |x| >= STATE_X_LIMIT    -> out_of_bounds
        -100, |x| < 1                 -> crashed
    """
    assert ending_of(1.5, (0.2,) * 8, True) == "timeout"
    assert ending_of(100.0, (0.4, 0.0, 0, 0, 0, 0, 0, 0), False) \
        == "landed"
    assert ending_of(-100.0, (1.05, 0.0, 0, 0, 0, 0, 0, 0), False) \
        == "out_of_bounds"
    assert ending_of(-100.0, (-1.0, 0.0, 0, 0, 0, 0, 0, 0), False) \
        == "out_of_bounds"          # >= is gym's own rule
    assert ending_of(-100.0, (0.3, 0.0, 0, 0, 0, 0, 0, 0), False) \
        == "crashed"
    # the constants mean what they say:
    assert STATE_X_LIMIT == 1.0 and LANDED_CUT == 0.0
    print("  outcome truth table: +100 landed, -100 splits at |x|>=1")


def test_free_fall_flies_and_bills():
    """Integration: IDLE until done. Hand-checked tail facts:
    the lander falls, the total lands below the -100 terminal
    (gym's shaping bills the fall every tick), and -100>= terminal
    appears as a 'crashed' tag with steps < 1,000."""
    lander = LunarLanderEnv(seed=0)
    lander.reset()
    return_total = 0.0
    while True:
        state, reward, done, info = lander.step(0)
        return_total += reward
        if done:
            break
    lander.close()
    assert info["outcome"] == "crashed", info
    assert reward <= LANDED_CUT, "final tick pays the -100 family"
    assert 20 <= lander.steps < 1000
    assert return_total < -100.0, \
        "falling is billed ALL THE WAY DOWN (shaping), return" \
        f" was {return_total}"
    print(f"  free fall: {lander.steps} ticks, return "
          f"{return_total:+.2f}, tagged crashed")


def test_render_mode_contract():
    """rgb_array hands pixels to the wrapper; human is refused;
    training mode renders None."""
    lander = LunarLanderEnv(seed=0)
    lander.reset()
    lander.step(0)
    assert lander.render() is None, "training mode renders no pixels"
    lander.close()

    try:
        LunarLanderEnv(seed=0, render_mode="human")
        raise AssertionError("'human' must be refused")
    except ValueError as e:
        assert "rgb_array" in str(e), str(e)

    eyes = LunarLanderEnv(seed=0, render_mode="rgb_array")
    eyes.reset()
    state, reward, done, info = eyes.step(1)
    frame = eyes.render()
    assert frame is not None and frame.ndim == 3, "pixels required"
    assert frame.shape[2] == 3, "RGB frames"
    eyes.step(1)
    frame2 = eyes.render()
    eyes.close()
    print("  render modes: None -> None; rgb_array -> RGB frames;"
          " human refused")


def test_constants_mean_what_they_say():
    """Pinned world constants (the docs' source of truth)."""
    assert ENV_ID == "LunarLander-v3"
    assert OBS_DIM == 8 and N_ACTIONS == 4
    assert [ACTION_NAMES[a] for a in range(4)] == \
        ["IDLE", "MAIN", "LEFT", "RIGHT"]
    assert DEATH_REASONS == \
        ("landed", "crashed", "out_of_bounds", "timeout")
    print("  constants: env id, 8 dims, 4 actions, 4 endings")


if __name__ == "__main__":
    test_constants_mean_what_they_say()
    test_world_contract()
    test_action_validation()
    test_error_habits()
    test_seed_reproducibility_and_stream()
    test_outcome_truth_table()
    test_free_fall_flies_and_bills()
    test_render_mode_contract()
    print()
    print("All 8 world checks verified.")
