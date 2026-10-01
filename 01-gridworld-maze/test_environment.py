"""
test_environment.py — prove the world's rules actually work
============================================================

Before building anything ON TOP of the environment (window, agent),
we verify it. Each test below plays a hand-chosen sequence of actions
and checks the outcome — like unit-testing the rules of a board game.

Run with:  python test_environment.py
"""

from environment import (
    GridWorld, UP, DOWN, LEFT, RIGHT,
    GOAL_REWARD, PIT_REWARD, MAX_STEPS,
)


def test_walls_block_movement():
    """Bumping into a wall keeps the agent in place, still costs a step."""
    env = GridWorld()
    env.reset()
    # Start is (0,0); moving LEFT would leave the grid -> must stay put.
    state, reward, done, info = env.step(LEFT)
    assert state == (0, 0), f"expected to stay at (0,0), got {state}"
    assert info["bumped"] is True
    assert reward == -0.1 and done is False


def test_goal_path_wins():
    """Walk the safe route from S to G: down the middle, along row 4,
    then straight down the right edge - never touching the pit."""
    env = GridWorld()
    env.reset()
    path = [
        DOWN, DOWN,             # (0,0) -> (2,0)
        RIGHT, RIGHT, RIGHT,    # (2,0) -> (2,3)  (cross row 2)
        DOWN,                   # (2,3) -> (3,3)  (the single door)
        DOWN,                   # (3,3) -> (4,3)
        RIGHT, RIGHT, RIGHT,    # (4,3) -> (4,6)
        DOWN, DOWN,             # (4,6) -> (6,6) = GOAL (from above!)
    ]
    done = False
    for a in path:
        state, reward, done, info = env.step(a)
    assert done is True, "episode should end at the goal"
    assert info["reason"] == "goal"
    assert state == (6, 6)
    # The final reward on the goal cell is the pure +10 bonus
    # (ordinary steps along the way each cost -0.1).
    assert abs(reward - GOAL_REWARD) < 1e-9, f"goal reward was {reward}"
    print(f"  goal reached in {info['steps']} steps, final reward {reward:+.1f}")


def test_pit_path_loses():
    """Take the careless route along the bottom row, straight into the pit."""
    env = GridWorld()
    env.reset()
    path = [
        DOWN, DOWN,             # (0,0) -> (2,0)
        RIGHT, RIGHT, RIGHT,    # (2,0) -> (2,3)
        DOWN, DOWN,             # (2,3) -> (4,3)
        LEFT, LEFT, LEFT,       # (4,3) -> (4,0)
        DOWN, DOWN,             # (4,0) -> (6,0)
        RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,   # (6,0) -> (6,5) = PIT
    ]
    for a in path:
        state, reward, done, info = env.step(a)
    assert done is True and info["reason"] == "pit"
    assert state == (6, 5)
    assert abs(reward - PIT_REWARD) < 1e-9, f"pit reward was {reward}"
    print(f"  pit fallen into after {info['steps']} steps, reward {reward:+.1f}")


def test_timeout_ends_episode():
    """Useless actions until the 100-step limit force the episode closed."""
    env = GridWorld()
    env.reset()
    for _ in range(MAX_STEPS):
        state, reward, done, info = env.step(UP)   # pinned in top-left corner
    assert done is True and info["reason"] == "timeout"
    print(f"  timed out correctly at {info['steps']} steps")


def test_step_after_done_raises():
    """Guards against the classic 'forgot to reset' bug."""
    env = GridWorld()
    env.reset()
    env.step(DOWN)
    env.done = True            # simulate a finished episode
    try:
        env.step(DOWN)
        raise AssertionError("step() after done should have raised")
    except RuntimeError:
        print("  correctly refuses to step after the episode ends")


def test_reset_restores_fresh_episode():
    """After any episode, reset() must return everything to square one."""
    env = GridWorld()
    env.reset()
    env.step(DOWN)
    env.step(RIGHT)
    state = env.reset()
    assert state == (0, 0) and env.done is False and env.steps == 0
    print("  reset() restores start position and clears state")


if __name__ == "__main__":
    tests = [
        test_walls_block_movement,
        test_goal_path_wins,
        test_pit_path_loses,
        test_timeout_ends_episode,
        test_step_after_done_raises,
        test_reset_restores_fresh_episode,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} rules verified.")
