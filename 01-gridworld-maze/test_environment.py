"""
test_environment.py — prove the world's rules actually work
============================================================

Before building anything ON TOP of the environment (window, agent),
we verify it. Each test below plays a hand-chosen sequence of actions
and checks the outcome — like unit-testing the rules of a board game.
Step 6 adds: the maze registry, the random generator, and the fact
that the start position is READ from the maze, not assumed.

Run with:  python test_environment.py
"""

import os
import random
import tempfile

from environment import (
    GridWorld, UP, DOWN, LEFT, RIGHT,
    GOAL_REWARD, PIT_REWARD, MAX_STEPS,
    MAZES, MAZE_MAP, VALID_MAZE_CHARS,
    generate_random_maze, path_exists,
    save_random_maze, load_random_maze, load_active_maze,
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


# ---------------------------------------------------------------
# MAZES: registry, scanning, and the random generator
# ---------------------------------------------------------------
def test_unknown_maze_name_raises_friendly():
    """A typo should list what IS available, not explode with KeyError."""
    try:
        GridWorld("atlantis")
        raise AssertionError("unknown maze name should raise")
    except ValueError as e:
        assert "classic" in str(e) and "random" in str(e)
    print("  unknown name -> ValueError listing the available mazes")


def test_every_registered_maze_is_fair():
    """Each code maze: 7x7, legal characters, one S/G/P, winnable."""
    for name, rows in MAZES.items():
        assert len(rows) == 7 and all(len(r) == 7 for r in rows), name
        for row in rows:
            assert set(row) <= VALID_MAZE_CHARS, (name, row)
        flat = "".join(rows)
        assert flat.count("S") == 1, name
        assert flat.count("G") == 1, name
        assert flat.count("P") == 1, name
        assert path_exists(rows), f"{name} maze must be winnable"
    print(f"  {len(MAZES)} registered maze(s): fair and winnable")


def test_start_position_is_scanned_not_assumed():
    """A maze whose S sits in the MIDDLE must start there — proving
    start_pos is read from the maze instead of hardcoded (0, 0)."""
    probe = [
        ".......",
        "#######",
        ".......",
        "...S...",
        ".......",
        "#######",
        ".......",
    ]
    MAZES["probe"] = probe           # temporarily join the registry
    try:
        env = GridWorld("probe")
        assert env.start_pos == (3, 3), env.start_pos
        assert env.reset() == (3, 3)
    finally:
        del MAZES["probe"]
    print("  start_pos comes from the maze's own 'S' (3, 3)")


def test_random_maze_generator_is_fair():
    """Generated mazes: right shape, legal chars, one S/G/P, winnable."""
    random.seed(7)                    # reproducible failure reports
    for _ in range(20):
        rows = generate_random_maze()
        assert len(rows) == 7 and all(len(r) == 7 for r in rows)
        flat = "".join(rows)
        for ch in flat:
            assert ch in VALID_MAZE_CHARS, ch
        assert flat.count("S") == 1 and flat.count("G") == 1
        assert flat.count("P") == 1
        assert path_exists(rows), "generated maze must be winnable"
    try:
        generate_random_maze(rows=6, cols=6)
        raise AssertionError("even sizes should be refused")
    except ValueError:
        pass
    print("  20 generated mazes: 7x7, one S/G/P, all winnable")


def test_random_maze_file_is_permanent():
    """First call generates + saves; later calls load the SAME maze —
    the 'stays until another one is created' promise."""
    tmp = os.path.join(tempfile.mkdtemp(prefix="mazes_"), "maze.json")
    try:
        # No file yet -> classic is active (and nothing is written).
        assert load_active_maze(tmp) == ("classic", MAZE_MAP)
        assert not os.path.exists(tmp)

        first = load_random_maze(tmp)          # generate + persist
        assert os.path.exists(tmp)
        assert load_active_maze(tmp)[0] == "random"
        assert load_random_maze(tmp) == first  # same maze forever
        save_random_maze(generate_random_maze(), tmp)   # "another one"
        assert load_random_maze(tmp) != first
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print("  classic first run -> random persisted -> replaced on demand")


if __name__ == "__main__":
    tests = [
        test_walls_block_movement,
        test_goal_path_wins,
        test_pit_path_loses,
        test_timeout_ends_episode,
        test_step_after_done_raises,
        test_reset_restores_fresh_episode,
        test_unknown_maze_name_raises_friendly,
        test_every_registered_maze_is_fair,
        test_start_position_is_scanned_not_assumed,
        test_random_maze_generator_is_fair,
        test_random_maze_file_is_permanent,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} rules verified.")
