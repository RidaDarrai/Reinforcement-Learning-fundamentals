"""
test_visualize.py — headless checks for the charts (step 5)
============================================================

Runs with matplotlib's "Agg" backend — no windows pop up, but every
figure is still built and written to disk.

Run with:  python test_visualize.py
"""

import os
os.environ["MPLBACKEND"] = "Agg"   # MUST precede the matplotlib import

import json
import shutil
import tempfile

import matplotlib
import matplotlib.pyplot as plt

assert matplotlib.get_backend().lower() == "agg", "headless test failed"

import visualize
from agent import QAgent
from environment import UP, DOWN, LEFT, RIGHT, MAZE_MAP, MAZE_CHARS, WALL


# ---------------------------------------------------------------
# THE SMOOTHER
# ---------------------------------------------------------------
def test_moving_average_hand_check():
    """Hand-calculated window=3 over [2, 4, 6, 8]."""
    assert visualize.moving_average([2, 4, 6, 8], window=3) == \
        [2.0, 3.0, 4.0, 6.0]          # partial at start, full after
    assert visualize.moving_average([], window=3) == []
    try:
        visualize.moving_average([1], window=0)
        raise AssertionError("window=0 should raise")
    except ValueError:
        pass
    print("  hand-checked [2,4,6,8] w=3 -> [2, 3, 4, 6]")


# ---------------------------------------------------------------
# THE LEARNING CURVES
# ---------------------------------------------------------------
def _fake_history(n=120):
    """A tiny diary: gets better, wins more, explores less."""
    return {
        "episode": list(range(1, n + 1)),
        "reward": [-9.5 if i < n // 2 else 8.8 for i in range(n)],
        "outcome": ["timeout"] * (n // 2) + ["goal"] * (n - n // 2),
        "epsilon": [max(0.05, 0.995 ** i) for i in range(n)],
        "steps": [100 if i < n // 2 else 12 for i in range(n)],
    }


def test_curves_reject_short_log():
    try:
        visualize.plot_learning_curves({"reward": [1.0]})
        raise AssertionError("missing keys should raise ValueError")
    except ValueError as e:
        assert "python train.py" in str(e)
    print("  incomplete log -> friendly ValueError")


def test_curves_build_and_save():
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        path = os.path.join(tmp, "curves.png")
        fig = visualize.plot_learning_curves(_fake_history(),
                                             save_path=path)
        assert os.path.exists(path) and os.path.getsize(path) > 1000
        assert len(fig.axes) == 4          # 2x2 grid
        titles = " ".join(ax.get_title() for ax in fig.axes)
        assert "reward" in titles and "Win rate" in titles
        plt.close(fig)
        print("  2x2 figure saved, 4 axes, titles present")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------
# THE POLICY MAP
# ---------------------------------------------------------------
def test_policy_map_walkable_cells_get_arrows():
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        # A brain that knows 3 non-wall cells: arrow + value each.
        agent = QAgent(n_actions=4)
        agent.q_table[(0, 0)] = [0.0, 0.0, 0.0, 4.5]   # -> RIGHT
        agent.q_table[(2, 3)] = [1.2, 3.0, 0.4, 0.9]   # -> DOWN
        agent.q_table[(4, 4)] = [0.1, 0.2, 3.3, 0.4]   # -> LEFT

        path = os.path.join(tmp, "policy.png")
        fig = visualize.plot_policy_map(agent, save_path=path)
        assert os.path.exists(path) and os.path.getsize(path) > 1000

        texts = [t.get_text() for t in fig.axes[0].texts]
        arrows = [t for t in texts if t in "↑↓←→"]
        assert len(arrows) == 3            # one per visited cell
        # Backgrounds: 49 maze cells + 1 blue ring on the start cell.
        assert len(fig.axes[0].patches) == \
            len(MAZE_MAP) * len(MAZE_MAP[0]) + 1
        plt.close(fig)
        print("  3 arrows on 3 visited cells, 49 cells + start ring")
    finally:
        shutil.rmtree(tmp)


def test_policy_map_empty_brain_survives():
    """An untrained agent (or one that never moved) still plots."""
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        path = os.path.join(tmp, "empty.png")
        fig = visualize.plot_policy_map(QAgent(n_actions=4),
                                        save_path=path)
        assert os.path.exists(path)
        plt.close(fig)
        print("  empty Q-table -> chart still renders")
    finally:
        shutil.rmtree(tmp)


def test_policy_map_draws_the_given_maze_not_the_classic_one():
    """Passing maze_rows swaps the drawing surface — the goal of a
    random maze must land where THAT maze put it."""
    custom = [
        "..S...G",
        "#######",
        ".......",
        "...#...",
        "...#...",
        ".......",
        "#######",
    ]
    agent = QAgent(n_actions=4)
    agent.q_table[(2, 0)] = [0.0, 2.0, 0.0, 0.0]   # one arrow
    fig = visualize.plot_policy_map(agent, maze_rows=custom)
    texts = [(t.get_position(), t.get_text()) for t in fig.axes[0].texts]
    assert ((6.0, 0.0), "G") in texts, "custom goal not drawn"
    # The start gets a blue ring, not a letter — at the CUSTOM 'S'.
    ring = fig.axes[0].patches[-1]          # 49 cells, then the ring
    assert tuple(ring.get_xy()) == (1.5, -0.5), ring.get_xy()
    assert len(fig.axes[0].patches) == 7 * 7 + 1
    plt.close(fig)
    print("  custom maze's own G and start ring drawn in place")


def test_main_passes_log_maze_rows_to_policy_map():
    """main() must hand the log's stored maze to the policy map."""
    tmp = tempfile.mkdtemp(prefix="viz_")
    original = visualize.plot_policy_map
    seen = {}

    def spy(agent, save_path=None, maze_rows=None):
        seen["rows"] = maze_rows
        return original(agent, save_path=save_path, maze_rows=maze_rows)

    try:
        from train import train
        agent, history = train(episodes=40, quiet=True)
        log_path = os.path.join(tmp, "training_log.json")
        q_path = os.path.join(tmp, "q_table.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        agent.save(q_path)

        visualize.plot_policy_map = spy
        code = visualize.main(log_path=log_path, q_path=q_path,
                              out_dir=os.path.join(tmp, "charts"),
                              show=False)
        assert code == 0
        assert seen["rows"] == history["maze_rows"], \
            "policy map did not receive the log's maze"
    finally:
        visualize.plot_policy_map = original
        shutil.rmtree(tmp)
    # Old logs (no maze_rows) fall back to the classic maze.
    fig = visualize.plot_policy_map(QAgent(n_actions=4))
    assert fig is not None
    plt.close(fig)
    print("  main() forwards the log's maze rows (classic fallback works)")


# ---------------------------------------------------------------
# MAIN: missing input is a friendly message, not a traceback
# ---------------------------------------------------------------
def test_main_reports_missing_files():
    empty_dir = tempfile.mkdtemp(prefix="viz_")
    try:
        code = visualize.main(
            log_path=os.path.join(empty_dir, "nope1.json"),
            q_path=os.path.join(empty_dir, "nope2.json"),
            out_dir=os.path.join(empty_dir, "charts"),
            show=False,
        )
        assert code == 1
        print("  missing inputs -> exit code 1, no exception")
    finally:
        shutil.rmtree(empty_dir)


def test_main_builds_both_charts():
    """End to end with a REAL (brief) training run — like a user
    running: python train.py; python visualize.py"""
    from train import train

    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        agent, history = train(episodes=400, quiet=True)
        log_path = os.path.join(tmp, "training_log.json")
        q_path = os.path.join(tmp, "q_table.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        agent.save(q_path)

        out_dir = os.path.join(tmp, "charts")
        code = visualize.main(log_path=log_path, q_path=q_path,
                              out_dir=out_dir, show=False)
        assert code == 0
        for name in ("learning_curves.png", "policy_map.png"):
            full = os.path.join(out_dir, name)
            assert os.path.exists(full) and os.path.getsize(full) > 1000
        print("  both PNGs built from a fresh 400-episode run")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [
        test_moving_average_hand_check,
        test_curves_reject_short_log,
        test_curves_build_and_save,
        test_policy_map_walkable_cells_get_arrows,
        test_policy_map_empty_brain_survives,
        test_policy_map_draws_the_given_maze_not_the_classic_one,
        test_main_reports_missing_files,
        test_main_builds_both_charts,
        test_main_passes_log_maze_rows_to_policy_map,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} chart checks verified.")
