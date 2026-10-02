"""
test_visualize.py — headless checks for the charts (step 5)
============================================================

Runs with matplotlib's "Agg" backend — no windows pop up, but every
figure is still built and written to disk.

Covers: the smoother, the 2x2 learning curves, the two policy
slices (colored cells, gray blind spots, binner-driven axes), and
main()'s missing-input / end-to-end paths.

Run with:  python test_visualize.py
"""

import os
os.environ["MPLBACKEND"] = "Agg"   # MUST precede the matplotlib import

import json
import math
import shutil
import tempfile

import matplotlib
import matplotlib.pyplot as plt

assert matplotlib.get_backend().lower() == "agg", "headless test failed"

import visualize
from agent import QAgent, StateBinner


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
    """A tiny diary: dies fast at first, survives later, explores less."""
    return {
        "episode": list(range(1, n + 1)),
        # reward == steps in cart-pole (kept because main() checks it)
        "reward": [8.0] * (n // 2) + [500.0] * (n - n // 2),
        "outcome": ["pole_fell"] * (n // 2)
                   + ["time_limit"] * (n - n // 2),
        "epsilon": [max(0.05, 0.995 ** i) for i in range(n)],
        "steps": [8] * (n // 2) + [500] * (n - n // 2),
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
        assert "Survival steps" in titles, titles
        assert "Win rate" in titles, titles
        assert "die" in titles, titles     # the failure-mix panel
        assert "ε" in titles, titles
        plt.close(fig)
        print("  2x2 figure saved: steps / wins / epsilon / deaths")
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------
# THE POLICY SLICES
# ---------------------------------------------------------------
def test_slices_reject_never_created_rows():
    """peek_action must be READ-ONLY: an unseen state returns None
    and leaves the Q-table untouched (no optimism rows created)."""
    agent = QAgent(n_actions=2)
    assert visualize.peek_action(agent, (0.3, 0.1, 0.05, 0.2)) is None
    assert agent.q_table == {}, "peek must never create rows"
    # A visited row gets an answer (ties break to LEFT = index 0).
    agent.q_table[(5, 4, 6, 4)] = [0.0, 0.0]
    assert visualize.peek_action(agent, (0.0, 0.0, 0.0, 0.0)) == 0
    print("  unseen state -> None, table untouched; visited -> action")


def test_slices_build_and_save():
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        agent = QAgent(n_actions=2)
        # Teach one row: at the home bucket, push RIGHT hard.
        agent.q_table[(5, 4, 6, 4)] = [0.0, 7.0]

        path = os.path.join(tmp, "policy.png")
        fig = visualize.plot_policy_slice(agent, save_path=path)
        assert os.path.exists(path) and os.path.getsize(path) > 1000
        assert len(fig.axes) == 2, "two slices -> two panels"

        nx, ny = visualize.SLICE_GRID
        for ax in fig.axes:
            assert len(ax.patches) == nx * ny, "one cell per patch"
            # The home bucket sits inside the grid -> at least one
            # colored cell, but everything else stays gray.
            colors = {p.get_facecolor() for p in ax.patches}
            never = matplotlib.colors.to_rgba(visualize.C_NEVER)
            assert len(colors) == 2, "visited + never-visited expected"
            assert never in colors
        plt.close(fig)
        print(f"  2 panels x {nx}x{ny} cells; visited cell colored")
    finally:
        shutil.rmtree(tmp)


def test_slices_are_empty_but_render_for_a_fresh_brain():
    """An untrained brain: EVERYTHING is a blind spot (gray), and
    the chart must still render — honest, not broken."""
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        path = os.path.join(tmp, "empty.png")
        fig = visualize.plot_policy_slice(QAgent(n_actions=2),
                                          save_path=path)
        assert os.path.exists(path)
        never = matplotlib.colors.to_rgba(visualize.C_NEVER)
        for ax in fig.axes:
            assert all(p.get_facecolor() == never
                       for p in ax.patches), "fresh brain = all gray"
        plt.close(fig)
        print("  empty Q-table -> all-gray slices, chart still renders")
    finally:
        shutil.rmtree(tmp)


def test_slices_follow_the_agents_own_binner_ranges():
    """The axes must come from agent.binner.ranges — a brain with
    custom bucket lines gets a custom map (default: +/-12 deg tilt,
    +/-4 rad/s spin, +/-2.4 m track, +/-3 m/s speed)."""
    agent = QAgent(n_actions=2)                # default binner
    fig = visualize.plot_policy_slice(agent)
    ax_rec, ax_pos = fig.axes

    # The tilt axis is drawn in DEGREES (the environment's cliff
    # unit); every other axis keeps the binner's own unit.
    lo, hi = agent.binner.ranges[2]             # theta, in radians
    assert math.isclose(ax_rec.get_xlim()[0], math.degrees(lo),
                        abs_tol=1e-9)
    assert math.isclose(ax_rec.get_xlim()[1], math.degrees(hi),
                        abs_tol=1e-9)
    lo, hi = agent.binner.ranges[3]             # theta_dot
    assert math.isclose(ax_rec.get_ylim()[0], lo, abs_tol=1e-9)
    assert math.isclose(ax_rec.get_ylim()[1], hi, abs_tol=1e-9)
    lo, hi = agent.binner.ranges[0]             # x
    assert math.isclose(ax_pos.get_xlim()[0], lo, abs_tol=1e-9)
    lo, hi = agent.binner.ranges[1]             # x_dot
    assert math.isclose(ax_pos.get_ylim()[0], lo, abs_tol=1e-9)
    plt.close(fig)

    # A custom-eyed brain: the slice must show ITS world, not ours.
    custom = QAgent(n_actions=2, binner=StateBinner(
        n_bins=(4, 4, 4, 4),
        ranges=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)),
    ))
    fig2 = visualize.plot_policy_slice(custom)
    assert math.isclose(fig2.axes[0].get_xlim()[1],
                        math.degrees(0.5), abs_tol=1e-9)
    assert math.isclose(fig2.axes[1].get_xlim()[1], 1.0, abs_tol=1e-9)
    plt.close(fig2)
    print("  axes mirror binner.ranges, including a custom binner")


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
        for name in ("learning_curves.png", "policy_slice.png"):
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
        test_slices_reject_never_created_rows,
        test_slices_build_and_save,
        test_slices_are_empty_but_render_for_a_fresh_brain,
        test_slices_follow_the_agents_own_binner_ranges,
        test_main_reports_missing_files,
        test_main_builds_both_charts,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} chart checks verified.")
