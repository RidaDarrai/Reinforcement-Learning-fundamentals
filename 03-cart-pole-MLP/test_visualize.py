"""
test_visualize.py — headless checks for the MLP charts (step 5)
================================================================

Runs with matplotlib's "Agg" backend — no windows pop up, but every
figure is still built and written to disk.

Covers: the smoother, the 2x2 learning curves, the two MLP policy
slices (blue/orange cells everywhere — NO gray blind spots, unlike
02's table), the canonical axis ranges, and main()'s missing-input
/ end-to-end paths.

Run with:  python test_visualize.py
"""

# MPLBACKEND MUST precede the matplotlib import, or windows pop up.
import os
os.environ["MPLBACKEND"] = "Agg"   # noqa: E402  (order matters)

# Every import earns its comment (repo rule: comment every file well).
import json        # write a fake training log for the end-to-end test
import math        # isclose: axis limits must match world limits exactly
import shutil      # remove temp chart dirs after each test
import tempfile    # isolated dirs — tests never touch the real charts/

import matplotlib                     # assert the backend really is Agg
import matplotlib.pyplot as plt       # close figures (else they pile up)
import torch                          # snapshot weights: peek is read-only?

assert matplotlib.get_backend().lower() == "agg", "headless test failed"

import visualize  # the report card under test (smoother, slices, main)
from agent import DQNAgent
from environment import ANGLE_LIMIT, X_LIMIT


# A deterministic stub brain: push RIGHT iff the pole leans right
# (theta > 0), else LEFT. Gives the slice exactly two colors on
# purpose — a real net's boundary is random until trained.
class _LeanRightAgent:
    """Test double with the same get_q contract as DQNAgent."""

    def get_q(self, state):
        return [0.0, 1.0] if state[2] > 0 else [1.0, 0.0]


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
    """A tiny diary: dies fast at first, survives later, explores less.

    Includes the DQN-only "loss" key (real logs carry it) to prove
    the curves tolerate — and ignore — extra keys.
    """
    return {
        "episode": list(range(1, n + 1)),
        # reward == steps in cart-pole (kept because main() checks it)
        "reward": [8.0] * (n // 2) + [500.0] * (n - n // 2),
        "outcome": ["pole_fell"] * (n // 2)
                   + ["time_limit"] * (n - n // 2),
        "epsilon": [max(0.05, 0.995 ** i) for i in range(n)],
        "steps": [8] * (n // 2) + [500] * (n - n // 2),
        "loss": [1.5] * n,   # ignored by the curves (by design)
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
# THE MLP POLICY SLICES (no gray — the net answers everywhere)
# ---------------------------------------------------------------
def test_peek_is_argmax_and_read_only():
    """peek_action = argmax get_q, and it must not move any weight
    (charts observe the brain; they never train it)."""
    assert visualize.peek_action(_LeanRightAgent(), (0, 0, 0.1, 0)) == 1
    assert visualize.peek_action(_LeanRightAgent(), (0, 0, -0.1, 0)) == 0

    agent = DQNAgent(seed=0)
    before = [p.detach().clone() for p in agent.online.parameters()]
    visualize.peek_action(agent, (0.3, 0.1, 0.05, 0.2))
    for b, p in zip(before, agent.online.parameters()):
        assert torch.equal(b, p), "peek moved a weight!"
    print("  stub rule respected; real weights untouched by peeking")


def test_slices_build_and_save():
    """The lean-right stub splits RECOVERY into blue AND orange (the
    tilt sign decides), with one cell per grid patch."""
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        path = os.path.join(tmp, "policy.png")
        fig = visualize.plot_policy_slice(_LeanRightAgent(),
                                          save_path=path)
        assert os.path.exists(path) and os.path.getsize(path) > 1000
        assert len(fig.axes) == 2, "two slices -> two panels"

        nx, ny = visualize.SLICE_GRID
        left_colors = {p.get_facecolor() for p in fig.axes[0].patches}
        assert len(fig.axes[0].patches) == nx * ny
        assert len(fig.axes[1].patches) == nx * ny
        # Recovery spans ±tilt: both halves exist -> both colors.
        want = {matplotlib.colors.to_rgba(visualize.C_LEFT),
                matplotlib.colors.to_rgba(visualize.C_RIGHT)}
        assert left_colors == want, "tilt rule must split the panel"
        plt.close(fig)
        print(f"  2 panels x {nx}x{ny} cells; tilt rule splits blue/orange")
    finally:
        shutil.rmtree(tmp)


def test_slices_have_no_gray_for_any_brain():
    """Even a FRESH MLP answers every cell (blue/orange only) — the
    honest DQN difference from 02's gray blind-spot map."""
    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        path = os.path.join(tmp, "fresh.png")
        fig = visualize.plot_policy_slice(DQNAgent(seed=0),
                                          save_path=path)
        assert os.path.exists(path)
        allowed = {matplotlib.colors.to_rgba(visualize.C_LEFT),
                   matplotlib.colors.to_rgba(visualize.C_RIGHT)}
        for ax in fig.axes:
            assert {p.get_facecolor() for p in ax.patches} <= allowed, \
                "MLP slices must never draw gray"
        plt.close(fig)
        print("  fresh net -> only blue/orange (generalization, not gaps)")
    finally:
        shutil.rmtree(tmp)


def test_slice_axes_match_world_limits():
    """Axes must equal the canonical limits: tilt ±12° (drawn in
    DEGREES), spin ±4, x ±2.4 m, x_dot ±3 — never retyped numbers."""
    fig = visualize.plot_policy_slice(_LeanRightAgent())
    ax_rec, ax_pos = fig.axes

    assert math.isclose(ax_rec.get_xlim()[0], -12.0, abs_tol=1e-9)
    assert math.isclose(ax_rec.get_xlim()[1], 12.0, abs_tol=1e-9)
    assert math.isclose(ax_rec.get_ylim()[0], -4.0, abs_tol=1e-9)
    assert math.isclose(ax_rec.get_ylim()[1], 4.0, abs_tol=1e-9)
    assert math.isclose(ax_pos.get_xlim()[0], -X_LIMIT, abs_tol=1e-9)
    assert math.isclose(ax_pos.get_xlim()[1], X_LIMIT, abs_tol=1e-9)
    assert math.isclose(ax_pos.get_ylim()[1], 3.0, abs_tol=1e-9)
    # And those canonical ranges must THEMSELVES match the source
    # modules (a silent physics/scale change breaks the chart here).
    assert math.isclose(math.degrees(ANGLE_LIMIT), 12.0, abs_tol=1e-9)
    assert visualize.SLICE_X == (-X_LIMIT, X_LIMIT)
    plt.close(fig)
    print("  axes = ±12° / ±4 / ±2.4 m / ±3 m/s, matching env + agent")


# ---------------------------------------------------------------
# MAIN: missing input is a friendly message, not a traceback
# ---------------------------------------------------------------
def test_main_reports_missing_files():
    empty_dir = tempfile.mkdtemp(prefix="viz_")
    try:
        code = visualize.main(
            log_path=os.path.join(empty_dir, "nope1.json"),
            model_path=os.path.join(empty_dir, "nope2.pt"),
            out_dir=os.path.join(empty_dir, "charts"),
            show=False,
        )
        assert code == 1
        print("  missing inputs -> exit code 1, no exception")
    finally:
        shutil.rmtree(empty_dir)


def test_main_builds_both_charts():
    """End to end with a REAL (brief) DQN run — like a user running:
    python train.py; python visualize.py (60 episodes keeps it fast)."""
    from train import train

    tmp = tempfile.mkdtemp(prefix="viz_")
    try:
        agent, history = train(episodes=60, quiet=True, seed=0)
        log_path = os.path.join(tmp, "training_log.json")
        model_path = os.path.join(tmp, "dqn_model.pt")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        agent.save(model_path)

        out_dir = os.path.join(tmp, "charts")
        code = visualize.main(log_path=log_path, model_path=model_path,
                              out_dir=out_dir, show=False)
        assert code == 0
        for name in ("learning_curves.png", "policy_slice.png"):
            full = os.path.join(out_dir, name)
            assert os.path.exists(full) and os.path.getsize(full) > 1000
        print("  both PNGs built from a fresh 60-episode DQN run")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [
        test_moving_average_hand_check,
        test_curves_reject_short_log,
        test_curves_build_and_save,
        test_slices_build_and_save,
        test_peek_is_argmax_and_read_only,
        test_slices_have_no_gray_for_any_brain,
        test_slice_axes_match_world_limits,
        test_main_reports_missing_files,
        test_main_builds_both_charts,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} chart checks verified.")
