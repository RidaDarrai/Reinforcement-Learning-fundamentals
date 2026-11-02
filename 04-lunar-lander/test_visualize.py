"""
test_visualize.py — can the report card lie?
============================================

A chart is the project's loudest artifact: nobody re-derives the
smoothing, they just look. So every computation the charts run on
is pinned here by hand:

- the moving average, hand-checked to the decimal (the one function
  three panels trust);
- the epsilon schedule recomputation, which MUST agree tick for
  tick with Trainer.epsilon_at — the chart derives what the diary
  did not store, and a diverging derivation would draw history
  that never happened;
- the rolling shares (landing rate, crash flavors) hand-computed
  on small crafted episode lists — including the None hole a
  crash-free window must leave instead of a lying zero;
- every chart renders headless (Agg backend, no windows), saves a
  non-empty PNG into a THROWAWAY directory, and never touches the
  official run's artifacts;
- the policy slices: every cell PROBED, every fill one of the four
  button colors (an MLP answers everywhere — no gray gaps allowed).
"""

# Imports, each earning its line (repo rule: comment every file well).
import json       # write/re-read synthetic diaries through the
                  # same file channel train.py uses
import math       # isclose on the hand-checked smoothing math
import os         # tempdir paths + artifact existence checks
import tempfile   # charts must land in throwaway dirs, not charts/

# The Agg backend MUST be chosen before pyplot/visualize import
# pyplot — headless CI has no window server.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches

from agent import DQNAgent
from train import Trainer
from visualize import (
    moving_average, load_history, epsilon_schedule,
    rolling_share, crash_flavor_share,
    plot_learning_curves, plot_reward_anatomy, plot_policy_slices,
    main, ACTION_COLORS,
)


# ---------------------------------------------------------------
# The synthetic diary — deterministic, tiny, honest-shaped
# ---------------------------------------------------------------
N_EPISODES = 120


def synthetic_history():
    """A diary shaped exactly like train.py's: 120 episodes whose
    outcome mix evolves (crash-heavy early, landed-heavy late),
    exams every 30, and the same money identity the world
    guarantees (return == terminal + shaping + fuel).
    """
    episodes = []
    for i in range(N_EPISODES):
        # A rough learning arc: landing odds rise with i.
        land = (i % 10) < 9 * (i / N_EPISODES)
        if land:
            outcome, ct = "landed", None
            terminal, shaping = 100.0, 105.0
        elif i % 7 == 3:
            outcome, ct = "out_of_bounds", None
            terminal, shaping = -100.0, -40.0
        elif i % 11 == 5:
            outcome, ct = "timeout", None
            terminal, shaping = 0.0, 1.5
        elif i % 3 == 0:
            outcome, ct = "crashed", "missed_pad"
            terminal, shaping = -100.0, -20.0
        else:
            outcome, ct = "crashed", "impact"
            terminal, shaping = -100.0, -60.0
        fuel = -4.2 if land else -7.1
        episodes.append({
            "return": terminal + shaping + fuel,    # THE identity
            "steps": 300 if land else 500,
            "outcome": outcome,
            "crash_type": ct,
            "terminal": terminal, "shaping": shaping, "fuel_cost": fuel,
        })

    exams = [{"episode": e, "avg_return": -40.0 + 3.0 * e,
              "land_rate": min(1.0, e / N_EPISODES), "avg_steps": 400.0}
             for e in range(30, N_EPISODES + 1, 30)]
    return {
        "config": {"episodes": N_EPISODES,
                   "epsilon_decay_episodes": 40,
                   "epsilon_floor": 0.05,
                   "exam_every": 30, "exam_episodes": 5, "seed": 42},
        "episodes": episodes,
        "exams": exams,
        "best": exams[-1],
        "summary": {},
    }


# ---------------------------------------------------------------
# The smoother — one function three panels trust
# ---------------------------------------------------------------
def test_moving_average_hand_checked():
    """The rolling mean is the window's arithmetic mean ending at
    each position — [1, 2, 3] with window 2 MUST give a growing
    prefix then full windows: [1.0, 1.5, 2.5]."""
    assert moving_average([1.0, 2.0, 3.0], window=2) == \
        [1.0, 1.5, 2.5]
    assert moving_average([4.0], window=100) == [4.0], \
        "window bigger than the data = honest growing mean"
    assert moving_average([7, 7, 7], window=3) == [7.0, 7.0, 7.0]
    assert all(a == b for a, b in zip(moving_average([1, 2, 3], 1),
                                      [1.0, 2.0, 3.0])), \
        "window 1 is the identity"
    try:
        moving_average([1], window=0)
    except ValueError:
        pass
    else:
        raise AssertionError("window 0 must be rejected")
    print("  moving_average: hand-checked, window 1, grows honestly")


def test_epsilon_schedule_matches_trainer():
    """The chart derives epsilon from the log's config; the derivation
    must agree EXACTLY with Trainer.epsilon_at at every episode —
    otherwise the chart draws an exploration history that never
    happened."""
    tr = Trainer()                      # the official knobs
    schedule = epsilon_schedule(1500, 1000, 0.05)
    for e in range(0, 1500, 7):         # spot-check + boundaries
        assert math.isclose(schedule[e], tr.epsilon_at(e), abs_tol=1e-12)
    assert schedule[0] == 1.0
    assert schedule[999] > schedule[1000] == 0.05
    assert schedule[-1] == 0.05
    print("  epsilon recompute: identical to Trainer.epsilon_at")


def test_rolling_share_hand_checked():
    """Rolling shares on a crafted list, every entry by hand."""
    rows = [
        {"outcome": "landed"},    # i=0
        {"outcome": "crashed"},   # i=1
        {"outcome": "landed"},    # i=2
        {"outcome": "crashed"},   # i=3
        {"outcome": "landed"},    # i=4
    ]
    landed = rolling_share(rows, lambda r: r["outcome"] == "landed",
                           window=2)
    # windows END at each position: [0] | [0,1] | [1,2] | [2,3] | [3,4]
    assert landed == [100.0, 50.0, 50.0, 50.0, 50.0]

    flag = rolling_share(rows, lambda r: r["outcome"] == "landed",
                         window=1)
    assert flag == [100.0, 0.0, 100.0, 0.0, 100.0], \
        "window 1 degenerates to a pointwise flag"
    print("  rolling shares: hand-pinned, window 1 degenerates")


def test_crash_flavor_share_central_and_holes():
    """The within-crashes split: full-share windows stay 100% for a
    single flavor, and a crash-free window leaves a None HOLE (the
    chart breaks the line) instead of a lying 0%."""
    rows = [
        {"outcome": "landed",  "crash_type": None},        # i=0
        {"outcome": "crashed", "crash_type": "impact"},    # i=1
        {"outcome": "landed",  "crash_type": None},        # i=2
        {"outcome": "crashed", "crash_type": "impact"},    # i=3
        {"outcome": "landed",  "crash_type": None},        # i=4
    ]
    impact = crash_flavor_share(rows, "impact", window=2)
    assert impact == [None, 100.0, 100.0, 100.0, 100.0], \
        "every window holding crashes holds ONLY impacts -> 100"

    pointwise = crash_flavor_share(rows, "impact", window=1)
    assert pointwise == [None, 100.0, None, 100.0, None], \
        "crash-free windows must HOLE, not report a fake 0%"

    hull = crash_flavor_share(rows, "hull", window=2)
    assert hull[1] == 0.0, \
        "crashes exist but none are hulls -> an honest 0% (not None)"
    print("  crash flavors: 100/0 honest, crash-free windows hole")


# ---------------------------------------------------------------
# The charts, headless
# ---------------------------------------------------------------
def test_learning_curves_builds_and_saves():
    """The 2x2 report card renders the diary: four axes, saved to a
    throwaway PNG, and a corrupted row fails LOUDLY (a silent chart
    of nothing must be impossible)."""
    history = synthetic_history()
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "curves.png")
        fig = plot_learning_curves(history, save_path=png)
        assert len(fig.axes) == 4                       # 2x2 grid
        assert os.path.exists(png) and os.path.getsize(png) > 0
        plt.close(fig)

        broken = synthetic_history()
        del broken["episodes"][3]["outcome"]
        try:
            plot_learning_curves(broken)
        except (ValueError, KeyError):
            pass                                        # loud is right
        else:
            raise AssertionError("a missing key must fail loudly")
    print("  learning curves: 4 axes, saved, missing keys scream")


def test_reward_anatomy_builds_and_saves():
    """The two-panel anatomy figure renders: the three currencies +
    raw return, the crash-flavor split — saved headless."""
    history = synthetic_history()
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "anatomy.png")
        fig = plot_reward_anatomy(history, save_path=png)
        assert len(fig.axes) == 2
        assert os.path.exists(png) and os.path.getsize(png) > 0
        plt.close(fig)
    print("  reward anatomy: money + crash flavors, saved")


def test_policy_slices_probe_every_cell():
    """Four slices of the 8-D policy: every cell was ACTUALLY probed
    (the patch count per axis equals the grid), every fill color is
    one of the four button colors (no gray gaps, no stray colors),
    and the whole thing saves headless."""
    agent = DQNAgent(seed=0)
    grid = (6, 4)                        # tiny: 24 probes per panel
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "slices.png")
        fig = plot_policy_slices(agent, save_path=png, grid=grid)
        assert len(fig.axes) == 4
        allowed = {matplotlib.colors.to_hex(c)
                   for c in ACTION_COLORS.values()}
        for ax in fig.axes:
            patches = [p for p in ax.patches
                       if isinstance(p, matplotlib.patches.Rectangle)]
            assert len(patches) == grid[0] * grid[1], \
                f"expected {grid[0] * grid[1]} probed cells, " \
                f"got {len(patches)}"
            for p in patches:
                hexed = matplotlib.colors.to_hex(
                    tuple(p.get_facecolor()[:3]))
                assert hexed in allowed, f"stray color {hexed}"
        assert os.path.exists(png) and os.path.getsize(png) > 0
        plt.close(fig)
    print("  policy slices: every cell probed, only button colors")


def test_main_smoke_end_to_end():
    """main() with a synthetic diary + a real checkpoint builds all
    three PNGs into a throwaway dir and returns 0; with artifacts
    missing it returns 1 (the train-first hint is printed)."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "log.json")
        model_path = os.path.join(tmp, "brain.pt")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_history(), f)
        DQNAgent(seed=0).save(model_path)

        code = main(log_path=log_path, model_path=model_path,
                    out_dir=os.path.join(tmp, "charts"), show=False)
        assert code == 0
        names = set(os.listdir(os.path.join(tmp, "charts")))
        assert names == {"learning_curves.png", "reward_anatomy.png",
                         "policy_slices.png"}

        code_missing = main(log_path=os.path.join(tmp, "nope.json"),
                            model_path=model_path, show=False)
        assert code_missing == 1
    print("  main(): 3 charts generated, exit 0; missing input -> 1")


def test_diary_loader_roundtrip():
    """The charts read train.py's EXACT JSON: write the synthetic
    diary to disk, re-read it with the loader the charts use, and
    require the episode count AND the money identity untouched."""
    history = synthetic_history()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        loaded = load_history(path)
        assert len(loaded["episodes"]) == N_EPISODES
        assert len(loaded["exams"]) == 4
        r = loaded["episodes"][0]
        assert math.isclose(r["return"], r["terminal"] + r["shaping"]
                            + r["fuel_cost"], abs_tol=1e-9)
    print("  diary roundtrip: loader reads what train.py writes")


if __name__ == "__main__":
    tests = [
        test_moving_average_hand_checked,
        test_epsilon_schedule_matches_trainer,
        test_rolling_share_hand_checked,
        test_crash_flavor_share_central_and_holes,
        test_learning_curves_builds_and_saves,
        test_reward_anatomy_builds_and_saves,
        test_policy_slices_probe_every_cell,
        test_main_smoke_end_to_end,
        test_diary_loader_roundtrip,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} report-card checks verified.")
