"""
test_visualize.py — THE REPORT CARD'S CONTRACT, pinned by tests
================================================================

The chart file's helpers do the math; the tests pin the math and
the IO manners:
    - moving_average: hand values, window 1, growth honesty
    - rolling_share: hand pin, window-1 degeneracy, holes honest
    - load_history: roundtrip + the named missing-key screams
    - SOLVED_BAR: the reference's avg-200 line, named
    - main(): charts build from a synthetic diary into tmp; a
      missing log exits 1 with the train-first hint
    - policy slices: every cell probed with a real brain (grid cut
      to 3x3 so the test stays fast)

    python test_visualize.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize import (
    moving_average, rolling_share, load_history, main, SOLVED_BAR,
    peek_action, hex_to_rgb, ACTION_COLORS,
)
from agent import ActorCritic


def _synthetic_diary(tmp, n=60):
    """A legal diary, shaped exactly like train.py writes it."""
    episodes = []
    for i in range(1, n + 1):
        episodes.append({
            "episode": i,
            "return": -100.0 + i,               # a steady climb
            "steps": 60 + (i * 7) % 40,
            "outcome": ("crashed", "timeout", "out_of_bounds")[i % 3],
            "loss": 3.0 - 0.002 * i,
        })
    history = {
        "config": {"seed": 543, "episodes": n, "gamma": 0.99,
                   "lr": 0.02, "betas": [0.9, 0.999],
                   "solve_sum_bar": 4000.0, "window": 20,
                   "exam_every": 50, "exam_episodes": 20},
        "episodes": episodes,
        "exams": [{"episode": 30, "avg_return": -55.0,
                   "land_rate": 0.0, "avg_steps": 90.0},
                  {"episode": 60, "avg_return": -30.0,
                   "land_rate": 0.5, "avg_steps": 70.0,
                   "final": True}],
        "best": {"episode": 60, "avg_return": -30.0},
        "summary": {"solved": False, "solved_episode": None},
    }
    path = os.path.join(tmp, "log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f)
    return history, path


def test_moving_average_hand_checked():
    """The one smoothers worth pinning: window 1 degenerates, a
    window of 2 averages-in growth, zeros never pad."""
    assert moving_average([], window=5) == []
    assert moving_average([7.0], window=5) == [7.0]
    assert moving_average([1.0, 3.0, 6.0], window=1) == \
        [1.0, 3.0, 6.0]
    got = moving_average([1.0, 3.0, 6.0], window=2)
    assert got == [1.0, 2.0, 4.5], got        # 1 | 2 | 4.5
    try:
        moving_average([1], window=0)
        raise AssertionError("window 0 must be rejected")
    except ValueError:
        pass
    print("  moving average: hand pin [1, 4.5] and window-1 "
          "identity")


def test_rolling_share_hand_checked():
    """Shares are shares: crashed share 1.0 where all crash, 0.5
    where half do, no padding at the start."""
    episodes = [{"outcome": "crashed"}, {"outcome": "landed"},
                {"outcome": "crashed"}]
    assert rolling_share(episodes, "crashed", window=1) == \
        [1.0, 0.0, 1.0]
    assert rolling_share(episodes, "crashed", window=3) == \
        [1.0, 0.5, 2.0 / 3.0]
    assert rolling_share(episodes, "timeout", window=2) == \
        [0.0, 0.0, 0.0]
    print("  rolling share: half/half/third pins")


def test_load_history_roundtrip_and_screams():
    """The loader echoes what train.py writes, and SCREAMS with
    the missing key's name when the schema drifts."""
    with tempfile.TemporaryDirectory() as tmp:
        history, path = _synthetic_diary(tmp)
        loaded = load_history(path)
        assert loaded["summary"]["solved"] is False
        assert len(loaded["episodes"]) == 60
        # break the schema loudly:
        with open(path, encoding="utf-8") as f:
            broken = json.load(f)
        del broken["exams"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(broken, f)
        try:
            load_history(path)
            raise AssertionError("missing 'exams' must scream")
        except KeyError as e:
            assert "exams" in str(e), str(e)
        print("  loader: roundtrip + named screams")


def test_solved_bar_named():
    """The gym classic: average 200 over the reference's 20-window."""
    assert SOLVED_BAR == 200.0
    print("  solved bar: 200 avg (the reference's 20x4000 sum)")


def test_hex_to_rgb_threesome():
    """The slice painter's one messy bridge, pinned by hand."""
    assert hex_to_rgb("#ff0000") == (1.0, 0.0, 0.0)
    assert hex_to_rgb("#000000") == (0.0, 0.0, 0.0)
    assert tuple(round(v, 4) for v in hex_to_rgb("#3b82f6")) == \
        (0.2314, 0.5098, 0.9647)
    assert set(hex_to_rgb(k) for k in ACTION_COLORS.values()) \
        and len(ACTION_COLORS) == 4
    print("  hex->rgb: exact pins, four button colors present")


def test_policy_slices_probe_every_cell():
    """4 panels x 3x3 grid: exactly 36 argmax calls happened (the
    winners dict is the receipt, every cell probed)."""
    brain = ActorCritic(seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        history, path = _synthetic_diary(tmp)
        # reuse visualize's own slice machinery with the small grid:
        from visualize import plot_policy_slices, SLICES
        calls = []
        real_peek = peek_action
        def spied(policy, a_dim, b_dim, a_lo, a_hi, b_lo, b_hi,
                  grid, fixed=None):
            got = real_peek(policy, a_dim, b_dim, a_lo, a_hi,
                            b_lo, b_hi, grid=grid, fixed=fixed)
            calls.append(len(got))
            return got
        # monkeypatch ONLY for the receipt:
        import visualize
        visualize.peek_action = spied
        png = os.path.join(tmp, "slices.png")
        plot_policy_slices(brain, save_path=png, grid=(3, 3))
        assert len(calls) == len(SLICES) == 4
        assert all(c == 9 for c in calls), calls
        assert os.path.exists(png)
        visualize.peek_action = real_peek
    print("  slices: 4 panels x 9 cells, file written")


def test_main_smoke_end_to_end():
    """main(): the curves chart builds from the synthetic diary,
    the slices chart joins with a real brain, exit 0; a missing
    log exits 1 with the train-first hint."""
    with tempfile.TemporaryDirectory() as tmp:
        _, log_path = _synthetic_diary(tmp)
        brain = ActorCritic(seed=3)
        model_path = os.path.join(tmp, "model.pt")
        brain.save(model_path)
        charts = os.path.join(tmp, "charts")
        rc = main(log_path=log_path, model_path=model_path,
                  out_dir=charts)
        assert rc == 0
        built = sorted(os.listdir(charts))
        assert built == ["learning_curves.png", "policy_slices.png"], \
            built

        rc2 = main(log_path=os.path.join(tmp, "absent.json"),
                   model_path=model_path, out_dir=charts)
        assert rc2 == 1, "missing diary -> 1, train-first"
    print("  main(): both charts from synthetic diary; missing"
          " input -> 1")


if __name__ == "__main__":
    test_moving_average_hand_checked()
    test_rolling_share_hand_checked()
    test_load_history_roundtrip_and_screams()
    test_solved_bar_named()
    test_hex_to_rgb_threesome()
    test_policy_slices_probe_every_cell()
    test_main_smoke_end_to_end()
    print()
    print("All 7 report-card checks verified.")
