"""
test_train.py — THE SCHOOL'S CONTRACT, pinned by tests
=======================================================

The loop is the reference's, so the tests pin THE REFERENCE'S
QUIRKS, not just happy paths:
    - the window accumulator: prints/reset at the border, the sum
      carries across the border otherwise (hand-checked tail sum)
    - the solve check fires EVERY episode on the window SUM (the
      partial-window quirk) and breaks + saves the SOLVED weights
    - the diary schema (config echoes the dials; per-episode keys)
    - the diploma: non-solved runs save the BEST-EXAM brain
    - exams are measurement-only (avg/land/steps records)

    python test_train.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Trainer, EPISODES, SOLVE_SUM_BAR, WINDOW, SEED, \
    GAMMA, LR, BETAS
from agent import ActorCritic


def test_default_dials_are_the_reference():
    """The reference's numbers, unpimped."""
    assert EPISODES == 10000
    assert SOLVE_SUM_BAR == 4000.0
    assert WINDOW == 20
    assert SEED == 543
    assert GAMMA == 0.99
    assert LR == 0.02
    assert BETAS == (0.9, 0.999)
    print("  dials: 10000 eps, bar 4000 (sum), window 20, seed 543,"
          " gamma 0.99, lr 0.02, betas (0.9, 0.999)")


def test_window_arithmetic():
    """With window=3 and 8 episodes: the accumulator resets at
    episodes 3 and 6, so at the END it holds exactly the last two
    returns (eps 7+8). Hand-checked against the diary."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Trainer(episodes=8, exam_every=0, window=3,
                    log_path=os.path.join(tmp, "log.json"),
                    model_path=os.path.join(tmp, "m.pt"))
        assert t.exam_every == 0, "exams off for this pin"
        summary = t.run()
        assert not summary["solved"]
        returns = [e["return"] for e in t.log_episodes]
        expected_tail = returns[6] + returns[7]
        assert abs(t.running_reward - expected_tail) < 1e-6, \
            f"accumulator tail {t.running_reward} vs " \
            f"{expected_tail}"
    print("  window arithmetic: border resets, tail sum by hand")


def test_solve_check_part_window_quirk():
    """The solve test fires on the PARTIAL window sum, every
    episode. Bar set below a first episode's return -> SOLVED at
    episode ONE, break, and the diploma is the CURRENT (solved)
    policy."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "log.json")
        model = os.path.join(tmp, "m.pt")
        t = Trainer(episodes=100, exam_every=10, exam_episodes=2,
                    solve_sum_bar=-1000.0,   # any one episode clears
                    window=20,
                    log_path=log, model_path=model)
        summary = t.run()
        assert summary["solved"] is True
        assert summary["solved_episode"] == 1
        assert len(t.log_episodes) == 1, "the loop must BREAK"
        assert os.path.exists(model), "solved weights saved"
        # the save was the CURRENT policy - reload and compare
        clone = ActorCritic.load(model)
        for w_saved, w_now in zip(clone.state_dict().values(),
                                  t.policy.state_dict().values()):
            assert torch.equal(w_saved, w_now), \
                "diploma must be the solved brain"
    print("  solve quirk: partial-window sum, break at ep 1, "
          "current brain saved")


def test_diary_schema_and_config_echo():
    """The diary shape visualize.py will bank on."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "log.json")
        t = Trainer(episodes=6, exam_every=3, exam_episodes=2,
                    window=10, seed=11,
                    log_path=log,
                    model_path=os.path.join(tmp, "m.pt"))
        summary = t.run()
        with open(log, encoding="utf-8") as f:
            diary = json.load(f)
        assert set(diary) == {"config", "episodes", "exams", "best",
                              "summary"}
        cfg = diary["config"]
        assert cfg["episodes"] == 6 and cfg["seed"] == 11
        assert cfg["gamma"] == 0.99 and cfg["lr"] == 0.02
        assert cfg["betas"] == [0.9, 0.999]
        assert cfg["solve_sum_bar"] == 4000.0 and cfg["window"] == 10
        # every episode record carries the flat keys:
        for rec in diary["episodes"]:
            assert sorted(rec) == ["episode", "loss", "outcome",
                                   "return", "steps"]
        # exams = cadence fires + the final look: eps 3, 6(+final):
        assert len(diary["exams"]) == 3          # @3, @6, final(+1)
        assert diary["exams"][-1].get("final") is True
        for exam in diary["exams"]:
            assert {"episode", "avg_return", "land_rate",
                    "avg_steps"} <= set(exam)
        assert diary["summary"]["solved"] is False
        assert abs(summary["avg_return_first_100"] -
                   sum(e["return"] for e in diary["episodes"]) / 6) \
            < 1e-9, "first-100 == the whole 6-episode mean"
    print("  diary: schema, config echo, exam cadence + final look")


def test_diploma_is_the_best_exam_brain():
    """Not-solved + exams on: the saved weights must BE the best
    exam's (02's never-trust-the-last-brain lesson), verified by
    reloading and comparing tensors."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "log.json")
        model = os.path.join(tmp, "m.pt")
        t = Trainer(episodes=6, exam_every=2, exam_episodes=2,
                    window=1000, seed=13,
                    log_path=log, model_path=model)
        t.run()
        clone = ActorCritic.load(model)
        best_dict = t._best_weights          # what run() copied aside
        for (name_saved, saved), (name_now, now) in zip(
                clone.state_dict().items(), best_dict.items()):
            assert name_saved == name_now
            assert torch.equal(saved, now), name_saved
        # and the BEST record's avg matches the exams' max:
        assert t.best["avg_return"] == max(
            e["avg_return"] for e in t.log_exams)
    print("  diploma: best-exam weights on disk, byte-identical")


def test_exam_records_measure_only():
    """Land rate in [0,1], steps positive; exams never TRAIN: the
    policy's parameter identity is preserved across an exam call."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Trainer(episodes=0, exam_every=0, exam_episodes=3,
                    window=1000, seed=5,
                    log_path=os.path.join(tmp, "log.json"),
                    model_path=os.path.join(tmp, "m.pt"))
        before = {n: w.clone() for n, w in
                  t.policy.state_dict().items()}
        exam = t.greedy_exam()
        after = t.policy.state_dict()
        for name, w in after.items():
            assert torch.equal(before[name], w), name
        assert 0.0 <= exam["land_rate"] <= 1.0
        assert exam["avg_steps"] > 0
    print("  exams: honest numbers, weights bit-frozen")


if __name__ == "__main__":
    import torch    # used for tensor compares, imported at main
    test_default_dials_are_the_reference()
    test_window_arithmetic()
    test_solve_check_part_window_quirk()
    test_diary_schema_and_config_echo()
    test_diploma_is_the_best_exam_brain()
    test_exam_records_measure_only()
    print()
    print("All 6 school checks verified.")
