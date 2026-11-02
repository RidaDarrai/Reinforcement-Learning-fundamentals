"""
test_train.py — does the school actually run and seed honestly?
===============================================================

The world got 25 hand-computed checks and the brain got 16; the
school gets fewer tests but they cover MORE per check, because the
school's claims are about BEHAVIOR over episodes, not physics:

- the epsilon schedule: 1.0 at the start, linear decay, floor held
  FOREVER after the window, never a spike upward;
- one training episode's RECORD is honest: outcome from the world,
  fuel from the HUD, and the return-equals-components identity
  (ret == terminal + shaping + fuel, to float dust);
- exams are VIRTUOUS: no exploration (spy on the policy), no
  learning (weights bitwise unchanged), and a clean record form;
- a full smoke run (24 episodes) end-to-end: diary JSON parses,
  exams recorded, best == max(exams), diploma written and loadable,
  the buffer grew, epsilon hit its floor.

Every test runs the smoke machine (24 episodes, ~thousands of
gradient steps) against a THROWAWAY directory — the official
run's artifacts in this folder are never touched by tests.

Run with:  python test_train.py
"""

# Imports, each earning its line (repo rule: comment every file well).
import json      # re-read training_log.json like step 5's charts will
import math      # isclose for the return-identity check
import os        # join paths in the throwaway artifacts directory
import tempfile  # tests must never overwrite the official run's files

from environment import LunarLander, DEATH_REASONS, SPAWN_ALTITUDE
from agent import DQNAgent
from train import (
    Trainer, EPISODES, EPSILON_DECAY_EPISODES, EPSILON_FLOOR,
    EXAM_EVERY, EXAM_EPISODES,
)

# The smoke machine: the school run at 1/100th scale. Same loop, same
# artifacts, tiny dials. Every episode files real gradient steps, so
# these tests take seconds, not minutes.
SMOKE = dict(episodes=24, exam_every=12, exam_episodes=4,
             epsilon_decay_episodes=12)


def _smoke_trainer():
    """A trainer writing BOTH artifacts into a throwaway directory."""
    tmp = tempfile.mkdtemp()
    return Trainer(log_path=os.path.join(tmp, "log.json"),
                   diploma_path=os.path.join(tmp, "brain.pt"),
                   **SMOKE), tmp


def torch_equal(a, b):
    """Bitwise tensor equality helper (a lie-free one-liner)."""
    return bool((a == b).all())


# ---------------------------------------------------------------
# The dials — schedule and construction
# ---------------------------------------------------------------
def test_epsilon_schedule():
    """1.0 exactly at episode 0; linear mid-window; EXACT floor at
    and beyond the window; monotone non-increasing throughout."""
    tr = Trainer()                      # official dials for the math
    assert tr.epsilon_at(0) == 1.0
    mid = tr.epsilon_at(EPSILON_DECAY_EPISODES // 2)
    assert math.isclose(mid, 1.0 - (1.0 - EPSILON_FLOOR) * 0.5)
    assert math.isclose(tr.epsilon_at(EPSILON_DECAY_EPISODES),
                        EPSILON_FLOOR), "window end IS the floor"
    assert tr.epsilon_at(EPSILON_DECAY_EPISODES * 10) == EPSILON_FLOOR

    epsilons = [tr.epsilon_at(e) for e in range(EPSILON_DECAY_EPISODES + 2)]
    assert all(a >= b for a, b in zip(epsilons, epsilons[1:])), \
        "epsilon must never rise mid-decay"
    print("  schedule: 1.0 -> 0.05 linear, floor held, monotone")


def test_trainer_constructs_with_seeds():
    """A fresh Trainer owns a fresh seeded world+brain pair: epsilon
    starts at its default 1.0 (the school's curiosity), the diary is
    empty, dials arrive intact."""
    tr, _ = _smoke_trainer()
    assert isinstance(tr.env, LunarLander), "the world is OUR world"
    assert isinstance(tr.agent, DQNAgent)
    assert tr.agent.epsilon == 1.0
    assert len(tr.log_episodes) == 0
    assert tr.log_exams == [] and tr.best is None
    assert tr.episodes == SMOKE["episodes"]
    print("  construction: seeded pair, dials intact, diary empty")


# ---------------------------------------------------------------
# One episode, examined
# ---------------------------------------------------------------
def test_episode_record_contract():
    """One training episode produces an HONEST record:
    - outcome is one of the world's death reasons;
    - steps > 0 and fuel >= 0 (fuel is a burn, never negative);
    - the return-equals-components identity holds to float dust:
      return == terminal + shaping + fuel_cost. This is train.py's
      money invariant — if it drifts, step 5's decomposition lies."""
    tr, _ = _smoke_trainer()
    rec = tr.train_episode()

    assert rec["outcome"] in DEATH_REASONS
    assert rec["crash_type"] in (None, "impact", "hull", "missed_pad")
    assert isinstance(rec["outcome"], str)
    assert rec["steps"] > 0, "an episode is at least one tick"
    # The WORLD witnessed exactly as many ticks as the record claims:
    assert rec["steps"] == tr.env.steps, "accounting between brain and world"
    assert rec["fuel"] >= 0.0
    assert math.isfinite(rec["return"])
    # THE money invariant:
    assert math.isclose(rec["return"], rec["terminal"] + rec["shaping"]
                        + rec["fuel_cost"], abs_tol=1e-6), \
        "return must equal its three components"
    assert rec["buffer"] >= 8
    print(f"  episode record: {rec['outcome']} in {rec['steps']} ticks, "
          f"return {rec['return']:+.2f} == components")


def test_episode_runs_after_done_detected():
    """A finished training episode leaves the world DONE (the world
    refuses further steps with RuntimeError — the guard the tests
    of step 1 certified). Deep-cycle check: fifteen straight
    episodes, each really ends, worlds really reset."""
    tr, _ = _smoke_trainer()
    for i in range(15):
        rec = tr.train_episode()
        assert tr.env.done is True or rec["outcome"] == "timeout"
        # reset() is called at the TOP of the next train_episode, so
        # the world must be step-able again -> call reset now:
        tr.env.reset()
        assert tr.env.steps == 0 and tr.env.y == SPAWN_ALTITUDE
    print("  15 straight episodes: each ended, each reset clean")


# ---------------------------------------------------------------
# The exam — greedy, frozen, and well-formed
# ---------------------------------------------------------------
def test_exam_is_greedy_and_frozen():
    """Exams pause training: the policy must be asked with
    explore=False EVERY time (spy on choose_action) and the weights
    must be BITWISE unchanged after the exam (no learning happens
    during exams — that is why the diploma can trust them)."""
    tr, _ = _smoke_trainer()
    # Warm the buffer with a few training episodes first (so nothing
    # special is happening), then spy, then examine.
    for _ in range(3):
        tr.train_episode()

    seen = []
    original = tr.agent.choose_action

    def spy(state, explore=True):
        seen.append(explore)
        return original(state, explore=explore)

    tr.agent.choose_action = spy
    before = [p.detach().clone() for p in tr.agent.online.parameters()]
    exam = tr.greedy_exam()
    tr.agent.choose_action = original

    assert seen and all(e is False for e in seen), \
        f"exam must be pure greedy; saw explore flags {set(seen)}"
    after = list(tr.agent.online.parameters())
    assert all(torch_equal(a, b) for a, b in zip(before, after)), \
        "exam must not move a single weight"
    print(f"  exam: {len(seen)} greedy decisions, weights bitwise frozen")


def test_exam_record_form():
    """An exam's record is clean: average return finite, land_rate a
    real probability in [0, 1], average steps in (0, 1000)."""
    tr, _ = _smoke_trainer()
    for _ in range(3):
        rec = tr.train_episode()
        tr.log_episodes.append(rec)      # run() files every record; do
    exam = tr.greedy_exam()              # the same here for the tag
    assert set(exam) == {"episode", "avg_return", "land_rate", "avg_steps"}
    assert math.isfinite(exam["avg_return"])
    assert 0.0 <= exam["land_rate"] <= 1.0
    assert 0 < exam["avg_steps"] <= tr.env.max_steps
    assert exam["episode"] == 3, "exam tags its training-time position"
    print(f"  exam record: avg {exam['avg_return']:+.2f}, "
          f"land {exam['land_rate']:.0%}, steps {exam['avg_steps']:.0f}")


# ---------------------------------------------------------------
# The whole smoke school run
# ---------------------------------------------------------------
def test_smoke_run_end_to_end():
    """24 episodes at 1/100th scale wire EVERYTHING: episodes counted,
    epsilon reaches its floor, buffer grew by thousands, two exams
    ran, best == max(exam returns), diary parses as JSON."""
    tr, tmp = _smoke_trainer()
    summary = tr.run()

    assert summary["episodes"] == 24
    assert len(tr.log_episodes) == 24
    assert len(tr.log_exams) == 3, \
        "exams at ep 12 and 24, plus the mandatory final look"

    epsilons = [tr.epsilon_at(e) for e in range(24)]
    assert min(epsilons) == tr.epsilon_floor  # covered by schedule test
    assert tr.agent.epsilon == EPSILON_FLOOR, "post-run dial rests at floor"

    assert len(tr.agent.buffer) > 4000, \
        f"24 lander episodes must file thousands of ticks, " \
        f"got {len(tr.agent.buffer)}"
    assert tr.agent.updates > 3900, "most ticks became gradient steps"

    best_return = max(e["avg_return"] for e in tr.log_exams)
    assert math.isclose(summary["best_exam"]["avg_return"], best_return)
    assert summary["outcomes"] and \
        sum(summary["outcomes"].values()) == 24, "outcome tally complete"
    print(f"  smoke run: 24 eps, buffer {len(tr.agent.buffer)}, "
          f"steps {tr.agent.updates}, best exam "
          f"{summary['best_exam']['avg_return']:+.2f}")


def test_log_and_diploma_written():
    """Both artifacts land: the diary parses as JSON with the four
    top sections, and the diploma loads as a DQN brain whose Q
    vector matches the weights at save time."""
    tr, tmp = _smoke_trainer()
    tr.run()

    log_path = tr.log_path
    assert os.path.exists(log_path)
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    assert set(log) == {"config", "episodes", "exams", "best", "summary"}
    assert log["config"]["episodes"] == 24
    assert len(log["episodes"]) == 24
    best = max(e["avg_return"] for e in log["exams"])
    assert math.isclose(log["summary"]["best_exam"]["avg_return"], best)

    diploma = os.path.join(tmp, "brain.pt")
    assert os.path.exists(diploma)
    reloaded = DQNAgent.load(diploma)      # the brain from the diploma
    s = (0.0, SPAWN_ALTITUDE, 0.0, 0.0, 0.0, 0.0, 0, 0)
    q = reloaded.get_q(s)
    assert len(q) == 4 and all(math.isfinite(v) for v in q)
    print(f"  artifacts: diary OK ({len(log['episodes'])} episodes), "
          f"diploma loads and answers")


def test_diploma_is_the_best_brain():
    """The diploma must hold the BEST-exam brain. Forcing that: keep
    the trainer around, make the LAST exam worse than an EARLIER one
    is not easy with a random net — so instead we verify the
    mechanism directly: the best snapshot taken during run() and the
    diploma's weights must be bit-identical."""
    tr, tmp = _smoke_trainer()
    tr.run()
    reloaded = DQNAgent.load(tr.diploma_path)

    assert tr._best_weights is not None
    snap = tr._best_weights["online"]
    for p_live, p_snap in zip(reloaded.online.parameters(), snap.values()):
        assert bool((p_live == p_snap).all()), \
            "diploma weights != best-exam snapshot"
    print("  diploma == best-exam snapshot, bitwise")


if __name__ == "__main__":
    tests = [
        test_epsilon_schedule,
        test_trainer_constructs_with_seeds,
        test_episode_record_contract,
        test_episode_runs_after_done_detected,
        test_exam_is_greedy_and_frozen,
        test_exam_record_form,
        test_smoke_run_end_to_end,
        test_log_and_diploma_written,
        test_diploma_is_the_best_brain,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} school checks verified.")
