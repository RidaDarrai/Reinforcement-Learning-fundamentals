"""
test_train.py — does the school actually teach?
================================================

Four things can go wrong in training even when world + brain are
perfect: the loop crashes, the numbers lie, shaping sneaks into the
charts, or nothing improves. This suite checks all four — plus the
diploma (best-exam) machinery.

Run with:  python test_train.py
"""

import json
import os
import random
import tempfile

from agent import QAgent, DEFAULT_Q_INIT
from train import (Trainer, evaluate, EPSILON_MIN, STAGE_NAMES,
                   FALL_PENALTY, EXAM_EVERY)


# ---------------------------------------------------------------
# 1. THE LOOP — runs, counts, and stops
# ---------------------------------------------------------------
def test_smoke_run():
    """A tiny session runs to completion and books every episode."""
    trainer = Trainer(episodes=5, quiet=True)
    ran = trainer.run_all()
    assert ran == 5 and trainer.done
    assert trainer.episode == 5
    for key in ("episode", "reward", "outcome", "epsilon", "steps"):
        assert len(trainer.history[key]) == 5, f"history[{key}] incomplete"
    assert trainer.run_all() == 0, "a finished trainer must not run more"
    print("  5 episodes ran, booked, and the trainer knows it's done")


def test_outcomes_are_known_reasons():
    """Every episode ends with one of the three named endings."""
    random.seed(1)
    trainer = Trainer(episodes=12, quiet=True)
    trainer.run_all()
    valid = {"pole_fell", "cart_out", "time_limit"}
    assert set(trainer.history["outcome"]) <= valid
    print(f"  endings seen: {sorted(set(trainer.history['outcome']))}")


# ---------------------------------------------------------------
# 2. THE NUMBERS — raw scores never lie
# ---------------------------------------------------------------
def test_reward_always_equals_steps():
    """+1 per tick means reward == steps in EVERY episode — even
    though the brain is separately TAUGHT the fall bill
    (FALL_PENALTY). Shaping must never reach the charts."""
    random.seed(2)
    trainer = Trainer(episodes=10, quiet=True)
    trainer.run_all()
    assert trainer.history["reward"] == trainer.history["steps"], \
        "raw history drifted from steps"
    print("  10/10 episodes: recorded reward == steps (raw, unshaped)")


def test_fall_penalty_is_taught_not_recorded():
    """The brain hears -30 on the last tick of doomed episodes;
    the diary still writes the raw +1."""
    original = QAgent.update
    taught = []

    def spy(self, state, action, reward, next_state, done):
        taught.append(reward)
        original(self, state, action, reward, next_state, done)

    QAgent.update = spy
    try:
        random.seed(3)
        trainer = Trainer(episodes=6, quiet=True)
        trainer.run_all()
    finally:
        QAgent.update = original

    assert taught, "update() was never called"
    allowed = (1.0, 1.0 + FALL_PENALTY)     # alive tick, or fall bill
    assert all(v in allowed for v in taught), f"unexpected taught: {taught}"
    assert 1.0 + FALL_PENALTY in taught, "no episode ended (no bill seen)"
    assert trainer.history["reward"] == trainer.history["steps"]
    print(f"  brain taught in {set(taught)}; history stayed raw "
          f"(fall bill = {FALL_PENALTY:+.0f} on the fatal tick)")


# ---------------------------------------------------------------
# 3. THE SCHEDULE — epsilon decays, floors, and drives stages
# ---------------------------------------------------------------
def test_epsilon_decays_and_floors():
    """Epsilon shrinks every episode but never dips below the floor."""
    random.seed(4)
    trainer = Trainer(episodes=4, quiet=True)
    trainer.agent.epsilon = 0.0501           # a hair above the floor
    trainer.run_all()
    assert trainer.agent.epsilon == EPSILON_MIN, \
        f"floor not respected: {trainer.agent.epsilon}"

    eps = trainer.history["epsilon"]
    assert all(a >= b for a, b in zip(eps, eps[1:])), \
        "epsilon must never rise mid-run"
    assert all(e >= EPSILON_MIN for e in eps)
    print(f"  epsilon decayed monotonically to the floor {EPSILON_MIN}")


def test_stage_follows_epsilon():
    """Stage names are a pure function of epsilon (the LIVE-mode
    dashboard reads this same property)."""
    trainer = Trainer(episodes=1, quiet=True)
    trainer.agent.epsilon = 0.9
    assert trainer.stage == 0 and STAGE_NAMES[trainer.stage] == "EXPLORING"
    trainer.agent.epsilon = 0.3
    assert trainer.stage == 1 and STAGE_NAMES[trainer.stage] == "LEARNING"
    trainer.agent.epsilon = EPSILON_MIN
    assert trainer.stage == 2 and STAGE_NAMES[trainer.stage] == "POLISHING"
    print("  eps 0.9/0.3/floor -> EXPLORING/LEARNING/POLISHING")


# ---------------------------------------------------------------
# 4. THE DIPLOMA — exams run, best brain gets saved
# ---------------------------------------------------------------
def test_finish_saves_the_diploma():
    """500 episodes = exactly one exam; finish() writes both files,
    and q_table.json holds the exam snapshot — with the binner
    recipe and optimism intact (QAgent.load proved in test_agent)."""
    random.seed(5)
    trainer = Trainer(episodes=EXAM_EVERY, quiet=True)
    trainer.run_all()

    assert trainer.best_q is not None, "the exam never ran"
    assert trainer.best_episode == EXAM_EVERY
    assert trainer.agent.q_table is not trainer.best_q, \
        "diploma must be a separate snapshot"

    q_path = os.path.join(tempfile.gettempdir(), "q_diploma_test.json")
    log_path = os.path.join(tempfile.gettempdir(), "log_diploma_test.json")
    try:
        trainer.finish(q_path=q_path, log_path=log_path)
        clone = QAgent.load(q_path)
        assert clone.q_table == trainer.best_q, \
            "saved brain must be the diploma, not the final mood"
        assert clone.q_init == DEFAULT_Q_INIT

        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        assert len(log["steps"]) == EXAM_EVERY
        assert log["reward"] == log["steps"], "log must stay raw"
    finally:
        for p in (q_path, log_path):
            if os.path.exists(p):
                os.remove(p)
    print(f"  exam at ep {trainer.best_episode} -> diploma saved, "
          f"log raw and complete")


def test_evaluate_is_greedy_and_pure():
    """Evaluation: explore=False, no learning — only numbers come
    back. Reading a never-seen bucket does lazily CREATE its row
    (choose_action must answer somehow), so the honest claim is:
    no update() ever fires, and every created row is untouched
    q_init defaults — looked at, never written."""
    random.seed(6)
    trainer = Trainer(episodes=30, quiet=True)
    trainer.run_all()
    keys_before = set(trainer.agent.q_table)
    eps_before = trainer.agent.epsilon

    updates = []
    original = QAgent.update

    def spy(self, state, action, reward, next_state, done):
        updates.append(1)
        original(self, state, action, reward, next_state, done)

    QAgent.update = spy
    try:
        avg, win = evaluate(trainer.agent, episodes=20)
    finally:
        QAgent.update = original

    assert 1.0 <= avg <= 500.0, f"avg steps out of range: {avg}"
    assert 0.0 <= win <= 100.0
    assert not updates, "evaluation must never call update()"
    for key in set(trainer.agent.q_table) - keys_before:
        assert trainer.agent.q_table[key] == [DEFAULT_Q_INIT] * 2, \
            "eval wrote data into a fresh row"
    assert trainer.agent.epsilon == eps_before, "eval must not explore"
    print(f"  evaluate(): avg {avg:.1f} steps, {win:.0f}% wins, "
          f"brain untouched (0 updates)")


# ---------------------------------------------------------------
# 5. THE POINT OF IT ALL — training makes survival go up
# ---------------------------------------------------------------
def test_training_actually_improves_survival():
    """THE test: a seeded 1,500-episode session must leave the
    agent surviving several times longer than the random beginner
    it started as (~10-15 steps)."""
    random.seed(42)
    trainer = Trainer(episodes=1500, quiet=True)
    trainer.run_all()

    first50 = sum(trainer.history["steps"][:50]) / 50
    last50 = sum(trainer.history["steps"][-50:]) / 50
    assert last50 > first50 * 3, \
        f"no visible learning: {first50:.1f} -> {last50:.1f}"
    assert last50 > 50, f"still beginner-level: {last50:.1f}"
    assert trainer.best_q is not None, "diploma system went missing"
    print(f"  avg steps {first50:.1f} -> {last50:.1f} over 1,500 "
          f"episodes; best exam {trainer.best_score[0]:.0f}% at "
          f"ep {trainer.best_episode}")


if __name__ == "__main__":
    tests = [
        test_smoke_run,
        test_outcomes_are_known_reasons,
        test_reward_always_equals_steps,
        test_fall_penalty_is_taught_not_recorded,
        test_epsilon_decays_and_floors,
        test_stage_follows_epsilon,
        test_finish_saves_the_diploma,
        test_evaluate_is_greedy_and_pure,
        test_training_actually_improves_survival,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} training checks verified.")
