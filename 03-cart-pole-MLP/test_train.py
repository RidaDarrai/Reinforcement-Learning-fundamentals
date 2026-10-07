"""
test_train.py — does the DQN school actually teach?
====================================================

Four things can go wrong in training even when world + brain are
perfect: the loop crashes, the numbers lie, shaping sneaks into the
charts, or nothing improves. This suite checks all four — plus the
diploma (best-exam) machinery.

WHY THESE TESTS ARE SLOWER THAN 02's:
--------------------------------------
02's tabular update was one row rewrite (1,500 episodes in seconds).
DQN does a 64-transition gradient step per environment step on CPU —
400 episodes here ≈ 40 s. So the "improvement" test runs 400 (not
1,500) episodes with gentler thresholds, and the full 1,500-episode
proof (12.9 → 278 training avg, 500/100% greedy) lives in the Step 3
commit message / README instead of CI.

Run with:  python test_train.py
"""

# Imports, each earning its line (repo rule: comment every file well).
import json      # read back training_log.json (diary must be raw + whole)
import os        # temp paths for the save-files test
import random    # seed training runs so thresholds are reproducible
import tempfile  # safe temp dir for diploma checkpoints

# The school and its knobs under test (importing the knob VALUES — not
# just the class — means silent hyperparam drift breaks loudly here).
from agent import DQNAgent
from train import (Trainer, evaluate, EPSILON_MIN, STAGE_NAMES,
                   FALL_PENALTY, EXAM_EVERY)


# ---------------------------------------------------------------
# 1. THE LOOP — runs, counts, and stops
# ---------------------------------------------------------------
def test_smoke_run():
    """A tiny session runs to completion and books every episode."""
    trainer = Trainer(episodes=5, quiet=True, seed=0)
    ran = trainer.run_all()
    assert ran == 5 and trainer.done
    assert trainer.episode == 5
    for key in ("episode", "reward", "outcome", "epsilon", "steps", "loss"):
        assert len(trainer.history[key]) == 5, f"history[{key}] incomplete"
    assert trainer.run_all() == 0, "a finished trainer must not run more"
    print("  5 episodes ran, booked, and the trainer knows it's done")


def test_outcomes_are_known_reasons():
    """Every episode ends with one of the three named endings."""
    trainer = Trainer(episodes=12, quiet=True, seed=1)
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
    trainer = Trainer(episodes=10, quiet=True, seed=2)
    trainer.run_all()
    assert trainer.history["reward"] == trainer.history["steps"], \
        "raw history drifted from steps"
    print("  10/10 episodes: recorded reward == steps (raw, unshaped)")


def test_fall_penalty_is_taught_not_recorded():
    """The brain hears the fall bill on the last tick of doomed
    episodes; the diary still writes the raw +1."""
    original = DQNAgent.update
    taught = []

    def spy(self, state, action, reward, next_state, done):
        taught.append(reward)
        return original(self, state, action, reward, next_state, done)

    DQNAgent.update = spy
    try:
        trainer = Trainer(episodes=6, quiet=True, seed=3)
        trainer.run_all()
    finally:
        DQNAgent.update = original

    assert taught, "update() was never called"
    # Alive ticks teach +1.0; doomed finals teach 1.0 + FALL_PENALTY
    # (=-9.0); time-limit finals teach +1.0 (+0 bonus). Only these.
    allowed = (1.0, 1.0 + FALL_PENALTY)
    assert all(v in allowed for v in taught), f"unexpected taught: {taught}"
    assert 1.0 + FALL_PENALTY in taught, "no episode ended (no bill seen)"
    assert trainer.history["reward"] == trainer.history["steps"]
    print(f"  brain taught in {sorted(set(taught))}; history stayed raw "
          f"(fall bill = {FALL_PENALTY:+.0f} on the fatal tick)")


# ---------------------------------------------------------------
# 3. THE SCHEDULE — epsilon decays, floors, and drives stages
# ---------------------------------------------------------------
def test_epsilon_decays_and_floors():
    """Epsilon shrinks every episode but never dips below the floor."""
    trainer = Trainer(episodes=4, quiet=True, seed=4)
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
    trainer = Trainer(episodes=1, quiet=True, seed=0)
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
    """EXAM_EVERY episodes = exactly one exam; finish() writes both
    files, and dqn_model.pt holds the exam snapshot weights."""
    trainer = Trainer(episodes=EXAM_EVERY, quiet=True, seed=5)
    trainer.run_all()

    assert trainer.best_online is not None, "the exam never ran"
    assert trainer.best_episode == EXAM_EVERY

    model_path = os.path.join(tempfile.gettempdir(), "dqn_diploma_test.pt")
    log_path = os.path.join(tempfile.gettempdir(), "log_diploma_test.json")
    try:
        trainer.finish(model_path=model_path, log_path=log_path)
        clone = DQNAgent.load(model_path)
        # Saved weights must EQUAL the diploma snapshot (not the live
        # brain, which kept training right up to episode 100).
        for saved, snap in zip(clone.online.parameters(),
                               trainer.best_online.values()):
            assert (saved.detach().cpu().numpy() == snap.numpy()).all(), \
                "saved brain must be the diploma, not the final mood"

        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        assert len(log["steps"]) == EXAM_EVERY
        assert log["reward"] == log["steps"], "log must stay raw"
    finally:
        for p in (model_path, log_path):
            if os.path.exists(p):
                os.remove(p)
    print(f"  exam at ep {trainer.best_episode} -> diploma saved, "
          f"log raw and complete")


def test_evaluate_is_greedy_and_pure():
    """Evaluation: explore=False, no learning — only numbers come
    back. update() must NEVER fire and epsilon must not move."""
    trainer = Trainer(episodes=10, quiet=True, seed=6)
    trainer.run_all()
    eps_before = trainer.agent.epsilon
    updates_before = trainer.agent.updates

    updates = []
    original = DQNAgent.update

    def spy(self, state, action, reward, next_state, done):
        updates.append(1)
        return original(self, state, action, reward, next_state, done)

    DQNAgent.update = spy
    try:
        avg, win = evaluate(trainer.agent, episodes=10)
    finally:
        DQNAgent.update = original

    assert 1.0 <= avg <= 500.0, f"avg steps out of range: {avg}"
    assert 0.0 <= win <= 100.0
    assert not updates, "evaluation must never call update()"
    assert trainer.agent.updates == updates_before, "eval trained the net"
    assert trainer.agent.epsilon == eps_before, "eval must not explore"
    print(f"  evaluate(): avg {avg:.1f} steps, {win:.0f}% wins, "
          f"brain untouched (0 updates)")


# ---------------------------------------------------------------
# 5. THE POINT OF IT ALL — training makes survival go up
# ---------------------------------------------------------------
def test_training_actually_improves_survival():
    """THE test (budget version): a seeded 400-episode session must
    leave the agent surviving clearly longer than the random
    beginner it started as (~13 steps). Full 1,500-episode proof:
    12.9 -> 278 training avg with a 500/100% greedy diploma."""
    trainer = Trainer(episodes=400, quiet=True, seed=42)
    trainer.run_all()

    first50 = sum(trainer.history["steps"][:50]) / 50
    last50 = sum(trainer.history["steps"][-50:]) / 50
    assert last50 > first50 * 2, \
        f"no visible learning: {first50:.1f} -> {last50:.1f}"
    assert last50 > 30, f"still beginner-level: {last50:.1f}"
    assert trainer.best_online is not None, "diploma system went missing"
    print(f"  avg steps {first50:.1f} -> {last50:.1f} over 400 "
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
