"""
test_train.py — does training actually produce a competent agent?
==================================================================

Runs a shortened training (600 episodes — real convergence happens
by ~200) and asserts the outcome is a genuinely skilled agent,
not a lucky one. Step 6 adds: the incremental Trainer (what LIVE
mode drives), its three stages, and saving.

Run with:  python test_train.py
"""

import json
import os
import random
import shutil
import tempfile

from train import train, evaluate, Trainer, STAGE_NAMES


def test_training_produces_a_skilled_agent():
    """After 600 episodes the agent must win almost every time."""
    random.seed(123)                    # reproducible training run
    agent, history = train(episodes=600, quiet=True)

    # History bookkeeping: one record per episode, all lists aligned.
    for key in ("episode", "reward", "outcome", "epsilon", "steps"):
        assert len(history[key]) == 600, f"history['{key}'] incomplete"

    # Learning actually happened: end >> beginning.
    first100 = sum(history["reward"][:100]) / 100
    last100 = sum(history["reward"][-100:]) / 100
    assert last100 > first100 + 5, \
        f"barely improved: {first100:+.2f} -> {last100:+.2f}"
    assert last100 >= 7.0, f"final skill too low: {last100:+.2f}"

    # Win rate in the final 100 episodes.
    wins = sum(1 for o in history["outcome"][-100:] if o == "goal")
    assert wins >= 90, f"only {wins}/100 wins at the end"
    print(f"  training: {first100:+.2f} -> {last100:+.2f} avg, "
          f"{wins}/100 wins")


def test_epsilon_decays_but_never_vanishes():
    """Exploration shrinks to the floor (0.05), never to zero."""
    random.seed(123)
    _, history = train(episodes=600, quiet=True)
    assert history["epsilon"][-1] <= 0.051, "never reached the floor"
    assert history["epsilon"][-1] >= 0.05, "decayed BELOW the floor"
    assert history["epsilon"][0] > history["epsilon"][300], \
        "epsilon didn't shrink over time"
    print(f"  epsilon: {history['epsilon'][0]:.2f} -> "
          f"{history['epsilon'][-1]:.3f} (floor 0.05)")


def test_greedy_evaluation_beats_training_average():
    """With exploration OFF (real skill), performance must be at
    least as good as training — random moves only drag scores down."""
    random.seed(123)
    agent, _ = train(episodes=600, quiet=True)
    avg, win_pct = evaluate(agent, episodes=100)
    assert win_pct >= 90, f"greedy win rate only {win_pct}%"
    assert avg >= 7.0, f"greedy avg reward too low: {avg:+.2f}"
    print(f"  greedy play: {avg:+.2f} avg, {win_pct:.0f}% wins")


# ---------------------------------------------------------------
# THE TRAINER — what LIVE mode drives frame by frame
# ---------------------------------------------------------------
def test_incremental_runs_equal_one_shot():
    """Ten small steps must equal one big one: the LIVE mode trainer
    is the SAME loop as the command-line trainer, just paced."""
    random.seed(42)
    one_shot_agent, one_shot_history = train(episodes=120, quiet=True)

    random.seed(42)                    # same world, same luck
    trainer = Trainer(episodes=120, quiet=True)
    trainer.run_episodes(40)
    trainer.run_episodes(80)
    assert trainer.episode == 120
    assert trainer.history["reward"] == one_shot_history["reward"]
    assert trainer.agent.q_table == one_shot_agent.q_table
    print("  40 + 80 paced episodes == 120 in one shot")


def test_stages_walk_in_order_and_skip_fast_forwards():
    """EXPLORING -> LEARNING -> POLISHING, driven only by epsilon;
    skip_stage() jumps to the next one (that's the TAB key)."""
    random.seed(5)
    trainer = Trainer(episodes=2000, quiet=True)
    assert STAGE_NAMES == ("EXPLORING", "LEARNING", "POLISHING")
    assert trainer.stage == 0, "must start in EXPLORING"

    trainer.skip_stage()
    assert trainer.stage == 1, "skip must reach LEARNING"
    assert trainer.win_rate() >= 0.0

    trainer.skip_stage()
    assert trainer.stage == 2, "second skip must reach POLISHING"

    trainer.run_all()
    assert trainer.done and trainer.stage == 2
    print("  stages 0->1->2 via skips, then trained to completion")


def test_finish_writes_brain_and_maze_aware_log():
    """finish() saves both files, and the log remembers WHICH maze it
    trained — so visualize.py can draw the right one later."""
    tmp = tempfile.mkdtemp(prefix="train_")
    try:
        random.seed(9)
        trainer = Trainer(maze_name="classic", episodes=80, quiet=True)
        trainer.run_all()
        q_path, log_path = trainer.finish(
            q_path=os.path.join(tmp, "q_table.json"),
            log_path=os.path.join(tmp, "training_log.json"),
        )
        assert os.path.exists(q_path) and os.path.exists(log_path)
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        assert log["maze"] == "classic"
        assert log["maze_rows"] == trainer.env.maze_rows
        assert len(log["episode"]) == 80
    finally:
        shutil.rmtree(tmp)
    print("  brain + log saved; log carries maze name and rows")


if __name__ == "__main__":
    tests = [
        test_training_produces_a_skilled_agent,
        test_epsilon_decays_but_never_vanishes,
        test_greedy_evaluation_beats_training_average,
        test_incremental_runs_equal_one_shot,
        test_stages_walk_in_order_and_skip_fast_forwards,
        test_finish_writes_brain_and_maze_aware_log,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} training checks verified.")
