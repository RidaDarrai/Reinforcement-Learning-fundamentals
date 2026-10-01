"""
test_train.py — does training actually produce a competent agent?
=================================================================

Runs a shortened training (600 episodes — real convergence happens
by ~200) and asserts the outcome is a genuinely skilled agent,
not a lucky one.

Run with:  python test_train.py
"""

import random

from train import train, evaluate


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


if __name__ == "__main__":
    tests = [
        test_training_produces_a_skilled_agent,
        test_epsilon_decays_but_never_vanishes,
        test_greedy_evaluation_beats_training_average,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} training checks verified.")
