"""
test_agent.py — prove the brain works before wiring it to the world
====================================================================

Three layers of trust:
  1. math      — does update() match a hand-computed calculation?
  2. behavior  — does choose_action() really explore vs exploit?
  3. learning  — fed real maze experiences, does it get smarter?

Run with:  python test_agent.py
"""

import os
import tempfile

from agent import QAgent
from environment import GridWorld, DOWN, UP


# ---------------------------------------------------------------
# 1. MATH
# ---------------------------------------------------------------
def test_update_matches_hand_calculation():
    """Q += alpha * (target - Q), target = r + gamma * max Q(next)."""
    agent = QAgent(n_actions=4, alpha=0.5, gamma=0.9)
    # next_state has best Q = 2.0 -> target = 1.0 + 0.9*2.0 = 2.8
    agent.get_q((9, 9))[2] = 2.0
    agent.update(state=(0, 0), action=0, reward=1.0,
                 next_state=(9, 9), done=False)
    expected = 0.0 + 0.5 * (2.8 - 0.0)      # = 1.4
    got = agent.get_q((0, 0))[0]
    assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"
    print(f"  non-terminal update: 0 -> {got} (hand: {expected})")


def test_terminal_update_ignores_future():
    """done=True means target = reward, even if the future looks rich."""
    agent = QAgent(n_actions=4, alpha=0.5, gamma=0.9)
    agent.get_q((9, 9))[0] = 100.0           # a juicy future value
    agent.update(state=(0, 0), action=0, reward=-10.0,
                 next_state=(9, 9), done=True)
    expected = 0.0 + 0.5 * (-10.0 - 0.0)     # = -5.0, NOT 100-based
    got = agent.get_q((0, 0))[0]
    assert abs(got - expected) < 1e-9, f"expected {expected}, got {got}"
    print(f"  terminal update ignores future: {got} (hand: {expected})")


def test_unseen_state_starts_as_zeros():
    """Ignorance is 0.0 across the board — not None, not random."""
    agent = QAgent(n_actions=4)
    assert agent.get_q((3, 4)) == [0.0, 0.0, 0.0, 0.0]
    print("  new states default to [0,0,0,0]")


# ---------------------------------------------------------------
# 2. BEHAVIOR
# ---------------------------------------------------------------
def test_greedy_picks_highest_q():
    """With explore=False, the best-scoring action always wins."""
    agent = QAgent(n_actions=4, epsilon=1.0)   # even at epsilon=1.0!
    agent.get_q((0, 0))[3] = 5.0               # action RIGHT dominates
    picks = {agent.choose_action((0, 0), explore=False) for _ in range(50)}
    assert picks == {3}, f"explore=False still wandered: {picks}"
    print("  explore=False always exploits the best action")


def test_epsilon_one_actually_explores():
    """epsilon=1.0 must produce genuinely random actions."""
    agent = QAgent(n_actions=4, epsilon=1.0)
    agent.get_q((0, 0))[0] = 999.0             # even though 0 is 'best'
    picks = [agent.choose_action((0, 0)) for _ in range(300)]
    assert len(set(picks)) == 4, f"never tried all actions: {set(picks)}"
    assert picks.count(0) < 250, "epsilon=1.0 was not random enough"
    print("  epsilon=1.0 explores all four actions")


def test_epsilon_zero_never_explores():
    """epsilon=0.0 must never pick the sub-optimal action."""
    agent = QAgent(n_actions=4, epsilon=0.0)
    agent.get_q((0, 0))[2] = 1.0               # only action 2 is good
    picks = {agent.choose_action((0, 0)) for _ in range(200)}
    assert picks == {2}, f"explored while epsilon=0: {picks}"
    print("  epsilon=0.0 exploits only")


# ---------------------------------------------------------------
# 3. LEARNING (frozen environment: replay one perfect episode)
# ---------------------------------------------------------------
def test_agent_learns_first_move_from_replay():
    """Replay the known winning path many times. The +10 at the goal
    must chain backwards until the agent, standing at the start,
    ranks DOWN (the first move of the path) above all other moves."""
    from environment import RIGHT, LEFT

    env = GridWorld()
    agent = QAgent(n_actions=4, alpha=0.5, gamma=0.95, epsilon=0.0)

    # The SAFE path S -> G (same route as test_environment):
    winning_path = [
        DOWN, DOWN, RIGHT, RIGHT, RIGHT, DOWN, DOWN,
        RIGHT, RIGHT, RIGHT, DOWN, DOWN,        # last move = +10
    ]

    for _ in range(300):            # 300 replays of the same perfect run
        state = env.reset()
        for action in winning_path:
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        assert info["reason"] == "goal", "path stopped being a win?!"

    q_start = agent.get_q((0, 0))
    assert q_start[DOWN] > 0, f"first move never learned: {q_start}"
    assert q_start[DOWN] == max(q_start), \
        f"DOWN should dominate the start, got {q_start}"
    print(f"  after 300 replays Q[(0,0)] = "
          f"[UP {q_start[0]:.1f}, DOWN {q_start[1]:.1f}, "
          f"LEFT {q_start[2]:.1f}, RIGHT {q_start[3]:.1f}] -> picks DOWN")


# ---------------------------------------------------------------
# 4. MEMORY — save / load round trip
# ---------------------------------------------------------------
def test_save_load_round_trip():
    """A trained brain must survive a trip through a JSON file."""
    agent = QAgent(n_actions=4, alpha=0.3, gamma=0.9, epsilon=0.25)
    agent.get_q((1, 2))[1] = 7.5
    agent.get_q((6, 6))[0] = -3.0

    path = os.path.join(tempfile.gettempdir(), "q_agent_test.json")
    try:
        agent.save(path)
        clone = QAgent.load(path)
        assert clone.alpha == 0.3 and clone.gamma == 0.9
        assert clone.epsilon == 0.25 and clone.n_actions == 4
        assert clone.get_q((1, 2)) == agent.get_q((1, 2))
        assert clone.get_q((6, 6)) == agent.get_q((6, 6))
        assert clone.q_table == agent.q_table
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  save/load keeps Q-table and hyperparameters intact")


if __name__ == "__main__":
    tests = [
        test_update_matches_hand_calculation,
        test_terminal_update_ignores_future,
        test_unseen_state_starts_as_zeros,
        test_greedy_picks_highest_q,
        test_epsilon_one_actually_explores,
        test_epsilon_zero_never_explores,
        test_agent_learns_first_move_from_replay,
        test_save_load_round_trip,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} agent checks verified.")
