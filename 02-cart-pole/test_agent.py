"""
test_agent.py — prove the brain works before wiring it to the world
===================================================================

Four layers of trust:
  1. the eyes — does the binner bucket and clip the way we drew it?
  2. math     — does update() match a hand-computed calculation?
  3. behavior — does choose_action() really explore vs exploit?
  4. memory   — does the brain survive a trip through a JSON file?

Run with:  python test_agent.py
"""

import math
import os
import tempfile

from agent import QAgent, StateBinner, DEFAULT_N_BINS, DEFAULT_RANGES
from environment import X_LIMIT, ANGLE_LIMIT


# ---------------------------------------------------------------
# 1. THE EYES — bucketing
# ---------------------------------------------------------------
def test_binner_centers_the_world():
    """The home bucket: upright, at rest, dead center of the track.

    Hand math (all zeros):
        x        = (0 + 2.4) / 4.8 * 10 = 5
        x_dot    = (0 + 3)   / 6   * 8  = 4
        theta    = (0 + 12)  / 24  * 12 = 6      (degrees: (deg+12)/2)
        theta_dot= (0 + 4)   / 8   * 8  = 4
    """
    binner = StateBinner()
    assert binner((0.0, 0.0, 0.0, 0.0)) == (5, 4, 6, 4)
    assert binner.n_bins == DEFAULT_N_BINS
    assert binner.n_states == 10 * 8 * 12 * 8 == 7680
    print("  (0,0,0,0) -> (5,4,6,4); table capacity 7,680 rows")


def test_binner_theta_is_two_degrees_per_bucket():
    """Tilt buckets are 2° slices over ±12° — bucket = (deg + 12) / 2,
    clipped at the edges."""
    binner = StateBinner()
    for deg, bucket in [(-12, 0), (-6, 3), (0, 6), (6, 9), (12, 11)]:
        key = binner((0.0, 0.0, math.radians(deg), 0.0))
        assert key[2] == bucket, f"{deg:+}deg -> bucket {key[2]}, " \
                                 f"expected {bucket}"
    print("  tilt: -12/-6/0/+6/+12 deg -> buckets 0/3/6/9/11")


def test_binner_clips_out_of_range():
    """The world may leave the lines we drew (a pole at 20° during
    its death thump) — the table must not crash, just clip."""
    binner = StateBinner()
    key = binner((99.0, -99.0, math.radians(90), 999.0))
    assert key == (9, 0, 11, 7), f"clipping failed: {key}"
    print("  (99, -99, 90deg, 999) -> edge buckets (9,0,11,7)")


def test_binner_config_matches_environment_limits():
    """The binner's map must match the world it watches: if someone
    widens the track in environment.py, this test screams until
    DEFAULT_RANGES is updated too."""
    assert DEFAULT_RANGES[0] == (-X_LIMIT, X_LIMIT)
    assert DEFAULT_RANGES[2] == (-ANGLE_LIMIT, ANGLE_LIMIT)
    print(f"  binner ranges = environment limits "
          f"(±{X_LIMIT} m, ±{math.degrees(ANGLE_LIMIT):.0f} deg)")


def test_nearby_states_share_one_row():
    """The whole point of bucketing: moments that differ slightly
    pool their experience into ONE row; clearly different moments
    get different rows."""
    agent = QAgent(n_actions=2)

    home = (0.00, 0.0, 0.0, 0.0)      # center bucket (5,4,6,4)
    near = (0.10, 0.0, 0.0, 0.0)      # 0.48 m bucket is wide -> SAME row
    far  = (1.00, 0.0, 0.0, 0.0)      # bucket 7 -> different row

    assert agent.get_q(home) is agent.get_q(near), \
        "nearby moments should share experience"
    assert agent.get_q(home) is not agent.get_q(far), \
        "clearly different moments should not"
    print("  10 cm apart shares a row; 1 m apart does not")


# ---------------------------------------------------------------
# 2. MATH — the TD update (hand-checked, like project 1)
# ---------------------------------------------------------------
def test_update_matches_hand_calculation():
    """Q += alpha * (target - Q), target = r + gamma * max Q(next)."""
    agent = QAgent(n_actions=2, alpha=0.5, gamma=0.9)
    state = (0.0, 0.0, 0.0, 0.0)
    next_state = (0.1, 0.0, math.radians(1), 0.0)
    agent.get_q(next_state)[0] = 2.0      # best future = 2.0
    # target = 1.0 + 0.9 * 2.0 = 2.8 -> 0 + 0.5 * 2.8 = 1.4
    agent.update(state, 1, reward=1.0, next_state=next_state, done=False)
    got = agent.get_q(state)[1]
    assert abs(got - 1.4) < 1e-9, f"expected 1.4, got {got}"
    print(f"  alive update: 0 -> {got} (hand: 1.4)")


def test_terminal_update_ignores_future():
    """done=True means target = reward, even if the future looks
    rich. Here: the last survival tick of a fall = +1.0, NOT
    gamma * whatever the next bucket promised."""
    agent = QAgent(n_actions=2, alpha=0.5, gamma=0.9)
    state = (0.0, 0.0, 0.0, 0.0)
    next_state = (0.1, 0.0, math.radians(1), 0.0)
    agent.get_q(next_state)[0] = 100.0    # a juicy future value
    agent.update(state, 1, reward=1.0, next_state=next_state, done=True)
    got = agent.get_q(state)[1]
    assert abs(got - 0.5) < 1e-9, f"expected 0.5, got {got}"
    print(f"  terminal update ignores future: {got} (hand: 0.5)")


def test_unseen_state_starts_as_zeros():
    """Ignorance is 0.0 across the board — not None, not random."""
    agent = QAgent(n_actions=2)
    assert agent.get_q((0.3, -1.0, math.radians(-5), 2.0)) == [0.0, 0.0]
    print("  new bucket rows default to [0,0]")


# ---------------------------------------------------------------
# 3. BEHAVIOR — explore vs exploit
# ---------------------------------------------------------------
def test_greedy_picks_highest_q():
    """With explore=False, the best-scoring action always wins."""
    agent = QAgent(n_actions=2, epsilon=1.0)   # even at epsilon=1.0!
    state = (0.0, 0.0, 0.0, 0.0)
    agent.get_q(state)[1] = 5.0                # action RIGHT dominates
    picks = {agent.choose_action(state, explore=False) for _ in range(50)}
    assert picks == {1}, f"explore=False still wandered: {picks}"
    print("  explore=False always exploits the best action")


def test_epsilon_one_actually_explores():
    """epsilon=1.0 must produce genuinely random actions."""
    agent = QAgent(n_actions=2, epsilon=1.0)
    state = (0.0, 0.0, 0.0, 0.0)
    agent.get_q(state)[0] = 999.0              # even though 0 is 'best'
    picks = [agent.choose_action(state) for _ in range(300)]
    assert set(picks) == {0, 1}, f"never tried both: {set(picks)}"
    assert picks.count(0) < 250, "epsilon=1.0 was not random enough"
    print("  epsilon=1.0 explores both actions")


def test_epsilon_zero_never_explores():
    """epsilon=0.0 must never pick the sub-optimal action."""
    agent = QAgent(n_actions=2, epsilon=0.0)
    state = (0.0, 0.0, 0.0, 0.0)
    agent.get_q(state)[1] = 1.0                # only RIGHT is good
    picks = {agent.choose_action(state) for _ in range(200)}
    assert picks == {1}, f"explored while epsilon=0: {picks}"
    print("  epsilon=0.0 exploits only")


# ---------------------------------------------------------------
# 4. MEMORY — save / load round trip (eyes included!)
# ---------------------------------------------------------------
def test_save_load_round_trip():
    """A trained brain must survive JSON — INCLUDING its binner
    recipe. A brain without eyes reads the wrong rows, so we check
    both: same scores, same bucketing."""
    agent = QAgent(n_actions=2, alpha=0.3, gamma=0.9, epsilon=0.25)
    state_a = (0.5, 1.0, math.radians(3), -1.0)
    state_b = (-1.0, -0.5, math.radians(-8), 2.0)
    agent.get_q(state_a)[0] = 7.5
    agent.get_q(state_b)[1] = -3.0

    path = os.path.join(tempfile.gettempdir(), "q_agent_cart_test.json")
    try:
        agent.save(path)
        clone = QAgent.load(path)

        assert clone.alpha == 0.3 and clone.gamma == 0.9
        assert clone.epsilon == 0.25 and clone.n_actions == 2

        # Same scorecard (keys are bucket tuples again)...
        assert clone.q_table == agent.q_table, "Q-table changed in transit"
        assert clone.get_q(state_a) == [7.5, 0.0]
        assert clone.get_q(state_b) == [0.0, -3.0]

        # ...and same eyes: the clone must bucket a raw state
        # EXACTLY like the original.
        sample = (0.0044, 0.2208, math.radians(-1), -0.8571)
        assert clone.binner(sample) == agent.binner(sample)
        assert clone.binner.n_bins == agent.binner.n_bins
        assert clone.binner.ranges == agent.binner.ranges
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("  save/load keeps Q-table, hyperparameters, AND binner")


if __name__ == "__main__":
    tests = [
        test_binner_centers_the_world,
        test_binner_theta_is_two_degrees_per_bucket,
        test_binner_clips_out_of_range,
        test_binner_config_matches_environment_limits,
        test_nearby_states_share_one_row,
        test_update_matches_hand_calculation,
        test_terminal_update_ignores_future,
        test_unseen_state_starts_as_zeros,
        test_greedy_picks_highest_q,
        test_epsilon_one_actually_explores,
        test_epsilon_zero_never_explores,
        test_save_load_round_trip,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} agent checks verified.")
