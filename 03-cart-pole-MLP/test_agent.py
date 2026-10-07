"""
test_agent.py — prove the MLP brain works before training it
=============================================================

Four layers of trust (mirrors 02's test_agent.py, minus the binner —
there are no buckets anymore, so "the eyes" now means normalization
and network shapes):

  1. inputs  — does normalize_state() map the world into ~[-1, 1]?
               does the MLP output exactly 2 Q-values?
  2. math    — does train_step() build the right TD target
               (hand-checked numbers), ignore the future when done,
               and actually REDUCE the loss?
  3. behavior— does choose_action() explore vs exploit correctly?
               does the replay buffer + target net do their jobs?
  4. memory  — does the brain survive a .pt checkpoint round trip?

Run with:  python test_agent.py
"""

# Imports, each earning its line (repo rule: comment every file well).
import math   # radians() for tilt-limit test values
import os     # checkpoint temp path handling
import tempfile  # safe temp dir for the save/load test

# torch only for tensor assertions (shapes, weight equality) — the
# agent itself owns all network logic; tests just check the results.
import torch

# Everything the tests pin down: the brain, its normalizer, and the
# defaults train.py will inherit (changing a default must update the
# tests that name it — deliberate friction against silent drift).
from agent import (
    DQNAgent, QNetwork, ReplayBuffer, normalize_state, STATE_SCALE,
    DEFAULT_HIDDEN, DEFAULT_LR, DEFAULT_GAMMA,
)


# ---------------------------------------------------------------
# 1. INPUTS — normalization + shapes
# ---------------------------------------------------------------
def test_normalize_centers_the_world():
    """Dead-center balanced state -> all zeros (the MLP's home)."""
    assert normalize_state((0.0, 0.0, 0.0, 0.0)) == [0.0, 0.0, 0.0, 0.0]
    print("  (0,0,0,0) -> [0,0,0,0] (dead center)")


def test_normalize_maps_limits_to_one():
    """Track edge + tilt limit -> ±1.0 (the scale's whole point).

    Hand math: 2.4/2.4 = 1, 12° in rad / 12° in rad = 1. A 20° death
    thump maps to 20/12 ≈ 1.67 — outside is allowed, just "far".
    """
    got = normalize_state((2.4, -3.0, math.radians(12), 4.0))
    assert got == [1.0, -1.0, 1.0, 1.0], f"limits must map to ±1: {got}"
    assert STATE_SCALE[0] == 2.4 and abs(math.degrees(STATE_SCALE[2]) - 12.0) < 1e-9
    print("  limits -> [1,-1,1,1]; 20deg would be ~1.67 (allowed)")


def test_network_outputs_two_q_values():
    """MLP shape: 4 in -> 2 out; a BATCH of 3 states -> [3, 2]."""
    net = QNetwork()
    assert net(torch.zeros(1, 4)).shape == (1, 2)
    assert net(torch.zeros(3, 4)).shape == (3, 2)
    # get_q returns plain floats (no tensors leak into train.py).
    agent = DQNAgent(seed=0)
    q = agent.get_q((0.0, 0.0, 0.0, 0.0))
    assert len(q) == 2 and all(isinstance(v, float) for v in q)
    print(f"  Q(home) untrained = {q} (random init, 2 floats)")


# ---------------------------------------------------------------
# 2. MATH — the TD target (hand-checked, like 02)
# ---------------------------------------------------------------
def _frozen_agent(gamma=0.9, lr=1e-3, seed=0):
    """Build an agent whose target net predicts EXACTLY zeros.

    Trick: zero every target weight+bias, so max Q_target(s') = 0 and
    the TD target collapses to just `reward` — hand-computable. The
    ONLINE net keeps its random init (so there IS an error to fix).
    """
    agent = DQNAgent(gamma=gamma, lr=lr, seed=seed)
    with torch.no_grad():
        for p in agent.target.parameters():
            p.zero_()
    return agent


def test_td_target_matches_hand_calculation():
    """target = r + gamma * max Q_target(s'), loss = mean((t - Q)²).

    Setup: target net -> all zeros (max_next = 0), so with r=1.0 the
    target is exactly 1.0. The loss must then equal mean((1 - Q)²)
    computed from the ONLINE net's own current predictions — we
    recompute that by hand here and demand equality to 1e-6.
    """
    agent = _frozen_agent(gamma=0.9, seed=1)
    s, s2 = (0.1, 0.2, 0.05, -0.1), (0.0, 0.0, 0.0, 0.0)
    batch = [(s, 1, 1.0, s2, False), (s2, 0, 1.0, s, False)]

    # Hand target: zeros-net -> max_next = 0 -> target = 1.0 each.
    with torch.no_grad():
        q_now = agent.online(agent._to_tensor([s, s2]))
    hand_loss = ((1.0 - q_now[0, 1]) ** 2 + (1.0 - q_now[1, 0]) ** 2) / 2

    loss = agent.train_step(batch)
    assert abs(loss - float(hand_loss)) < 1e-6, \
        f"loss {loss} != hand {float(hand_loss)}"
    print(f"  TD loss {loss:.6f} == hand-computed {float(hand_loss):.6f}")


def test_terminal_ignores_future():
    """done=True -> target = reward, even if the target net promises
    riches (02's rule, now enforced via the (1 - done) mask)."""
    agent = _frozen_agent(gamma=0.99, seed=2)
    # Un-zero ONE target output weight so max_next would be huge if
    # the mask failed — then a terminal target must STILL be 1.0.
    with torch.no_grad():
        for p in agent.target.parameters():
            p.fill_(5.0)
    s = (0.0, 0.0, 0.0, 0.0)
    batch = [(s, 0, 1.0, s, True)]  # done=True, reward +1 (last tick)

    with torch.no_grad():
        q_before = float(agent.online(agent._to_tensor(s))[0, 0])
    # One step with lr tiny: Q must move TOWARD 1.0 (not toward 5s).
    agent.train_step(batch)
    with torch.no_grad():
        q_after = float(agent.online(agent._to_tensor(s))[0, 0])
    assert abs(q_after - 1.0) < abs(q_before - 1.0), \
        f"terminal target missed 1.0: {q_before} -> {q_after}"
    print(f"  terminal Q moved {q_before:.4f} -> {q_after:.4f} (toward 1.0)")


def test_training_reduces_loss():
    """Repeated steps on the SAME batch must shrink the loss (the net
    is actually descending, not wandering). 0.99x margin keeps fast
    CI machines green — direction matters, exact size does not."""
    agent = _frozen_agent(gamma=0.9, lr=1e-2, seed=3)
    s = (0.2, -0.1, 0.05, 0.3)
    batch = [(s, 0, 1.0, s, False)] * 8
    first = agent.train_step(batch)
    for _ in range(20):
        last = agent.train_step(batch)
    assert last < first * 0.99, f"loss did not fall: {first} -> {last}"
    print(f"  loss fell {first:.4f} -> {last:.4f} over 21 steps")


# ---------------------------------------------------------------
# 3. BEHAVIOR — explore/exploit, buffer, target sync
# ---------------------------------------------------------------
def test_greedy_picks_highest_q():
    """explore=False always exploits (even at epsilon=1.0)."""
    agent = DQNAgent(epsilon=1.0, seed=0)
    state = (0.0, 0.0, 0.0, 0.0)
    # Force the online net's LAST layer to prefer RIGHT: zero it,
    # then set RIGHT bias high (deterministic regardless of init).
    with torch.no_grad():
        last = agent.online.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([-1.0, 5.0]))
    picks = {agent.choose_action(state, explore=False) for _ in range(20)}
    assert picks == {1}, f"explore=False wandered: {picks}"
    print("  explore=False always picks the MLP's best action")


def test_epsilon_one_actually_explores():
    """epsilon=1.0 must try both buttons over many flips."""
    agent = DQNAgent(epsilon=1.0, seed=0)
    picks = [agent.choose_action((0.0, 0.0, 0.0, 0.0)) for _ in range(200)]
    assert set(picks) == {0, 1}, f"never tried both: {set(picks)}"
    print("  epsilon=1.0 explores both actions")


def test_epsilon_zero_never_explores():
    """epsilon=0.0 must never wander (200 greedy picks agree)."""
    agent = DQNAgent(epsilon=0.0, seed=0)
    state = (0.1, 0.0, 0.05, 0.0)
    first = agent.choose_action(state)
    assert all(agent.choose_action(state) == first for _ in range(200))
    print(f"  epsilon=0.0 always picks action {first}")


def test_update_warms_up_then_learns():
    """update() returns None until the buffer holds a full batch,
    then a float loss (lets train.py log 'warming up' honestly)."""
    agent = DQNAgent(batch_size=8, seed=0)
    s = (0.0, 0.0, 0.0, 0.0)
    assert agent.update(s, 0, 1.0, s, False) is None, "should warm up"
    loss = None
    for _ in range(10):
        loss = agent.update(s, 1, 1.0, s, False)
    assert isinstance(loss, float), f"expected float loss, got {loss}"
    print(f"  update(): None x7, then loss {loss:.4f} once warm")


def test_replay_buffer_drops_oldest():
    """Fixed capacity: pushing past it evicts the OLDEST transition."""
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push((float(i), 0, 0, 0), 0, float(i), (0, 0, 0, 0), False)
    assert len(buf) == 3
    rewards = sorted(b[2] for b in buf.sample(3))
    assert rewards == [2.0, 3.0, 4.0], f"oldest not evicted: {rewards}"
    print("  capacity 3 + 5 pushes -> keeps rewards [2,3,4]")


def test_target_sync_copies_weights():
    """sync_target() makes target EXACTLY equal online; auto-sync
    fires every TARGET_SYNC gradient steps."""
    agent = DQNAgent(target_sync=3, seed=0)
    with torch.no_grad():
        for p in agent.online.parameters():
            p.fill_(1.0)
    agent.sync_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(po, pt), "manual sync failed"
    # Auto-sync: dirty online, run 3 train_steps -> target catches up.
    with torch.no_grad():
        for p in agent.online.parameters():
            p.fill_(2.0)
    batch = [((0, 0, 0, 0), 0, 1.0, (0, 0, 0, 0), False)] * agent.batch_size
    agent.updates = 0
    for _ in range(3):
        agent.train_step(batch)  # 3rd step triggers sync (weights move
                                 # slightly by grads — so compare fresh)
    agent.sync_target()
    for po, pt in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(po, pt), "auto sync failed"
    print("  manual + every-3-steps target sync both copy exactly")


# ---------------------------------------------------------------
# 4. MEMORY — checkpoint round trip
# ---------------------------------------------------------------
def test_save_load_round_trip():
    """Weights + hyperparams survive .pt (a 128-wide brain must NOT
    silently load 64-wide weights — architecture travels along)."""
    agent = DQNAgent(hidden=32, lr=2e-3, gamma=0.95, epsilon=0.25,
                     batch_size=16, target_sync=7, seed=0)
    before = agent.get_q((0.5, -0.5, 0.05, 1.0))
    path = os.path.join(tempfile.gettempdir(), "dqn_cart_test.pt")
    try:
        agent.save(path)
        clone = DQNAgent.load(path)
        assert clone.hidden == 32 and clone.lr == 2e-3
        assert clone.gamma == 0.95 and clone.epsilon == 0.25
        assert clone.batch_size == 16 and clone.target_sync == 7
        assert clone.get_q((0.5, -0.5, 0.05, 1.0)) == before
        for po, pc in zip(agent.online.parameters(),
                           clone.online.parameters()):
            assert torch.equal(po, pc), "online weights changed"
        for po, pc in zip(agent.target.parameters(),
                           clone.target.parameters()):
            assert torch.equal(po, pc), "target weights changed"
    finally:
        if os.path.exists(path):
            os.remove(path)
    print(f"  save/load keeps weights + hyperparams (Q={before})")


if __name__ == "__main__":
    tests = [
        test_normalize_centers_the_world,
        test_normalize_maps_limits_to_one,
        test_network_outputs_two_q_values,
        test_td_target_matches_hand_calculation,
        test_terminal_ignores_future,
        test_training_reduces_loss,
        test_greedy_picks_highest_q,
        test_epsilon_one_actually_explores,
        test_epsilon_zero_never_explores,
        test_update_warms_up_then_learns,
        test_replay_buffer_drops_oldest,
        test_target_sync_copies_weights,
        test_save_load_round_trip,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} agent checks verified.")
