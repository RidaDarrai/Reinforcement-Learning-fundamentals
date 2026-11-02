"""
test_agent.py — is the brain honest?
====================================

The environment tests pinned the WORLD by hand; this file pins the
LEARNING RULE the same way:

- the TD loss is recomputed BY HAND from the nets' own outputs
  (test_td_loss_hand_recompute) — target = r + gamma * max Q(s'),
  loss = (target - Q(s,a))^2, and the code must agree;
- the done-mask is proven BEHAVIORALLY: hammer one bootstrapped
  transition a few hundred times and Q(s,a) must settle at the
  reward, not run away to 100/(1-gamma) (test_done_mask_pulls_q_home);
- one gradient step must move Q TOWARD its target, whichever way
  the gap points (test_q_moves_toward_target);
- both stabilizers get a dedicated check: the frozen target only
  exists if it lags while training and re-syncs when told to.

Also here: the contract with the world's scales (agent normalizes
using environment's own normalize_state, no private copy), the
4-button policy (epsilon-greedy + lowest-index tie-break), warm-up
feeding, FIFO buffer eviction, and the save/load round-trip —
including the corrupted-file failure path (03 earned that test,
and it keeps earning).

Run with:  python test_agent.py     (needs torch — see ../requirements.txt)
"""

# Imports, each earning its line (repo rule: comment every file well).
import math        # isfinite on Q outputs, isclose on hand-checked math
import os          # build paths into the temp dir for save()/load()
import tempfile    # throwaway directory for checkpoint round-trip tests

# torch only for direct tensor forensics inside a few tests — the
# agent hides torch everywhere else.
import torch

# The public contract of the brain, imported in one place so a
# rename in agent.py breaks loudly here instead of silently.
from agent import (
    DQNAgent, QNetwork, ReplayBuffer,
    DEFAULT_N_INPUTS, DEFAULT_N_ACTIONS, DEFAULT_HIDDEN,
    DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_SIZE,
)
from agent import normalize_state as agent_normalize
from environment import normalize_state as world_normalize

# A known pose to probe repeatedly (spawn altitude, drifting).
STOCK_STATE = (0.0, 70.0, 0.5, -1.0, 0.05, 0.0, 0, 0)


# ---------------------------------------------------------------
# The network itself
# ---------------------------------------------------------------
def test_brain_shapes():
    """MLP 8 -> 64 -> 64 -> 4: the world's 8-float contract in, one
    Q per button out, batched tensors keep their leading axis."""
    net = QNetwork()
    assert net.net[0].in_features == 8, "8 state floats in"
    assert net.net[0].out_features == 64, "hidden width 64"
    assert net.net[4].out_features == 4, "one Q per button out"

    x = torch.zeros(3, 8)                 # a batch of three states
    assert net(x).shape == (3, 4), "batching must preserve rows"
    print(f"  MLP({'8 -> 64 -> 64 -> 4'}): shapes verified")


def test_normalization_single_source():
    """The brain must NOT keep a private copy of the state scales —
    it borrows environment.normalize_state, so the world stays the
    single source of truth for the state contract."""
    assert agent_normalize is world_normalize, \
        "agent.py must import the world's normalizer, not copy it"
    assert DEFAULT_N_INPUTS == 8 == len(STOCK_STATE)
    assert DEFAULT_N_ACTIONS == 4
    print("  normalization: borrowed from environment.py (one source)")


def test_seed_reproducibility():
    """Twins with the same seed init to identical weights and answer
    identically; a different seed diverges somewhere."""
    a, b = DQNAgent(seed=0), DQNAgent(seed=0)
    assert a.get_q(STOCK_STATE) == b.get_q(STOCK_STATE), \
        "same seed -> identical Q-value vector"

    c = DQNAgent(seed=1)
    differing = any(
        qc != a.get_q(STOCK_STATE)[i] for i, qc in enumerate(c.get_q(STOCK_STATE))
    )
    assert differing, "a different seed must shuffle the weights"
    print("  seeded twins agree; seed 1 disagrees somewhere")


def test_get_q_contract():
    """get_q returns 4 finite plain floats, for every pose — even
    absurd ones far outside the normalized range (no crash, no NaN)."""
    agent = DQNAgent(seed=0)
    for state in (
        STOCK_STATE,
        (0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0, 0),
        (-50.0, 80.0, 6.0, 8.0, math.pi, 4.0, 1, 1),   # the limits
        (0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1, 1),          # touching down
    ):
        q = agent.get_q(state)
        assert isinstance(q, list) and len(q) == 4
        assert all(isinstance(v, float) for v in q)
        assert all(math.isfinite(v) for v in q)
    print("  get_q: 4 finite floats at the spawn, limits and pad")


# ---------------------------------------------------------------
# The policy — four buttons, epsilon-greedy
# ---------------------------------------------------------------
def test_exploit_policy_and_ties():
    """Exploitation picks the argmax; ties fall to the LOWEST index
    (the 02/03 rule, now over four buttons) — deterministic either
    way, with or without the exploration coin."""
    agent = DQNAgent(seed=0, epsilon=0.0)
    agent.get_q = lambda s: [1.0, 1.0, 0.5, 0.5]     # a stub: two-way tie
    assert agent.choose_action(STOCK_STATE) == 0
    assert agent.choose_action(STOCK_STATE, explore=False) == 0

    agent.get_q = lambda s: [0.1, 2.0, 5.0, 3.0]
    assert agent.choose_action(STOCK_STATE) == 2
    assert agent.choose_action(STOCK_STATE, explore=False) == 2
    print("  argmax + lowest-index tie-break, explore flag respected")


def test_epsilon_zero_never_explores():
    """epsilon=0.0 -> the coin never flips: 100 decisions must ALL be
    the net's argmax for that state."""
    agent = DQNAgent(seed=0, epsilon=0.0)
    expected = max(range(4), key=lambda i: agent.get_q(STOCK_STATE)[i])
    picks = {agent.choose_action(STOCK_STATE) for _ in range(100)}
    assert picks == {expected}, f"epsilon-0 must always exploit: {picks}"
    print(f"  epsilon 0.0: 100 picks, all argmax (button {expected})")


def test_epsilon_one_explores_wildly():
    """epsilon=1.0 -> every press is a coin flip: 200 picks MUST
    visit all four buttons (probability of missing one ~ 4e-23)."""
    agent = DQNAgent(seed=0, epsilon=1.0)
    picks = {agent.choose_action(STOCK_STATE) for _ in range(200)}
    assert picks == {0, 1, 2, 3}, f"exploration must cover all: {picks}"
    print("  epsilon 1.0: 200 picks cover IDLE/MAIN/LEFT/RIGHT")


# ---------------------------------------------------------------
# The replay diary
# ---------------------------------------------------------------
def test_buffer_fifo_and_unique_sampling():
    """The diary holds the N MOST RECENT transitions (oldest evicted
    first), and a random sample never double-draws inside a batch."""
    buf = ReplayBuffer(capacity=10)
    for i in range(12):
        buf.push((float(i),), i, float(i), (float(i),), False)
    assert len(buf) == 10, "capacity caps the diary"
    # The two oldest (i=0,1) must be GONE, the third (i=2) must remain.
    assert buf.buffer[0][1] == 2, "FIFO must evict oldest first"
    assert buf.buffer[-1][1] == 11, "newest must survive"

    batch = buf.sample(5)
    actions = [b[1] for b in batch]
    assert len(batch) == 5 and len(set(actions)) == 5, \
        f"sampling without replacement broke: {actions}"
    print("  buffer: FIFO eviction at capacity, samples stay unique")


def test_update_warms_up():
    """update() files experiences and stays SILENT (returns None)
    until the diary holds a full batch — then every call trains and
    reports a positive, finite loss."""
    agent = DQNAgent(batch_size=8, seed=0)
    s, s2 = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0, 0), \
            (1.0, 69.0, 0.1, -0.5, 0.0, 0.0, 0, 0)
    losses = [agent.update(s, i % 4, 1.0, s2, False) for i in range(8)]

    assert losses[:7] == [None] * 7, "7 ticks must still be warming up"
    assert len(agent.buffer) == 8
    assert isinstance(losses[7], float)
    assert math.isfinite(losses[7]) and losses[7] > 0.0
    print(f"  warm-up: 7x None then loss={losses[7]:.4f} on tick 8")


# ---------------------------------------------------------------
# THE BIG ONES — the learning rule, hand-checked
# ---------------------------------------------------------------
def test_td_loss_hand_recompute():
    """Recompute one gradient step's loss ON PAPER.

    Setup: a batch of ONE looping transition (s' = s, r = +1,
    alive). Then, by hand, from the nets' own current outputs:

        Q(s,a)    = online(s)[0]                  (captured BEFORE
        max Q(s') = target(s).max()               the step touches
        target    = 1 + 0.99 * max Q(s')          the weights)
        loss      = (target - Q(s,a))^2

    The returned loss must equal this to 1e-6. If the formula, the
    gather() (column of the taken action), or the frozen-target
    split drift, this check screams.
    """
    agent = DQNAgent(batch_size=4, target_sync=10 ** 9, seed=0)
    x = torch.tensor([agent_normalize(STOCK_STATE)], dtype=torch.float32)
    with torch.no_grad():
        q_before = float(agent.online(x)[0][0])       # action 0 column
        max_next = float(agent.target(x).max())       # frozen net

    loss = agent.train_step([(STOCK_STATE, 0, 1.0, STOCK_STATE, False)])

    target = 1.0 + agent.gamma * max_next
    manual = (target - q_before) ** 2
    assert math.isclose(loss, manual, abs_tol=1e-6), \
        f"returned {loss} vs hand-recomputed {manual}"
    print(f"  TD loss hand-recomputed: {loss:.6f} == (target - Q)^2")


def test_done_mask_pulls_q_home():
    """THE behavioral proof of the (1 - done) target mask: hammer
    one terminal transition until Q forgets its random guess —
    it must settle at the reward itself, NOT chase the bootstrapped
    100/(1-gamma) blow-up that a broken mask produces."""
    agent = DQNAgent(batch_size=64, seed=0)
    batch = [(STOCK_STATE, 0, 100.0, STOCK_STATE, True)] * 64
    for _ in range(300):
        agent.train_step(batch)
    q = agent.get_q(STOCK_STATE)[0]
    assert abs(q - 100.0) < 5.0, \
        f"terminal TD must anchor at +100, got {q:.2f}"
    print(f"  done-mask: 300 terminal hits anchored Q(s,*) at {q:.2f}")


def test_q_moves_toward_target():
    """One gradient step moves Q TOWARD the target — positive gaps
    push Q up, negative gaps push it down. Sign, not magnitude."""
    agent = DQNAgent(batch_size=64, seed=0)
    q_before = agent.get_q(STOCK_STATE)[0]

    # A big positive reward with s'=s: target = r + gamma*maxQ >> Q
    batch_up = [(STOCK_STATE, 0, 100.0, STOCK_STATE, False)] * 64
    for _ in range(5):
        agent.train_step(batch_up)
    assert agent.get_q(STOCK_STATE)[0] > q_before + 1e-5, \
        "Q must climb toward r + gamma*maxQ(s')"

    agent2 = DQNAgent(batch_size=64, seed=0)
    q2_before = agent2.get_q(STOCK_STATE)[0]
    # Terminal -100: target exactly -100 << the random Q guess
    batch_down = [(STOCK_STATE, 0, -100.0, STOCK_STATE, True)] * 64
    for _ in range(5):
        agent2.train_step(batch_down)
    assert agent2.get_q(STOCK_STATE)[0] < q2_before - 1e-5, \
        "Q must sink toward the -100 bill"
    print("  descent direction: up for profit, down for the bill")


def test_target_sync_both_ways():
    """Two stabilizer behaviors in one: with target_sync=1 every
    gradient step re-freezes the aim (target == online after each
    step); with the DEFAULT 500 the target must LAG (they differ
    after a few steps on live weights)."""
    eager = DQNAgent(batch_size=64, target_sync=1, seed=0)
    batch = [(STOCK_STATE, 0, 1.0, STOCK_STATE, False)] * 64
    for _ in range(3):
        eager.train_step(batch)
    for p_online, p_target in zip(eager.online.parameters(),
                                  eager.target.parameters()):
        assert torch.equal(p_online, p_target), \
            "target_sync=1 must re-sync after every step"

    lazy = DQNAgent(batch_size=64, seed=0)      # default sync: 500
    for _ in range(3):
        lazy.train_step(batch)
    assert any(not torch.equal(po, pt)
               for po, pt in zip(lazy.online.parameters(),
                                 lazy.target.parameters())), \
        "default 500 must leave the target frozen BEHIND the online"
    print("  target net: re-syncs at sync=1, lags frozen at default")


# ---------------------------------------------------------------
# Memory outside the brain — checkpoints and their failure paths
# ---------------------------------------------------------------
def test_save_load_roundtrip():
    """save() -> load() must rebuild the SAME brain: identical Q
    outputs, identical dials (epsilon, batch), identical diploma
    counter (updates) — play.py's TRAINED mode depends on this."""
    agent = DQNAgent(batch_size=4, seed=0)   # small batch: 7 files = 4
    for _ in range(7):                    # give it a history to carry
        agent.update(STOCK_STATE, 1, 1.0, STOCK_STATE, False)
    assert agent.updates == 4, "7 files at batch 4 -> 4 gradient steps"
    q_before = agent.get_q(STOCK_STATE)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lunar_brain.pt")
        agent.save(path)
        loaded = DQNAgent.load(path)

    loaded_q = loaded.get_q(STOCK_STATE)
    for v_a, v_b in zip(q_before, loaded_q):
        assert math.isclose(v_a, v_b, abs_tol=1e-9), \
            "loaded brain must answer identically"
    assert loaded.epsilon == agent.epsilon
    assert loaded.batch_size == agent.batch_size
    assert loaded.hidden == agent.hidden
    assert loaded.updates == agent.updates, "diploma counter travels"
    print(f"  save/load: Q identical, dials restored, "
          f"updates={loaded.updates} preserved")


def test_load_rejects_garbage():
    """A corrupted .pt must fail LOUDLY at load time — the crash
    03 caught in the window loader before any user ever saw it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "garbage.pt")
        with open(path, "wb") as f:
            f.write(b"\x00de" * 10 + b"not a checkpoint")
        try:
            DQNAgent.load(path)
        except Exception as e:            # torch raises variant errors
            assert str(e) != "", "the error should be descriptive"
        else:
            raise AssertionError("garbage bytes must not load as a brain")
    print("  corrupted checkpoint: rejected with a clear error")


def test_load_missing_file():
    """A missing .pt (never trained) must also fail loudly — the
    play.py 04 window will use this to offer a quick course, like
    03's 300-episode fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "never_existed.pt")
        try:
            DQNAgent.load(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            raise AssertionError(
                f"expected FileNotFoundError, got {type(e).__name__}: {e}")
        else:
            raise AssertionError("missing file must not load")
    print("  missing checkpoint: FileNotFoundError, loud and clear")


if __name__ == "__main__":
    tests = [
        test_brain_shapes,
        test_normalization_single_source,
        test_seed_reproducibility,
        test_get_q_contract,
        test_exploit_policy_and_ties,
        test_epsilon_zero_never_explores,
        test_epsilon_one_explores_wildly,
        test_buffer_fifo_and_unique_sampling,
        test_update_warms_up,
        test_td_loss_hand_recompute,
        test_done_mask_pulls_q_home,
        test_q_moves_toward_target,
        test_target_sync_both_ways,
        test_save_load_roundtrip,
        test_load_rejects_garbage,
        test_load_missing_file,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} brain checks verified.")
