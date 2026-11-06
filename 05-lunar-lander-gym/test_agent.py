"""
test_agent.py — THE BRAIN'S CONTRACT, pinned by tests
======================================================

The actor-critic is small on purpose, so its tests can be small
and still land hard:
    - shapes (8 -> 128 -> 4/1) and the wrapper's role split
    - act() samples AND remembers; act_greedy is deterministic,
      memory-clean and consumes NO dice (stream-freeze pin)
    - calculate_loss re-derived ONCE here with the same memory
      tensors, by hand formula — the wiring and the backward
      (future-first) return order both pin this way
    - the normalization's documented NO-GUARD quirk: constant
      rewards -> std 0 -> NaN (kept slavish from the reference;
      the test says exactly what will happen)
    - save/load roundtrip; seeded twins replay identically

    python test_agent.py
"""
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import ActorCritic, DEFAULT_HIDDEN, DEFAULT_GAMMA
from environment import OBS_DIM, N_ACTIONS


def test_net_shapes():
    """affine 8->128; two heads 128->4 and 128->1; default dials."""
    brain = ActorCritic(seed=0)
    assert brain.affine.in_features == OBS_DIM
    assert brain.affine.out_features == DEFAULT_HIDDEN == 128
    assert brain.action_layer.in_features == DEFAULT_HIDDEN
    assert brain.action_layer.out_features == N_ACTIONS
    assert brain.value_layer.out_features == 1
    assert brain.gamma == DEFAULT_GAMMA == 0.99
    print("  net: 8 -> 128 -> (4 probs, 1 value); gamma 0.99")


def test_act_appends_memory():
    """act() = sample + remember: one logprob and one value per
    call; rewards are the CALLER'S list (train.py files them)."""
    brain = ActorCritic(seed=0)
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0, 0)
    for _ in range(5):
        action = brain.act(state)
        assert isinstance(action, int) and 0 <= action < N_ACTIONS
    assert len(brain.logprobs) == 5 == len(brain.state_values)
    assert len(brain.rewards) == 0, "rewards wait for the caller"
    brain.rewards.extend([0.0] * 5)
    brain.clear_memory()
    assert len(brain.logprobs) == 0 == len(brain.state_values) \
        == len(brain.rewards)
    print("  act(): sampled ints, memory per tick, caller files "
          "rewards, clear empties all three")


def test_act_greedy_deterministic_and_clean():
    """act_greedy: same state -> same button, NO memory write, and
    NO dice consumed the torch stream goes wherever it would have.

    The pin: ONE brain does 30 greedy reads TO THE DICE-STREAM'S
    LEFT (before its samples), a twin does the same 30 AFTER its
    samples; both 4-sample windows and both 5th samples must match
    (torch's multinomial burns a varying number of uniforms per
    call, so interleaved-position comparisons are meaningless
    [learned the hard way]; independent stream starts are the
    only honest twin).
    """
    brain = ActorCritic(seed=7)
    state = (0.1, 0.8, 0.0, -0.05, 0.05, 0.01, 0, 0)
    picks = {brain.act_greedy(state) for _ in range(30)}
    assert len(picks) == 1, f"greedy must be a function: {picks}"
    assert len(brain.logprobs) == 0, "greedy pollutes nothing"

    # dice reads on the LEFT of the samples:
    four_left = [brain.act(state) for _ in range(4)]
    fifth_left = brain.act(state)

    # twin: samples first, THEN the 30 greedy reads (right side):
    twin = ActorCritic(seed=7)
    four_right = [twin.act(state) for _ in range(4)]
    fifth_right = twin.act(state)
    assert four_left == four_right, "sampling must be reproducible"
    assert fifth_left == fifth_right, \
        "greedy reads must not consume the dice stream"
    brain.clear_memory(); twin.clear_memory()
    print("  act_greedy: deterministic, memory-clean, dice "
          "untouched (positions LEFT vs RIGHT of the samples)")


def test_calculate_loss_by_hand():
    """The loss, re-derived OUTSIDE the net with the SAME memory
    tensors. This pins:
        - the backward (future-first) discounted-return ORDER
        - the zero-mean/unit-std normalization arithmetic
        - the detached-advantage policy gradient and the
          smooth_l1 value head, summed over the episode
    A forward-order (bug) cumulative would flip the first tick's
    advantage sign and FAIL this equality.
    """
    gamma = 0.99
    brain = ActorCritic(gamma=gamma, seed=3)
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0, 0)
    rewards = [1.0, -0.5, -0.5, -0.5, 100.0]
    for r in rewards:
        brain.act(state)            # fills logprobs + state_values
        brain.rewards.append(r)

    # --- the net's loss -------------------------------------------
    loss = brain.calculate_loss(gamma)

    # --- the by-hand replica (THE reference's formula) -------------
    discounted = []
    running = 0.0
    for r in rewards[::-1]:                 # FUTURE first
        running = r + gamma * running
        discounted.insert(0, running)
    disc = torch.tensor(discounted)
    disc = (disc - disc.mean()) / disc.std()   # no guard: faithful
    import torch.nn.functional as F
    hand = 0.0
    for logprob, value, reward in zip(brain.logprobs,
                                      brain.state_values, disc):
        advantage = reward - value.item()   # DETACHED baseline
        hand = hand + (-logprob * advantage
                       + F.smooth_l1_loss(value, reward.view(1)))

    brain.clear_memory()
    assert abs(loss.item() - hand.item()) < 1e-5, \
        f"net {loss.item():.9f} vs hand {hand.item():.9f}"
    print(f"  loss by hand: net == replica ({loss.item():+.6f}); "
          "future-first order + normalization pinned")


def test_normalization_no_guard_quirk():
    """THE documented slavish quirk: division by rewards.std() with
    no epsilon addend. An ALL-ZERO-REWARD episode has every
    discounted return exactly 0.0, so std is exactly 0 and the
    loss is NaN. (Constant NONZERO rewards escape via backward
    discounting: R_t = r + gamma*R_{t+1}, so the R's drift apart
    tick by tick - zero is the only constant series.) This is the
    reference's behavior, kept on purpose (file-top comment
    story). If this test starts failing because someone ADDED a
    guard - update BOTH places consciously."""
    brain = ActorCritic(gamma=0.5, seed=0)
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0, 0)
    for _ in range(4):
        brain.act(state)
        brain.rewards.append(0.0)     # an all-zero episode
    loss = brain.calculate_loss(0.5)
    brain.clear_memory()
    assert math.isnan(loss.item()), \
        f"no-guard division must NaN out: {loss.item()}"
    print("  no-guard quirk: all-zero-reward episode -> NaN loss "
          "(documented, faithful)")


def test_seed_twins_replay():
    """Same seed -> same init AND same sampled stream - measured
    straight after EACH net's construction (the construction
    itself does manual_seed, so the stream is freshly reset per
    net; the torch stream is ONE global generator, so any act in
    between shifts positions [the multiround lesson] - sampling
    each brain first thing is the only honest twin)."""
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0, 0)

    a = ActorCritic(seed=543)
    seq_a = [a.act(state) for _ in range(20)]
    a.clear_memory()

    b = ActorCritic(seed=543)   # the fresh reset, inside init
    seq_b = [b.act(state) for _ in range(20)]
    b.clear_memory()
    assert seq_a == seq_b

    c = ActorCritic(seed=777)
    seq_c = [c.act(state) for _ in range(20)]
    c.clear_memory()
    assert seq_a != seq_c
    print("  twin replay: same seed, same 20 samples")


def test_save_load_roundtrip():
    """save/load: the weights AND the dials survive, greedy picks
    identical post-load."""
    brain = ActorCritic(seed=42)
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0, 0)
    for _ in range(7):
        brain.act(state)
        brain.rewards.append(-0.3)
    brain.clear_memory()
    greedy_before = [brain.act_greedy(state) for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "brain.pt")
        brain.save(path)
        clone = ActorCritic.load(path)
        assert clone.hidden == brain.hidden == 128
        assert clone.gamma == brain.gamma
        assert clone.n_inputs == OBS_DIM and clone.n_actions == N_ACTIONS
        greedy_after = [clone.act_greedy(state) for _ in range(4)]
        assert greedy_before == greedy_after, "weights drifted?"
    print("  save/load: dials + weights identical, greedy agrees")


if __name__ == "__main__":
    test_net_shapes()
    test_act_appends_memory()
    test_act_greedy_deterministic_and_clean()
    test_calculate_loss_by_hand()
    test_normalization_no_guard_quirk()
    test_seed_twins_replay()
    test_save_load_roundtrip()
    print()
    print("All 7 brain checks verified.")
