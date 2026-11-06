"""
agent.py — THE BRAIN (step 2 of the lunar-lander-gym project)
=============================================================

This project's whole experiment is the BRAIN. Project 04 tried to
land the same moon with a DQN (value learning: "how good is each
button in this state?") and, after six runs and two retunes, died
from a value-fiction: four near-tied Q-values ~800 (reality pays
-100..-260), argmax decided by fitting noise, and a greedy policy
blasting straight up. The full 04 postmortem lives over there; its
one-line stem: when the score is REACH-and-GENTLE, a value net
chasing discounted bootstraps turned delusional POLICY, fine in its
training diary and lawless in greedy play.

So step 2 learns the other classical family: a POLICY network
(reference: github.com/nikhilbarhate99/Actor-Critic-PyTorch, a
vanilla episodic actor-critic on gym's LunarLander-v2). The loss is
written the other way around:

    DQN  predicts Q(s, a)         and the policy is ARGMAX of it
    A2C  LEARNS the probabilities -- pi(a | s), a stuck on the net
         -- and the gradient says directly: "actions that ended
         better than my value predicted (advantage > 0) get their
         probability RAISED; worse ones lowered."

    loss per step in the episode:
        advantage   = discounted_reward(s..end) - VALUE(s)
        action_loss = -log pi(a|s) * advantage      (policy gradient)
        value_loss  = smooth_l1(VALUE(s), discounted_reward)
                                                      (the critic)

THE ARCHITECTURE (gym's 8-float state, 4 buttons — same contract
the wrapper in environment.py pins):

    state (8,) -> Linear(8, 128) -> ReLU -> heads
              |                       |-> Linear(128, 4) -> softmax
              |                       |            = pi(a | s)
              |                       |-> Linear(128, 1)
              |                                    = VALUE(s)
    action head: SAMPLES an action (stochastic — exploration is
                 FREE: randomness IS the policy, no epsilon dial)
    value head : one number, the critic's guess of this state's
                 worth, used as the baseline (advantage = R - V)

"128" is not 04's 64x64 by mistake or vanity — it is the
reference's layer width, kept slavish: the reference test.py suite
runs this exact shape, and porting means locking hands with the
original dial-for-dial.

THE EPISODIC MEMORY (why there is no replay buffer, no target net)
------------------------------------------------------------------
A DQN files every tick and replays random batches through a frozen
target. An episodic actor-critic collects ONE WHOLE episode in
memory (this net keeps the log-probs, the state values, and the
rewards as three plain lists), then:

    1. computes the DISCOUNTED return R_t for each tick (backward
       pass over the episode: R_t = r_t + gamma * R_{t+1})
    2. NORMALIZES all R_t to zero mean / unit std (variance-killing
       scale-shift; the gradient step's steering wheel)
    3. takes ONE gradient step for the whole episode (policy
       gradients are noisy tick-by-tick; the full episode is the
       sample), then CLEARS the memory.

        logprobs     <- appended by forward() at act time
        state_values <- ditto
        rewards      <- appended by train.py after env.step()

All of that lives in calculate_loss() and clear_memory() below —
the reference's calculateLoss()/clearMemory(), renamed to snake_
case and otherwise hand-copied.

WHY RAW STATE (no normalize_state(), the docstring's second act):
    04's MLP ate SI meters and needed a private normalizer. Gym's
    state prints PRE-normalized x / y / speeds (environment.py's
    header) — so the preprocessing aisle stays empty this time.
    A second normalizer stacked on gym's would be the lesson
    inverted into a bug.

THE PIECES AND WHAT EACH IS FOR
    save(path)          copy the brain to disk (diploma / exam best)
    load(path) cls      rebuild from a saved brain (play.py)

API (the exact shape train.py and play.py already expect;
    compatible-in-shape with 02-04 where shape can be kept):

    action = agent.act(state)          # SAMPLES, remembers (training)
    agent.rewards.append(reward)       # then train.py files the tick
    loss    = agent.calculate_loss(gamma)   # END of episode, once
    agent.clear_memory()               # between episodes
    agent.act_greedy(state)            # exams + replay: no dice, no
                                       # memory pollution, no_grad

Try it (a tiny hands-on tour — one 5-tick imaginary episode):

    python agent.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import os          # join paths — save() writes next to agent.py

# PyTorch + torch.distributions.Categorical: the sampling head.
# Categorical turns 4 probabilities into a legally-sampled action
# index AND hands out log_prob(action) — the exact terms of the
# policy gradient. (02-04 had no need for it: argmax needs only
# torch, no distribution.)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# The state and action contracts live in the WORLD — imported, never
# retyped (one source of truth; 04's habit, kept).
from environment import OBS_DIM, N_ACTIONS

# Friendly defaults — the REFERENCE'S dials read through our knob
# style (train.py may shrink them; defaults ARE the official run's).
DEFAULT_HIDDEN = 128   # neurons in the (single) hidden layer — the
                       # reference's every-file number, not ours
DEFAULT_GAMMA   = 0.99 # discount inside calculate_loss (train.py
                       # passes it again — the knob is testable both
                       # places, like the reference's own signature)
DEFAULT_LR      = 0.02  # Adam step size — the reference's; 100x
                        # ours in 04! Policy gradients are one-noisy
                        # sum per episode (no minibatch averaging to
                        # calm them), they NEED a big step; and the
                        # warm betas help them not to overshoot.

# THE NORMALIZATION GUARD STORY — IMPORTANT, read before edit:
# the reference divides by rewards.std() with NO epsilon guard. A
# degenerate episode of a single tick (std undefined) or identical
# discounted rewards (std exactly 0) would feed back nan/inf loss.
# The port keeps the reference's raw division — slavishly — and the
# test suite PROVES the products on a real window: this is a
# documented no-guard, not an accident. If a real run ever dies of
# nan, the fix is one line, pre-tested, right here.
NO_GUARD = True        # (a named flag so no one thinks it is magic)


# ==============================================================
# 1. THE NETWORK — policy head + value head over one spine
# ==============================================================
class ActorCritic(nn.Module):
    """The policy AND its critic, one trunk, two heads.

    The reference's class, verb-for-verb, with our comments. The
    three memory lists live ON the net (episodes are short; they
    are cleared every episode by clear_memory()).
    """

    def __init__(self, n_inputs=OBS_DIM, n_actions=N_ACTIONS,
                 hidden=DEFAULT_HIDDEN, gamma=DEFAULT_GAMMA,
                 seed=None):
        """Build the trunk + heads (+ the empty episode memory).

        Args: n_inputs  — 8 floats, the wrapper's frozen contract.
              n_actions — 4 buttons (counted, never learned).
              hidden    — trunk width (the reference's 128).
              gamma     — discount used by calculate_loss (the
                          reference's signature passes it in, so
                          tests can pin the sliding scale).
              seed      — torch.manual_seed(seed) INSIDE the net's
                          construction, so weight init and sampling
                          replay identically (same as 02-04's
                          reproducibility habit; the reference
                          does this at the top of train.py — same
                          physics, one belt, worn on the net).
        """
        super().__init__()
        self.n_inputs, self.n_actions, self.hidden = \
            n_inputs, n_actions, hidden
        self.gamma = gamma

        if seed is not None:
            torch.manual_seed(seed)

        # The trunk + the two heads, names inherited from the
        # reference ("affine" = the shared Linear projection):
        self.affine        = nn.Linear(n_inputs, hidden)
        self.action_layer  = nn.Linear(hidden, n_actions)  # actor
        self.value_layer   = nn.Linear(hidden, 1)          # critic

        # THE EPISODIC MEMORY: three parallel lists, one entry per
        # tick of the CURRENT episode (the DQN's 100k replay diary
        # shrinks to three scalars per tick and empties nightly).
        self.logprobs     = []   # log pi(a_t | s_t), from forward()
        self.state_values = []   # VALUE(s_t), the critic's guess
        self.rewards      = []   # r_t, filed by train.py's loop

    # ----------------------------------------------------------
    # forward() — pick an action AND remember it, like breathing
    # ----------------------------------------------------------
    def forward(self, state):
        """Sample an action from the policy for this state.

        Appends (log_prob, state_value) to the episode memory —
        the policy gradient needs BOTH for the action it just took:
            - log_prob: "how much of my probability did I spend
              on this exact button" (the pi-gradient's handle)
            - state_value: the critic's guess, subtracted from the
              discounted return as the ADVANTAGE baseline.

        Returns the sampled action index (int, 0..3).

        TRAINING ONLY (it mutates the memory): exams and play.py's
        greedy replay go through act_greedy() instead.
        """
        state = torch.tensor(state, dtype=torch.float32)
        hidden   = F.relu(self.affine(state))
        state_value = self.value_layer(hidden)

        action_probs = F.softmax(self.action_layer(hidden), dim=-1)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    # two thin names for the two ways of choosing, so callers never
    # touch forward() directly (04's choose_action precedent):
    def act(self, state, explore=True):
        """Sample an action (training). explore=False is ASSERTED
        unused — a policy net does not have a "safe" sample mode."""
        if not explore:
            raise AssertionError(
                "actor-critic's act() is sampling-only; use "
                "act_greedy() for exams and play")
        return self.forward(state)

    # ----------------------------------------------------------
    # calculate_loss() — ONE gradient step's worth of teaching
    # ----------------------------------------------------------
    def calculate_loss(self, gamma=None):
        """The reference's calculateLoss(): the whole EPISODE's loss.

        1) discounted returns, computed BACKWARD (future first, so
           a tick's R borrows the NEXT tick's discounted sum):

               R_t = r_t + gamma * R_{t+1}

           (a landing at the last tick smears its +100 backward
           into every tick that led to it, fading by gamma^n)

        2) normalized: zero mean, unit std over the episode. The
           SCALE jars off (an episode's absolute return depends on
           the spawn's luck), the ORDER stays (0.99-shrank rank of
           which ticks led to the payoff) — the classic
           high-variance-killer of this family. NO epsilon guard:
           the reference's raw division, kept slavish (NO_GUARD
           flag; the comment at the file top tells the story).

        3) the per-tick loss pair, summed over the episode:

               advantage    = R_t - VALUE(s_t)      (.item() — the
                              value is DETACHED from the policy
                              gradient's path; a critic's miscal-
                              ibration dribbles into actions only
                              through the BASELINE, never the
                              gradient)
               action_loss  = -log_prob * advantage
               value_loss   = smooth_l1_loss(value, R_t)
                              (a SOFT square: Huber to not blow up
                              on the occasional fat R_t)

           total = sum of both, over all ticks.

        Returns the scalar loss TENSOR. The optimizer's dance
        (zero_grad -> backward -> step) is train.py's business:
        the reference splits it the same way, and keeping the split
        is what makes tests able to READ the loss before descent.
        """
        gamma = self.gamma if gamma is None else gamma

        # --- 1. discounted returns, future-first -----------------
        discounted = []
        running = 0.0
        for reward in self.rewards[::-1]:
            running = reward + gamma * running
            discounted.insert(0, running)

        # --- 2. normalize the episode's returns -------------------
        discounted = torch.tensor(discounted)
        discounted = (discounted - discounted.mean()) / discounted.std()

        # --- 3. the two-headed loss, summed -----------------------
        loss = 0.0
        for logprob, value, reward in zip(self.logprobs,
                                          self.state_values,
                                          discounted):
            advantage   = reward - value.item()
            action_loss = -logprob * advantage
            # reward is a 0-dim tensor out of `discounted`; view(1)
            # matches value's [1]-shape (the reference broadcast
            # silently — same numbers, modern torch just warns):
            value_loss  = F.smooth_l1_loss(value, reward.view(1))
            loss += (action_loss + value_loss)
        return loss

    def clear_memory(self):
        """Empty the three lists (between episodes; exams pollute
        nothing, because act_greedy never appends)."""
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    # ----------------------------------------------------------
    # act_greedy() — the exam mode (deterministic, memory-clean)
    # ----------------------------------------------------------
    def act_greedy(self, state):
        """The exam button: argmax of the policy's probabilities.

        no_grad (reading the policy must never build a gradient
        graph — 04's get_q habit) and NOTHING appended (an exam is
        a SERVICE the brain performs, not an experience it
        remembers; the next training episode must not find a
        phantom tick in its memory).
        """
        self.eval()   # eval mode (no dropout/batchnorm here, but
                      # the future-proof habit stays)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            hidden = F.relu(self.affine(state))
            probs  = F.softmax(self.action_layer(hidden), dim=-1)
            return int(torch.argmax(probs).item())

    # ----------------------------------------------------------
    # the diploma: save / load (04's contract, same file shape)
    # ----------------------------------------------------------
    def save(self, path):
        """Weights + the shape metadata, one torch.save away.

        (Explicit meta instead of pickling the WHOLE object: the
        file must load years later with a different __init__
        signature, with no surprises about module internals.)
        """
        data = {
            "n_inputs": self.n_inputs,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "gamma": self.gamma,
            "state_dict": self.state_dict(),
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path):
        """Rebuild a brain from save()'s file. Raises (loud, honest)
        if anything is missing — play.py catches what IT can
        degrade from, tests pin the happy path."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(n_inputs=data["n_inputs"],
                    n_actions=data["n_actions"],
                    hidden=data["hidden"],
                    gamma=data["gamma"])
        agent.load_state_dict(data["state_dict"])
        return agent


# ==============================================================
# DEMO — one imaginary 5-tick episode, every number by hand
# ==============================================================
if __name__ == "__main__":
    print("=== the brain, one imaginary episode ===")
    agent = ActorCritic(seed=0, gamma=0.99)
    state = (0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # a legal spawn

    rewards_given = [1.0, -0.5, -0.5, -0.5, 100.0]     # a landing!
    for r in rewards_given:
        action = agent.act(state)
        agent.rewards.append(r)
        state = (state[0] + 0.01, state[1] - 0.01,) + state[2:]  # drift

    print(f"greedy picks after   : {[agent.act_greedy(state) for _ in range(4)]}")
    print(f"memory after    : {len(agent.logprobs)} logprobs, "
          f"{len(agent.state_values)} values, "
          f"{len(agent.rewards)} rewards")

    # The discounted returns, BY HAND (gamma = 0.99, backward math):
    #     R4 = 100.0
    #     R3 = -0.5 + 0.99 * 100.0   = 98.5
    #     R2 = -0.5 + 0.99 * 98.5    = 97.015
    #     R1 = -0.5 + 0.99 * 97.015  = 95.54485
    #     R0 = 1.0 + 0.99 * 95.54485 = 95.5894015
    hand = [95.5894015, 95.54485, 97.015, 98.5, 100.0]
    calc, running = [], 0.0        # the SAME backward math the net
    for r in rewards_given[::-1]:  # runs in calculate_loss step 1
        running = r + 0.99 * running
        calc.insert(0, running)
    assert calc == hand, f"backward math disagrees: {calc}"
    print(f"hand discounted : {hand}   (re-derived, must match)")
    print("(calculate_loss normalizes these, mean/std, before use)")
    agent.clear_memory()
    print(f"memory cleared  : {len(agent.logprobs) + len(agent.rewards)} "
          f"entries left")
