"""
agent.py — THE BRAIN (step 2 of the cart-pole MLP project)
===========================================================

Project 02's brain was a Q-TABLE with a binner in front of it:

    4 floats -> buckets (5,4,6,4) -> one row of 7,680 numbers.

This brain is a DQN: the SAME Q-learning math, but an MLP predicts
the Q-values instead of looking them up:

    4 floats -> normalize -> MLP(4 -> 64 -> 64 -> 2) -> [Q(LEFT), Q(RIGHT)]

WHY AN MLP NEEDS NO BINNER
--------------------------
A table needs integer keys, so continuous floats had to be chopped
into buckets (and we paid for it: boundary dust, clipping, the
few-vs-many tradeoff). A neural net eats floats natively — matrix
multiplies don't care whether x is 0.0044 or 0.00441. So the whole
StateBinner class from 02 is GONE. The only preprocessing left is
NORMALIZATION (divide each input by its typical scale so no single
input dominates the gradients — see STATE_SCALE below).

THE THREE DQN TRICKS (beyond vanilla Q-learning)
------------------------------------------------
1. REPLAY BUFFER: instead of learning from each step once and
   throwing it away, we store the last N transitions and re-sample
   random mini-batches. Consecutive cart-pole steps look almost
   identical; training on them in order would chase its own tail.
   Random sampling breaks that correlation.

2. TARGET NETWORK: the TD target uses a SECOND, slow-moving copy
   of the MLP (updated every TARGET_SYNC steps). Without it, the
   network chases a target it moves every gradient step — like a
   dog chasing its own tail. The frozen copy holds still long
   enough to aim at.

3. EPSILON-GREEDY (unchanged from 02): with probability epsilon
   pick random (EXPLORE), else the MLP's best action (EXPLOIT).
   train.py decays epsilon 1.0 -> 0.05 exactly like before.

API (kept compatible with 02 so train.py barely changes):

    choose_action(state, explore=True)  -> LEFT (0) or RIGHT (1)
    update(state, action, reward, next_state, done) -> loss or None
        (stores the transition, then takes ONE gradient step on a
        random batch once the buffer is warm)

    sync_target()  — copy online weights -> target net
    save(path) / DQNAgent.load(path) — .pt checkpoint

DEFAULTS (picked to hit 475+ avg steps in step 3 — you can tune):
    hidden 64x64 ReLU, Adam lr=1e-3, gamma=0.99,
    buffer 50_000, batch 64, target sync every 500 updates.
"""

# Every import earns its comment (repo rule: explain every file well).
import math      # radians() in the demo's tilt example
import os        # join paths — save() writes next to agent.py by default
import random    # epsilon-greedy coin flips + replay-buffer sampling
from collections import deque  # the replay buffer: auto-evicts oldest

# PyTorch is the ONE new dependency vs 02 (see requirements.txt).
# We use only nn / optim / tensor basics — no Lightning, no Gym.
import torch
import torch.nn as nn

# Friendly defaults (same knobs style as 02, values retuned for DQN).
DEFAULT_HIDDEN = 64        # neurons per hidden layer (64x64 is the
                           # classic cart-pole DQN size: big enough to
                           # draw the decision boundary, small enough
                           # to train in seconds on CPU)
DEFAULT_LR = 1e-3          # Adam step size: 1e-3 learns fast without
                           # oscillating on this small net
DEFAULT_GAMMA = 0.99       # discount: rewards 10 ticks away count ~90%
DEFAULT_EPSILON = 1.0      # start as a pure explorer (decayed in train.py)
DEFAULT_BUFFER_SIZE = 50_000  # transitions remembered (500-step
                           # episodes x 100 recent episodes)
DEFAULT_BATCH_SIZE = 64    # transitions per gradient step (the DQN
                           # standard: stable average, cheap on CPU)
DEFAULT_TARGET_SYNC = 500  # gradient steps between target-net copies

# Normalization scales — one per state dimension. Raw cart-pole floats
# live on wildly different scales (x in ±2.4, spin in ±4 rad/s), and
# an MLP trains badly when one input is 100x another: its gradients
# drown the rest. Dividing maps "typical max" -> ±1.0:
#   x         ±2.4 m    (the track edge — matches environment.X_LIMIT)
#   x_dot     ±3.0 m/s  (same range 02's binner drew)
#   theta     ±12°      (the pole-fell limit, in radians)
#   theta_dot ±4.0 rad/s(same range 02's binner drew)
# Values CAN exceed ±1.0 mid-fall (a 20° death thump -> 1.67) — that
# is fine, the net just sees "far outside". No clipping: clipping
# would hide how badly it is falling.
STATE_SCALE = (2.4, 3.0, math.radians(12.0), 4.0)


def normalize_state(state):
    """Scale 4 raw floats into MLP-friendly ~[-1, 1] inputs.

    Args: state — (x, x_dot, theta, theta_dot) raw floats.
    Returns: list of 4 normalized floats (NOT a tensor — the net
             methods convert; keeping this plain keeps it testable
             without torch in the assertion).
    """
    # Zip pairs each value with its scale — one division per dim.
    return [float(v) / s for v, s in zip(state, STATE_SCALE)]


# ==============================================================
# 1. THE NETWORK — 4 inputs -> 2 Q-values
# ==============================================================
class QNetwork(nn.Module):
    """Tiny MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.

    4 inputs (normalized state), 2 outputs (Q for LEFT and RIGHT).
    No softmax at the end: Q-values are REGRESSION targets (any real
    number, e.g. 87.3), not probabilities — the policy just picks
    argmax, and softmax would squash the scale the TD error needs.
    """

    def __init__(self, n_inputs=4, n_actions=2, hidden=DEFAULT_HIDDEN):
        """Build the three weight matrices (+ biases).

        Args: n_inputs — 4 (x, x_dot, theta, theta_dot, normalized).
              n_actions — 2 (LEFT/RIGHT Q-values out).
              hidden — width of the two hidden layers.
        """
        super().__init__()
        # Sequential reads like the forward pass itself:
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),  # 4 -> 64: learn tilt detectors
            nn.ReLU(),                    # bend the lines (without this
                                          # the whole net is one plane)
            nn.Linear(hidden, hidden),    # 64 -> 64: combine detectors
            nn.ReLU(),                    # second bend: curved boundary
            nn.Linear(hidden, n_actions),  # 64 -> 2: one Q per action
        )

    def forward(self, x):
        """Run a batch of normalized states -> batch of Q-vectors."""
        return self.net(x)


# ==============================================================
# 2. THE REPLAY BUFFER — the brain's diary
# ==============================================================
class ReplayBuffer:
    """A fixed-size diary of transitions, sampled at random.

    deque(maxlen=N) auto-evicts the OLDEST transition when full —
    the diary always holds the N MOST RECENT steps. Random sampling
    breaks the fatal correlation of learning from consecutive,
    near-identical steps in order.
    """

    def __init__(self, capacity=DEFAULT_BUFFER_SIZE):
        """Allocate the diary (empty — train.py fills it by playing)."""
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        """How many transitions are stored (train.py waits until this
        passes BATCH_SIZE before each gradient step)."""
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        """File one experience away (called every environment step)."""
        self.buffer.append((tuple(state), int(action), float(reward),
                            tuple(next_state), bool(done)))

    def sample(self, batch_size):
        """Draw a RANDOM batch (the whole point — decorrelated)."""
        return random.sample(self.buffer, batch_size)


# ==============================================================
# 3. THE DQN AGENT — scorecard + update rule + exploration
# ==============================================================
class DQNAgent:
    """DQN brain: MLP + replay buffer + frozen target net.

    The learning rule is 02's TD update with two upgrades:
      02: Q(s,a) += alpha * (r + gamma*max Q(next) - Q(s,a))  (table)
      03: same target, but gradient descent on the MLP moves ALL
          similar states a little (generalization — the binner's
          "share a row" idea, but smooth instead of blocky).
    """

    def __init__(self, n_actions=2, hidden=DEFAULT_HIDDEN, lr=DEFAULT_LR,
                 gamma=DEFAULT_GAMMA, epsilon=DEFAULT_EPSILON,
                 buffer_size=DEFAULT_BUFFER_SIZE,
                 batch_size=DEFAULT_BATCH_SIZE,
                 target_sync=DEFAULT_TARGET_SYNC, seed=None):
        """
        Args:
            n_actions: 2 (LEFT=0, RIGHT=1 — counted, never learned).
            hidden: MLP width (both hidden layers share it).
            lr: Adam learning rate.
            gamma: discount factor (future rewards still count).
            epsilon: exploration rate (probability of a random move).
            buffer_size: replay diary capacity.
            batch_size: transitions per gradient step.
            target_sync: gradient steps between target-net copies.
            seed: optional int — seeds random + torch so tests and
                  demos are reproducible (train.py passes one in).
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.lr = lr
        self.hidden = hidden

        # Reproducibility: one seed drives Python-random (buffer
        # sampling, epsilon flips) AND torch (weight init, batches).
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # CPU on purpose: this 4-64-64-2 net trains in seconds on CPU,
        # and CPU is deterministic across machines (no cuDNN jitter).
        self.device = torch.device("cpu")
        self.online = QNetwork(4, n_actions, hidden).to(self.device)
        self.target = QNetwork(4, n_actions, hidden).to(self.device)
        self.sync_target()  # target starts as an exact copy

        # Adam: per-weight adaptive step sizes (forgives our 1e-3 guess
        # across layers with different gradient scales). MSE loss:
        # plain squared TD-error — the direct translation of 02's
        # (target - Q)², and hand-checkable in tests.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size)
        self.updates = 0  # gradient steps taken (drives target sync)

    # ------------------------------------------------------------------
    # Helpers — state -> tensor, net -> Q-values
    # ------------------------------------------------------------------
    def _to_tensor(self, states):
        """Normalize + convert state(s) to a float32 batch tensor.

        Accepts ONE state (4 floats) or a BATCH (list of states).
        A single state becomes a batch of 1 — Linear layers always
        want 2-D [batch, features], so this keeps call sites simple.
        """
        # Single state looks like (0.1, 0.2, 0.05, -0.3): first
        # element is a float, not a tuple — wrap it into a batch.
        if isinstance(states[0], (int, float)):
            states = [states]
        normed = [normalize_state(s) for s in states]
        return torch.tensor(normed, dtype=torch.float32,
                            device=self.device)

    def get_q(self, state):
        """Return this state's Q-values as a plain 2-float list.

        No auto-create (02 made table ROWS here) — the MLP already
        answers for EVERY state, visited or not (generalization).
        Runs under no_grad: reading Q-values must never build a
        gradient graph (only training does that).
        """
        self.online.eval()  # eval mode (no dropout/batchnorm here, but
                            # correct habit — future layers may need it)
        with torch.no_grad():
            q = self.online(self._to_tensor(state))[0]
        return [float(v) for v in q.tolist()]

    # ------------------------------------------------------------------
    # 1. THE POLICY — decide what to do
    # ------------------------------------------------------------------
    def choose_action(self, state, explore=True):
        """Pick an action: sometimes random, otherwise best guess.

        Epsilon-greedy, identical contract to 02:
            with prob epsilon -> random   (EXPLORE)
            else              -> argmax Q (EXPLOIT)
        explore=False forces pure exploitation — the mode used to
        WATCH a trained agent (no more mistakes).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # exploration
        q = self.get_q(state)
        # Exploit: highest Q. Ties break to the lowest index —
        # deterministic and testable (02's rule, reused).
        return max(range(self.n_actions), key=lambda a: q[a])

    # ------------------------------------------------------------------
    # 2. THE LEARNING RULE — file one experience, take one step
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One environment step of learning (the Trainer calls this).

        Stores the transition, then — once the buffer holds a full
        batch — takes ONE gradient step on a random batch. Returns
        the loss float, or None while still warming up (lets train.py
        log "warming up" instead of fake zeros).

        Keeping this single-transition signature (same as 02) means
        step 3's training loop does NOT change shape — only the
        brain's insides got smarter.
        """
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return None  # warming up: not enough diary pages yet
        batch = self.buffer.sample(self.batch_size)
        return self.train_step(batch)

    def train_step(self, batch):
        """ ONE gradient step on a given batch (testable in isolation).

        The DQN loss, piece by piece:
            target = r                              if done
                   = r + gamma * max Q_target(s')  otherwise
            loss   = mean((target - Q_online(s,a))²)
        Crucial detail: Q_target comes from the FROZEN net (stable
        aim), Q_online from the trained net. Gradients flow ONLY
        through Q_online — targets are detached (no_grad), or the
        net would chase its own tail.
        """
        # --- unpack the batch into column tensors ---------------------
        states = self._to_tensor([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long,
                               device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                               device=self.device)
        next_states = self._to_tensor([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32,
                             device=self.device)

        # --- current estimates: Q_online(s, a taken) -------------------
        # gather(1, actions): pick the ONE column (LEFT or RIGHT) that
        # was actually taken — the loss only judges what we did.
        q_sa = self.online(states).gather(1, actions).squeeze(1)

        # --- TD targets from the FROZEN net (no gradients) -------------
        with torch.no_grad():
            # max(1)[0]: best next-Q per row (greedy bootstrap — the
            # DQN half of "Q-learning"; the MLP half is below).
            max_next = self.target(next_states).max(1)[0]
            # (1 - dones): zeroes the future on terminal steps — a
            # fall's target is just the fatal reward (02's rule).
            target = rewards + self.gamma * (1.0 - dones) * max_next

        # --- descend ----------------------------------------------------
        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()  # clear last step's gradients first
        loss.backward()             # blame each weight for the error
        self.optimizer.step()       # Adam nudges every weight once

        # --- frozen target catches up on schedule -----------------------
        self.updates += 1
        if self.updates % self.target_sync == 0:
            self.sync_target()

        return float(loss.item())

    def sync_target(self):
        """Copy online weights -> target net (the stabilizing trick).

        Called automatically every TARGET_SYNC gradient steps, and
        once at construction so both nets start identical.
        """
        self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    # 3. MEMORY OUTSIDE THE BRAIN — save / load checkpoints
    # ------------------------------------------------------------------
    def save(self, path):
        """Write MLP weights + hyperparams to a .pt checkpoint.

        Unlike 02's JSON table, weights are tensors — torch.save
        handles them natively. Hyperparams travel along so load()
        rebuilds the IDENTICAL architecture (a 64-wide brain cannot
        load 128-wide weights).
        """
        torch.save({
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "batch_size": self.batch_size,
            "target_sync": self.target_sync,
            "updates": self.updates,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
        }, path)

    @classmethod
    def load(cls, path):
        """Rebuild an agent from save() (weights + hyperparams)."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(n_actions=data["n_actions"], hidden=data["hidden"],
                    lr=data["lr"], gamma=data["gamma"],
                    epsilon=data["epsilon"],
                    batch_size=data["batch_size"],
                    target_sync=data["target_sync"])
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.updates = data.get("updates", 0)
        return agent


# ------------------------------------------------------------------
# Tiny demo: normalization, shapes, one learning step.
# (The real training loop arrives in step 3 — train.py.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("normalize (0,0,0,0)          ->", normalize_state((0, 0, 0, 0)))
    print("  hand: every 0 / scale = 0.0 -> [0, 0, 0, 0] (dead center)")
    print("normalize (2.4, 0, 12deg, 0) ->",
          normalize_state((2.4, 0.0, math.radians(12), 0.0)))
    print("  hand: track edge + tilt limit -> [1, 0, 1, 0] (the cliff)")

    agent = DQNAgent(seed=0)
    home = (0.0, 0.0, 0.0, 0.0)
    print(f"\nQ(home) untrained            -> {agent.get_q(home)}")
    print("  random small numbers (torch init) — no optimism table now;\n"
          "  exploration comes from epsilon, not q_init.")

    # One hand-checkable gradient step lives in test_agent.py — here
    # just prove the plumbing: fill a tiny buffer, learn once.
    tiny = DQNAgent(batch_size=4, target_sync=100, seed=0)
    for i in range(4):
        tiny.buffer.push(home, i % 2, 1.0, home, False)
    loss = tiny.train_step(list(tiny.buffer.buffer))
    print(f"one train_step on 4x(+1 alive) -> loss {loss:.4f}")
