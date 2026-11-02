"""
agent.py — THE BRAIN (step 2 of the lunar lander project)
=========================================================

The cart-pole projects built the same lesson twice on purpose:
    02: a Q-TABLE with a binner in front of it (4 floats -> buckets
        -> 7,680 rows).
    03: a DQN — the SAME Q-learning math, but an MLP predicts the
        Q-values (no buckets, no clipping).

This project reuses NOTHING from those folders (independence is a
rule here) — but it keeps the DQN lesson, retuned for a 3-DOF world:

    8 floats -> normalize -> MLP(8 -> 64 -> 64 -> 4) -> 4 Q-values
                                              (IDLE/MAIN/LEFT/RIGHT)

WHAT CHANGED vs 03's DQN (same three stabilizers, new numbers)
-------------------------------------------------------------
1. QUADRUPLE the output: 4 button Q-values instead of 2. The argmax
   policy is unchanged — highest Q wins, ties break to the lowest
   index (deterministic, testable).
2. The 8 inputs come pre-normalized by environment.py's
   normalize_state() — imported from THERE, not copied here. One
   source of truth: if the world changes its scales, the brain
   follows automatically. 03 kept a private copy; this project has
   the stricter habit.
3. The replay diary almost doubled to 100,000 transitions: lander
   episodes can run 1,000 ticks (cart-pole capped at 500), so the
   diary must hold ~100 recent episodes of correlated adjacent
   steps before random sampling decorrelates them into signal.

THE THREE DQN TRICKS (unchanged in spirit — they are why naive
neural Q-learning does not explode):
1. REPLAY BUFFER: store the last N transitions, train on RANDOM
   64-batches. Consecutive lander ticks look nearly identical
   (falling, falling, falling...) — learning from them in order
   chases its own tail.
2. TARGET NETWORK: the TD target r + gamma * max Q_target(s') is
   computed by a FROZEN copy of the net, synced every
   TARGET_SYNC gradient steps. Without it the net aims at a target
   that moves every step.
3. EPSILON-GREEDY: with probability epsilon pick a random button
   (explore), else the best Q (exploit). train.py decays it
   1.0 -> 0.05; here it is a stored dial.

API (compatible in shape with 02/03 so step 3's loop feels at home):

    choose_action(state, explore=True) -> 0..3
    update(state, action, reward, next_state, done) -> loss or None

    sync_target()
    save(path) / DQNAgent.load(path)

DEFAULTS (picked to land reliably in step 3 — tune via knobs):
    hidden 64x64 ReLU, Adam lr=1e-3, gamma=0.99,
    buffer 100_000, batch 64, target sync every 500 gradient steps.

Try it (a tiny hands-on tour — no training needed):

    python agent.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import math      # isfinite in the demo's finiteness check
import os        # join paths — save() writes next to agent.py by default
import random    # epsilon-greedy coin flips + replay-buffer sampling
from collections import deque  # the replay buffer: auto-evicts oldest

# PyTorch is the ONE heavy dependency (see ../requirements.txt: torch
# was added for 03). Only nn / optim / tensor basics — no Lightning,
# no Gym, no training frameworks.
import torch
import torch.nn as nn

# The state contract lives in the WORLD — the brain borrows its
# scales instead of keeping a private copy that could drift. This
# import stays inside the project folder (independence kept).
from environment import N_ACTIONS, normalize_state

# Friendly defaults (same knobs style as 02/03, retuned for landing):
DEFAULT_N_INPUTS = 8       # the documented 8-float state — never
                           # hardcode a magic 4 again
DEFAULT_N_ACTIONS = N_ACTIONS   # 4 buttons, straight from the world
DEFAULT_HIDDEN = 64        # neurons per hidden layer — 64x64 solved
                           # cart-pole DQN; the lander's value surface
                           # is richer, 128 exists as a step-3 knob
DEFAULT_LR = 1e-3          # Adam step size — same as 03: learns fast
                           # without oscillating on a net this small
DEFAULT_GAMMA = 0.99       # discount. Landing's +100 sits ~3-20 s away
                           # -> ~150-1000 ticks -> 0.99^300 ~ 0.05: high
                           # enough that the far promise still counts
DEFAULT_EPSILON = 1.0      # start a pure explorer (train.py decays)
DEFAULT_BUFFER_SIZE = 100_000  # transitions kept. 1,000-tick episodes
                           # mean ~100 recent episodes — enough for
                           # random sampling to break the correlation
DEFAULT_BATCH_SIZE = 64    # transitions per gradient step (DQN classic)
DEFAULT_TARGET_SYNC = 500  # gradient steps between target copies


# ==============================================================
# 1. THE NETWORK — 8 inputs -> 4 Q-values
# ==============================================================
class QNetwork(nn.Module):
    """Tiny MLP: Linear -> ReLU -> Linear -> ReLU -> Linear.

    8 inputs (normalized state: x, y, vx, vy, theta, omega, legL,
    legR), 4 outputs (one Q per engine button). No softmax at the
    end: Q-values are REGRESSION targets (any real number, e.g.
    87.3), not probabilities — the policy picks argmax, and a
    softmax would squash the scale the TD error needs.
    """

    def __init__(self, n_inputs=DEFAULT_N_INPUTS,
                 n_actions=DEFAULT_N_ACTIONS, hidden=DEFAULT_HIDDEN):
        """Build the three weight matrices (+ biases).

        Args: n_inputs  — 8 (the world's documented state width).
              n_actions — 4 (IDLE/MAIN/LEFT/RIGHT Q-values out).
              hidden    — width of the two hidden layers.
        """
        super().__init__()
        # Sequential reads like the forward pass itself:
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),   # 8 -> 64: "approach" and
                                           # "level" detectors
            nn.ReLU(),                     # bend the lines (without
                                           # this it is one plane)
            nn.Linear(hidden, hidden),     # 64 -> 64: combine them
            nn.ReLU(),                     # second bend: curved lines
            nn.Linear(hidden, n_actions),  # 64 -> 4: one Q per button
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
    near-identical steps in order (falling, falling, falling...).
    """

    def __init__(self, capacity=DEFAULT_BUFFER_SIZE):
        """Allocate the diary (empty — step 3 fills it by flying)."""
        self.buffer = deque(maxlen=int(capacity))

    def __len__(self):
        """How many transitions are stored (train.py waits until
        this passes BATCH_SIZE before each gradient step)."""
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        """File one experience away (called every environment tick)."""
        self.buffer.append((tuple(state), int(action), float(reward),
                            tuple(next_state), bool(done)))

    def sample(self, batch_size):
        """Draw a RANDOM batch (the whole point — decorrelated)."""
        return random.sample(self.buffer, batch_size)


# ==============================================================
# 3. THE DQN AGENT — policy + learning rule + exploration dial
# ==============================================================
class DQNAgent:
    """DQN brain: MLP + replay buffer + frozen target net.

    Same learning rule as 03, wider aim:
      03: TD target r + gamma*max Q(next) -> MSE into a 2-output MLP
      04: SAME rule -> MSE into a 4-output MLP; argmax policy now
          balances thrust AND rotation instead of pushing a cart.
    """

    def __init__(self, n_inputs=DEFAULT_N_INPUTS,
                 n_actions=DEFAULT_N_ACTIONS, hidden=DEFAULT_HIDDEN,
                 lr=DEFAULT_LR, gamma=DEFAULT_GAMMA,
                 epsilon=DEFAULT_EPSILON,
                 buffer_size=DEFAULT_BUFFER_SIZE,
                 batch_size=DEFAULT_BATCH_SIZE,
                 target_sync=DEFAULT_TARGET_SYNC, seed=None):
        """
        Args:
            n_inputs: 8 (state width; keep == environment's contract).
            n_actions: 4 (one Q each; counted, never learned).
            hidden: MLP width (both hidden layers share it).
            lr: Adam learning rate.
            gamma: discount factor (future rewards still count).
            epsilon: exploration rate (probability of random move).
            buffer_size: replay diary capacity.
            batch_size: transitions per gradient step.
            target_sync: gradient steps between target-net copies.
            seed: optional int — seeds Python-random (epsilon coin
                  flips, buffer sampling) AND torch (weight init, so
                  tests and demos replay identically).
        """
        self.n_inputs = n_inputs
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

        # CPU on purpose: this 8-64-64-4 net trains in minutes on CPU,
        # and CPU is deterministic across machines (no cuDNN jitter).
        self.device = torch.device("cpu")
        self.online = QNetwork(n_inputs, n_actions, hidden).to(self.device)
        self.target = QNetwork(n_inputs, n_actions, hidden).to(self.device)
        self.sync_target()  # target starts as an exact copy

        # Adam: per-weight adaptive step sizes (forgives our 1e-3
        # guess across layers with different gradient scales). MSE:
        # plain squared TD-error — the direct translation of 02's
        # (target - Q)^2, hand-checkable in tests.
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_size)
        self.updates = 0  # gradient steps taken (drives target sync)

    # ------------------------------------------------------------------
    # Helpers — state(s) -> tensor pass
    # ------------------------------------------------------------------
    def _to_tensor(self, states):
        """Normalize + convert state(s) into a float32 batch tensor.

        Accepts ONE state (8 numbers) or a BATCH (list of states).
        A single state becomes a batch of 1 — Linear layers always
        want 2-D [batch, features], so this keeps call sites simple.
        """
        # A single state looks like (0.1, 70.0, ...): the FIRST entry
        # is a number, not a tuple — wrap it into a batch of one.
        if isinstance(states[0], (int, float)):
            states = [states]
        normed = [normalize_state(s) for s in states]
        return torch.tensor(normed, dtype=torch.float32,
                            device=self.device)

    def get_q(self, state):
        """Return this state's Q-values as a plain 4-float list.

        No auto-create (02 made table ROWS here) — an MLP answers
        for EVERY state, visited or not (generalization). Runs under
        no_grad: reading Q-values must never build a gradient graph
        (only training does that).
        """
        self.online.eval()  # eval mode (no dropout/batchnorm here, but
                            # correct habit — future layers may need it)
        with torch.no_grad():
            q = self.online(self._to_tensor(state))[0]
        return [float(v) for v in q.tolist()]

    # ------------------------------------------------------------------
    # 1. THE POLICY — decide which button to press
    # ------------------------------------------------------------------
    def choose_action(self, state, explore=True):
        """Epsilon-greedy over FOUR buttons:

            with prob epsilon -> random   (EXPLORE)
            else              -> argmax Q (EXPLOIT)
        explore=False forces pure exploitation — the mode used to
        watch a trained agent land (no more coin flips).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # exploration
        q = self.get_q(state)
        # Exploit: highest Q. Ties break to the lowest index —
        # deterministic and testable (the 02/03 rule, kept).
        return max(range(self.n_actions), key=lambda a: q[a])

    # ------------------------------------------------------------------
    # 2. THE LEARNING RULE — file one experience, take one step
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One environment tick of learning (the Trainer calls this).

        Stores the transition, then — once the buffer holds a full
        batch — takes ONE gradient step on a random batch. Returns
        the loss float, or None while still warming up (lets step 3
        log "warming up" instead of fake zeros).

        Same single-transition signature as 02/03, so the training
        loop does NOT change shape — the world got harder, the loop
        stayed boring (that is the recipe).
        """
        self.buffer.push(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return None  # warming up: not enough diary pages yet
        batch = self.buffer.sample(self.batch_size)
        return self.train_step(batch)

    def train_step(self, batch):
        """ONE gradient step on a given batch (testable in isolation).

        The DQN loss, piece by piece:
            target = r                                  if done
                   = r + gamma * max Q_target(s')       otherwise
            loss   = mean((target - Q_online(s,a))^2)

        Crucial detail: Q_target comes from the FROZEN net (stable
        aim), Q_online from the trainable net. Gradients flow ONLY
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
        # gather(1, actions): pick the ONE of the 4 columns that was
        # actually pressed — the loss only judges what we did.
        q_sa = self.online(states).gather(1, actions).squeeze(1)

        # --- TD targets from the FROZEN net (no gradients) -------------
        with torch.no_grad():
            # max(1)[0]: best next-Q per row (greedy bootstrap — the
            # Q-learning half of "Q-learning"; the neural half above).
            max_next = self.target(next_states).max(1)[0]
            # (1 - dones): zeroes the future on terminal ticks — a
            # crash's target is exactly its -100 bill, nothing more.
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

        Like 03: weights are tensors, so torch.save handles them
        natively (02 used a JSON table). Hyperparams travel along so
        load() rebuilds the IDENTICAL architecture — a 64-wide brain
        cannot load 128-wide weights.
        """
        torch.save({
            "n_inputs": self.n_inputs,
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
        agent = cls(n_inputs=data["n_inputs"],
                    n_actions=data["n_actions"],
                    hidden=data["hidden"],
                    lr=data["lr"],
                    gamma=data["gamma"],
                    epsilon=data["epsilon"],
                    batch_size=data["batch_size"],
                    target_sync=data["target_sync"])
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.updates = data.get("updates", 0)
        return agent


# ------------------------------------------------------------------
# Tiny demo: state width, normalization, Q shapes, one learning step.
# (The real training loop arrives in step 3 — train.py.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"state width: {DEFAULT_N_INPUTS} floats in, "
          f"{DEFAULT_N_ACTIONS} Q-values out")
    print("brain shape : MLP(8 -> 64 -> 64 -> 4), Adam 1e-3, "
          f"gamma {DEFAULT_GAMMA}, buffer {DEFAULT_BUFFER_SIZE:,}")

    # Normalization rides on the WORLD's contract (single source).
    spawn = (0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    print(f"normalize (0,70,0,0,0,0,0,0) -> "
          f"{[round(v, 3) for v in normalize_state(spawn)]}"
          "  (y/80 ~ 0.875: high above the pad)")

    agent = DQNAgent(seed=0)
    q = agent.get_q(spawn)
    print(f"\nQ(untrained, spawn)        -> "
          f"{[round(v, 3) for v in q]}")
    print("  random small numbers (torch init), no optimism table")
    print(f"  all finite: {all(math.isfinite(v) for v in q)}")

    # Exploration coin flips: with epsilon=1.0 every press is random.
    presses = [agent.choose_action(spawn) for _ in range(12)]
    names = (
        "IDLE", "MAIN", "LEFT", "RIGHT",
    )
    print(f"\nepsilon 1.0 -> 12 random presses: "
          f"{[names[p] for p in presses]}")

    # One honest gradient step on a crafted batch: 64 copies of the
    # same bootstrapped experience, loss visibly > 0.
    agent2 = DQNAgent(batch_size=4, target_sync=1000, seed=0)
    for _ in range(4):
        agent2.buffer.push(spawn, 1, 1.0, spawn, False)
    loss = agent2.train_step(list(agent2.buffer.buffer))
    print(f"\none train_step on 4 copies of (r=+1, loop) -> "
          f"loss {loss:.6f}  (this exact loss is recomputed BY HAND"
          f"\n  in test_agent.py - if the test passes, the TD"
          f"\n  machinery is honest)")
