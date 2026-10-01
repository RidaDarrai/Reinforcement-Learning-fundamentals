"""
agent.py — THE LEARNER (step 2 of our RL project)
===================================================

environment.py is the world. This file is the brain that will one
day replace the random-line in play.py:

    apply_step(env, ui, random.randrange(N_ACTIONS))   # monkey
    apply_step(env, ui, agent.choose_action(state))    # brain

An agent = a POLICY (decide what to do) + a LEARNING RULE (get
better from experience).

Two skills, two methods:

    choose_action(state)          -> "what should I do here?"
    update(s, a, r, s2, done)     -> "file this experience away"

THE BIG IDEA — Q-learning
-------------------------
The agent keeps a scorecard: for every position and every action,
one number Q = "how much total reward do I expect from here if I
take this move, then play reasonably after?"

    Q[(2,3)][DOWN] = 8.2   "moving down from (2,3) is worth ~8.2"

After every single step, the score you just used gets nudged
toward a better estimate ("temporal difference" update):

    target = r                          if the episode ended
           = r + gamma * max Q(s2, ...)  if it continues

    Q(s,a) += alpha * (target - Q(s,a))
              |________|      |________|
               learning     how wrong     -> the ERROR we fix
                 rate        we were

The three knobs (they will haunt you forever — in a good way):

    alpha   (0..1)  learning rate  — 0.01 = tiny timid updates,
                                      0.99 = rewrite memory wildly
    gamma   (0..1)  discount       — 0.9 = "reward soon beats reward
                                      later"; 0.99 = plan far ahead
    epsilon (0..1)  exploration    — chance of doing something
                                      RANDOM instead of the current
                                      best guess (see choose_action)

This file is environment-agnostic on purpose: it never imports
environment.py, takes n_actions as a parameter, and works unchanged
on our maze, CartPole, or anything else with discrete actions.
"""

import json
import random

# Friendly defaults for our maze; every one can be overridden.
DEFAULT_ALPHA = 0.1     # small steps: trust experience gradually
DEFAULT_GAMMA = 0.99    # far-sighted: future rewards still matter
DEFAULT_EPSILON = 1.0   # start as a pure explorer (see choose_action)


class QAgent:
    """A tabular Q-learning agent: scorecard + update rule."""

    def __init__(self, n_actions, alpha=DEFAULT_ALPHA,
                 gamma=DEFAULT_GAMMA, epsilon=DEFAULT_EPSILON):
        """
        Args:
            n_actions: how many distinct moves exist (4 in the maze)
            alpha:   learning rate   (how hard we re-learn per step)
            gamma:   discount factor (how much future counts now)
            epsilon: exploration rate (probability of a random move)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # THE SCORECARD. A dict keyed by state -> list of n_actions
        # floats. A dict (not a giant pre-filled array) because most
        # states are never visited — we only store what we've seen.
        # Missing states auto-create all-zero rows in get_q(): zero
        # means "no experience yet", i.e. "as good as anything else".
        self.q_table = {}

    # ------------------------------------------------------------------
    # 1. THE SCORECARD — get (and auto-create) the row for a state
    # ------------------------------------------------------------------
    def get_q(self, state):
        """Return this state's list of Q-values, creating it if new.

        New states start at 0.0 for every action — the honest value
        of complete ignorance ("might be great, might not").
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
        return self.q_table[state]

    # ------------------------------------------------------------------
    # 2. THE POLICY — decide what to do
    # ------------------------------------------------------------------
    def choose_action(self, state, explore=True):
        """Pick an action: sometimes random, otherwise the best guess.

        This is epsilon-greedy, the classic explore/exploit compromise:

            with probability epsilon  -> random action  (EXPLORE:
                wander, try dumb things, discover hidden treasure)
            with probability 1-eps    -> the CURRENT BEST action
                (EXPLOIT: cash in what we think we know)

        explore=False forces pure exploitation — that's the mode we'll
        use to WATCH a trained agent perform (no more mistakes).
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.n_actions)   # exploration

        q = self.get_q(state)
        # Exploit: highest Q. Ties break toward the lowest index,
        # which keeps behavior deterministic and testable.
        best_action = max(range(self.n_actions), key=lambda a: q[a])
        return best_action

    # ------------------------------------------------------------------
    # 3. THE LEARNING RULE — file one experience away
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """Nudge Q(state,action) toward what that experience suggests.

        Args:
            state:     where we were
            action:    what we did there
            reward:    what the environment said it was worth
            next_state:where we ended up
            done:      did the episode end? (crucial — see below)

        The target has two flavors:

            done=True : the future is EMPTY — a win or a loss is the
                        whole truth. target = reward (no daydreaming
                        about rewards that can never arrive).

            done=False: we didn't just get `reward`, we also inherited
                        the best future available from next_state.
                        target = reward + gamma * max Q(next_state).
                        This "chaining" is how a +10 at the goal
                        slowly teaches the moves from ten steps back.
        """
        q = self.get_q(state)

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.get_q(next_state))

        # The learning step: move the old value a fraction (alpha)
        # of the way from where it was to the new target.
        # alpha=1.0 would snap to the target; alpha=0.1 creeps.
        error = target - q[action]          # how wrong were we?
        q[action] += self.alpha * error     # fix it gradually

    # ------------------------------------------------------------------
    # 4. MEMORY OUTSIDE THE BRAIN — save / load the scorecard
    # ------------------------------------------------------------------
    def save(self, path):
        """Write the scorecard to a JSON file (so we train once,
        watch forever — no retraining after closing VS Code)."""
        data = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            # JSON can't use tuples as keys, so "(row, col)" becomes
            # the string "row,col" — parsed back in load().
            "q_table": {
                f"{r},{c}": values
                for (r, c), values in self.q_table.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)

    @classmethod
    def load(cls, path):
        """Rebuild an agent from a file saved by save()."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        agent = cls(
            n_actions=data["n_actions"],
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
        )
        for key, values in data["q_table"].items():
            r, c = (int(x) for x in key.split(","))
            agent.q_table[(r, c)] = [float(v) for v in values]
        return agent


# ------------------------------------------------------------------
# Tiny demo: watch a single Q-value learn from one experience.
# (The real training loop arrives in step 3 — train.py.)
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = QAgent(n_actions=4, alpha=0.5, gamma=0.9)

    print("initial Q[(5,6)]      :", agent.get_q((5, 6)))
    # Standing one step from the goal, moving DOWN onto it:
    # reward +10, episode over -> target is exactly 10 (no future).
    agent.update(state=(5, 6), action=1, reward=+10.0,
                 next_state=(6, 6), done=True)
    print("after landing on goal :", agent.get_q((5, 6)))
    # Hand check: 0 + 0.5 * (10 - 0) = 5.0

    # One step back: reward -0.1, continues into (5,6) whose best
    # value is now 5.0 -> target = -0.1 + 0.9*5.0 = 4.4
    agent.update(state=(4, 6), action=1, reward=-0.1,
                 next_state=(5, 6), done=False)
    print("one step further back :", agent.get_q((4, 6)))
    # Hand check: 0 + 0.5 * (4.4 - 0) = 2.2
