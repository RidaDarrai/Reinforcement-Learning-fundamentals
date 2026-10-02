"""
train.py — THE SCHOOL (step 3 of our RL project)
=================================================

Step 1 gave the world (environment.py).
Step 2 gave an empty brain (agent.py).
Step 3 puts them together and fills the notebook:

    for each episode:
        start with 4 noisy floats
        while not finished:
            action  = agent.choose_action(state)        # decide
            state', reward, done, info = env.step(action)   # act
            agent.update(state, action, reward, state', done)  # learn
        remember how long we survived
        shrink epsilon a little (explore less as we know more)

That inner loop IS reinforcement learning. Everything else is
bookkeeping: logging, saving, and the report card.

THE METRIC IS DIFFERENT HERE (read this twice)
----------------------------------------------
In the maze, reward could be negative and only "reached the goal"
counts as winning. Cart-pole pays +1.0 for every tick alive, so:

    total reward  ==  steps survived   (ALWAYS, by design)
    a "perfect" episode = the 500-step time limit
    falling at step 12 scores 12 — failure and progress are
    the SAME axis here

So THE number to watch is **avg steps survived** (over the last
100 episodes). The learning curve of this project IS the survival
curve: it should climb from ~20 toward 500.

THREE LESSONS THIS FILE LEARNED THE HARD WAY (we tested each)
--------------------------------------------------------------
1. Pure "+1 per tick" PLATEAUED around ~100 steps: dying at tick
   95 and balancing beautifully looked the same to the brain.
   Fix: FALL_PENALTY — the final tick of a doomed episode is also
   taught a bill (default -30). The CHARTS still show raw +1/tick
   (history never lies); only what the brain is TAUGHT changes.
2. Brand-new rows started at 0.0, so the never-tried action lost
   every comparison in the death zone and never got a chance.
   Fix lives in agent.py: rows start OPTIMISTIC (q_init = 50).
3. Training OSCILLATES: a brain scoring 95% at episode 9,000 can
   be at 20% by 10,000 (fixed learning rate + random shakes).
   Fix: a GREEDY EXAM every EXAM_EVERY episodes, and finish()
   saves the BEST exam brain — the diploma, not the last mood.

The loop lives in the Trainer class, so it can be driven step by
step: this file's command line runs it all at once (fast, silent),
while play.py's LIVE mode feeds it a few episodes per frame so you
can WATCH learning happen (step 4).

Two files come out of training:

    q_table.json        the trained brain (loaded later by play.py)
    training_log.json   every episode's score + outcome — the raw
                        material for step 5's charts

    python train.py
"""

import json
import os
import time

from agent import QAgent
from environment import CartPole, MAX_STEPS

# Save next to THIS file (not whatever folder you launched python
# from) — play.py looks for q_table.json in exactly this location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# THE KNOBS — hyperparameters (change these, change everything)
# ---------------------------------------------------------------
EPISODES = 15000       # practice runs (more episodes = more exams =
                       # more chances to catch a peak — see below)
ALPHA = 0.1            # learning rate: rewrite 10% toward each new truth
GAMMA = 0.99           # discount: rewards 10 ticks away still count ~90%
EPSILON_START = 1.0    # start as a pure explorer...
EPSILON_MIN = 0.05     # ...end as a 95% exploiter
EPSILON_DECAY = 0.995  # multiply epsilon by this after every episode
LOG_EVERY = 100        # print a progress line every N episodes

# -- reward shaping: how endings PAY --------------------------------
# The raw game pays +1 every tick and nothing at the end. Our
# first runs PLATEAUED around ~100 steps with that: dying at tick
# 95 and balancing beautifully both just "stop earning". These
# knobs sharpen the ending without touching environment.py — and
# without lying to the charts (history always stores the RAW
# +1/tick score; only what the brain is TAUGHT gets shaped):
#
#   FALL_PENALTY     added to the final tick's update when an
#                    episode ends early (pole_fell / cart_out).
#                    Dying has a bill -> the actions right before
#                    the fall learn "avoid", and the untried
#                    alternative stops looking free. -30 was the
#                    value that finally broke the plateau (we
#                    swept -10/-20/-30/-50: -10 too soft, -50 too
#                    brutal — it punished lucky recoveries too).
#   TIME_LIMIT_BONUS extra payment for surviving all MAX_STEPS.
#                    OFF by default: in our tests it changed
#                    nothing measurable — kept as a knob to try.
FALL_PENALTY = -30.0
TIME_LIMIT_BONUS = 0.0

# -- the graduation exam (the "best diploma" trick) -----------------
# Training here oscillates badly (lesson 3 in the module docstring):
# the same seed can score 98% at episode 9,000 and 0% at 13,000.
# So every EXAM_EVERY episodes the agent sits a short GREEDY exam
# (no learning, no exploration) and finish() saves whichever brain
# scored best — like keeping the best checkpoint, except you can
# watch it happen. Exams run even in quiet mode (they're silent).
EXAM_EVERY = 500       # how often the exam comes around
EXAM_EPISODES = 50     # how long each exam lasts

SAVE_Q = os.path.join(BASE_DIR, "q_table.json")
SAVE_LOG = os.path.join(BASE_DIR, "training_log.json")

# The three phases every training run passes through. Which phase you
# are in depends ONLY on epsilon (the exploration rate):
#   EXPLORING  eps > 0.5    mostly random LEFT/RIGHT — shaking the
#                           cart to see what happens (yes, really)
#   LEARNING   eps > 0.05   mixing finds with cashing in
#   POLISHING  eps = floor  almost pure exploitation — riding balance
STAGE_NAMES = ("EXPLORING", "LEARNING", "POLISHING")


class Trainer:
    """One training run, advanced from the outside.

    The plain train() function below just calls run_all() once —
    but play.py's LIVE mode calls run_episodes(n) every frame,
    which is what lets you watch the agent improve stage by stage.
    """

    def __init__(self, episodes=EPISODES, alpha=ALPHA,
                 gamma=GAMMA, epsilon_start=EPSILON_START, quiet=False):
        """Prepare a fresh run: one world, one empty brain."""
        self.env = CartPole()
        self.episodes = episodes
        self.quiet = quiet
        self.agent = QAgent(n_actions=2, alpha=alpha, gamma=gamma,
                            epsilon=epsilon_start)
        self.started = time.time()

        # The diploma cabinet: best GREEDY-exam result so far, and
        # a snapshot of the brain that earned it (see _exam()).
        self.best_score = (-1.0, -1.0)   # (win %, avg steps)
        self.best_episode = 0
        self.best_q = None

        # history is a dict of parallel lists — one entry per episode —
        # ready to be turned into charts in step 5. NOTE: "reward" and
        # "steps" are the same number in this game (+1 per tick);
        # both names kept because the chart says "reward" and the
        # physics says "steps" — test_train.py checks they never
        # drift apart.
        self.history = {
            "episode": [],     # episode number
            "reward": [],      # total reward earned that episode
            "outcome": [],     # "pole_fell" / "cart_out" / "time_limit"
            "epsilon": [],     # exploration chance after this episode
            "steps": [],       # how many ticks it took
        }

    # -- where are we? ------------------------------------------------
    @property
    def episode(self):
        """How many episodes have been finished so far."""
        return len(self.history["episode"])

    @property
    def done(self):
        """True once the planned number of episodes has run."""
        return self.episode >= self.episodes

    @property
    def stage(self):
        """Index into STAGE_NAMES — driven purely by epsilon."""
        if self.agent.epsilon > 0.5:
            return 0
        if self.agent.epsilon > EPSILON_MIN:
            return 1
        return 2

    def win_rate(self, n=100):
        """% of the last n episodes that hit the 500-step cap.

        That's THIS game's definition of a win: survive the full
        time limit. (Falling is a loss, but a loss at step 400 is
        a much better loss than a fall at step 12 — that's what
        avg steps tracks, and why we print both.)
        """
        tail = self.history["outcome"][-n:]
        if not tail:
            return 0.0
        return 100.0 * sum(o == "time_limit" for o in tail) / len(tail)

    def avg_steps(self, n=100):
        """Average ticks survived over the last n episodes — THE
        metric of this project."""
        tail = self.history["steps"][-n:]
        if not tail:
            return 0.0
        return sum(tail) / len(tail)

    def best_avg_steps(self, n=100):
        """Best rolling n-episode average anywhere in the run —
        training doesn't improve in a straight line, and we want
        to remember the peak even if the tail cooled off."""
        steps = self.history["steps"]
        if len(steps) <= n:
            return self.avg_steps(n)
        return max(sum(steps[i:i + n]) / n
                   for i in range(len(steps) - n + 1))

    # -- advancing the run -------------------------------------------
    def run_episodes(self, n):
        """Run up to n more episodes. Returns how many actually ran.

        Each episode is the same inner loop you'd write by hand:
        choose, act, learn, remember — then shrink epsilon a little.
        """
        ran = 0
        for _ in range(n):
            if self.done:
                break

            state = self.env.reset()
            total_reward = 0.0

            # ---- THE LOOP (the actual reinforcement learning) ------
            while True:
                action = self.agent.choose_action(state)          # 1. decide
                next_state, reward, done, info = self.env.step(action)  # 2. act

                # What the brain is TAUGHT (may be shaped — see the
                # knobs at the top of this file):
                taught = reward
                if done:
                    if info["reason"] == "time_limit":
                        taught += TIME_LIMIT_BONUS    # the good ending
                    else:
                        taught += FALL_PENALTY        # 0.0 = no change

                self.agent.update(state, action, taught,        # 3. learn
                                  next_state, done)
                state = next_state
                total_reward += reward       # RAW +1/tick only: the
                if done:                     # history/charts stay honest
                    break
            # -------------------------------------------------------

            # Shrink exploration: early episodes shake the cart with
            # pure random LEFT/RIGHT; later ones cash in what was
            # found. Never below MIN — a 5% chance of silliness keeps
            # the agent honest forever.
            self.agent.epsilon = max(EPSILON_MIN,
                                     self.agent.epsilon * EPSILON_DECAY)

            self.history["episode"].append(self.episode + 1)
            self.history["reward"].append(total_reward)
            self.history["outcome"].append(info["reason"])
            self.history["epsilon"].append(self.agent.epsilon)
            self.history["steps"].append(info["steps"])
            ran += 1

            # ---- graduation exam (every EXAM_EVERY episodes) --------
            if self.episode % EXAM_EVERY == 0:
                self._exam()

            # ---- progress report ------------------------------------
            if not self.quiet and self.episode % LOG_EVERY == 0:
                self._report()

        return ran

    def run_all(self):
        """Run the whole planned session in one go."""
        return self.run_episodes(self.episodes - self.episode)

    def skip_stage(self):
        """Fast-forward to the next stage (or the end). Returns episodes run."""
        before = self.stage
        run = 0
        while self.stage == before and not self.done:
            run += self.run_episodes(1)
        return run

    def _exam(self):
        """Short GREEDY exam — no learning, no exploration — and if
        this brain beats every previous one, it becomes the
        diploma (finish() will save it instead of the final brain).

        Exams exist because training oscillates: the LAST brain of
        a run is often in a dip, while one from 3,000 episodes ago
        was near a peak. (evaluate() lives below the class; Python
        resolves the name when we call, not when we define.)
        """
        avg, win = evaluate(self.agent, episodes=EXAM_EPISODES)
        score = (win, avg)
        if score > self.best_score:
            self.best_score = score
            self.best_episode = self.episode
            # Snapshot just the scorecard — hyperparameters and the
            # binner recipe stay identical (same agent, same brain).
            self.best_q = {k: v[:] for k, v in self.agent.q_table.items()}

    def _report(self):
        recent = self.history["steps"][-LOG_EVERY:]
        limits = sum(1 for o in self.history["outcome"][-LOG_EVERY:]
                     if o == "time_limit")
        avg = sum(recent) / len(recent)
        print(f"ep {self.episode:>5}/{self.episodes} | "
              f"avg steps {avg:>6.1f} | "
              f"hit limit {limits:>3}/{LOG_EVERY} | "
              f"epsilon {self.agent.epsilon:.3f} | "
              f"{STAGE_NAMES[self.stage]}")

    # -- saving -------------------------------------------------------
    def finish(self, q_path=None, log_path=None):
        """Save brain + training diary. Returns (q_path, log_path).

        Saves the DIPLOMA (best-exam scorecard) whenever one
        exists — not necessarily the final brain, which may be
        caught in a dip. One exception runs first: if episodes ran
        since the last exam (say, a run of 537), we take one final
        exam so nothing trained goes unjudged.

        LIVE mode calls this at every stage boundary and when you
        leave the mode, so the saved agent is never lost mid-run.
        """
        if self.episode and self.episode % EXAM_EVERY != 0:
            self._exam()

        q_path = SAVE_Q if q_path is None else q_path
        log_path = SAVE_LOG if log_path is None else log_path

        # Temporarily swap in the diploma scorecard for saving,
        # then put the live one back (LIVE mode keeps training!).
        live_q = self.agent.q_table
        if self.best_q is not None:
            self.agent.q_table = self.best_q
        self.agent.save(q_path)
        self.agent.q_table = live_q

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=1)
        return q_path, log_path

    def summary(self):
        """The end-of-training printout (a.k.a. the report card)."""
        elapsed = time.time() - self.started
        total_buckets = self.agent.binner.n_states
        visited = len(self.agent.q_table)
        diploma = ""
        if self.best_q is not None:
            diploma = (f"diploma: best exam was episode "
                       f"{self.best_episode} -> "
                       f"{self.best_score[0]:.0f}% limit hits / "
                       f"{self.best_score[1]:.0f} avg steps "
                       f"(THIS brain got saved)\n")
        return (f"\ntrained in {elapsed:.1f}s\n"
                f"last 100 episodes: avg steps {self.avg_steps():.1f}, "
                f"hit the {MAX_STEPS}-step limit {self.win_rate():.0f}%\n"
                f"best 100-episode stretch: {self.best_avg_steps():.1f} "
                f"avg steps\n"
                f"{diploma}"
                f"notebook filled: {visited} buckets visited "
                f"(of {total_buckets:,})")


def train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
          epsilon_start=EPSILON_START, quiet=False):
    """Run a full training session. Returns (agent, history).

    Thin wrapper around Trainer — kept because it's the friendliest
    way to grab a trained agent in two lines of code (play.py's
    safety net calls this on a first run).
    """
    trainer = Trainer(episodes=episodes, alpha=alpha, gamma=gamma,
                      epsilon_start=epsilon_start, quiet=quiet)
    trainer.run_all()
    if not quiet:
        print(trainer.summary())
    return trainer.agent, trainer.history


def evaluate(agent, episodes=100):
    """Pure performance check: explore=False, no learning, just play.

    Training mixes in random moves on purpose (that's exploration);
    evaluation switches it off so the numbers show the REAL skill.
    Returns (avg_steps, win_pct) where a win = reaching the
    500-step time limit.
    """
    env = CartPole()
    steps, wins = [], []
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose_action(state, explore=False)
            state, reward, done, info = env.step(action)
            if done:
                break
        steps.append(info["steps"])
        wins.append(info["reason"] == "time_limit")
    return sum(steps) / episodes, sum(wins) / episodes * 100


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run_all()
    trainer.finish()

    print(f"saved {SAVE_Q} and {SAVE_LOG}")
    print(trainer.summary())

    # How does the SAVED brain perform — the diploma play.py will
    # actually load, read back off the disk, training wheels off?
    graduate = QAgent.load(SAVE_Q)
    avg, win_pct = evaluate(graduate, episodes=100)
    print(f"\ngreedy evaluation of the saved brain (no exploration):")
    print(f"  avg steps {avg:.1f}   hit the {MAX_STEPS}-step "
          f"limit {win_pct:.0f}%")
