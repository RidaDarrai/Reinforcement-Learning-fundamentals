"""
train.py — THE SCHOOL (step 3 of our RL project)
=================================================

Step 1 gave the world (environment.py).
Step 2 gave an empty brain (agent.py).
Step 3 puts them together and fills the notebook:

    for each episode:
        start at S
        while not finished:
            action  = agent.choose_action(state)   # decide
            state', reward, done, info = env.step(action)   # world answers
            agent.update(state, action, reward, state', done)  # write
        remember the episode's total reward
        shrink epsilon a little (explore less as we know more)

That inner loop IS reinforcement learning. Everything else is
bookkeeping: logging, charts-data, and saving the result.

The loop lives in the Trainer class below, so it can be driven
step by step: this file's command line runs it all at once (fast,
silent), while play.py's LIVE mode feeds it a few episodes per
frame so you can WATCH learning happen.

Two files come out of training:

    q_table.json        the trained brain (loaded later by play.py)
    training_log.json   every episode's score + outcome — the raw
                        material for step 5's charts

It trains whichever maze is ACTIVE (classic on a first run, your
saved random maze after you press N in play.py):

    python train.py
"""

import json
import os
import time

from agent import QAgent
from environment import GridWorld, MAX_STEPS, load_active_maze

# Save next to THIS file (not whatever folder you launched python
# from) — play.py looks for q_table.json in exactly this location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# THE KNOBS — hyperparameters (change these, change everything)
# ---------------------------------------------------------------
EPISODES = 2000        # how many practice runs the agent gets
ALPHA = 0.1            # learning rate: rewrite 10% toward each new truth
GAMMA = 0.99           # discount: rewards 12 steps away still count ~89%
EPSILON_START = 1.0    # start as a pure explorer...
EPSILON_MIN = 0.05     # ...end as a 95% exploiter
EPSILON_DECAY = 0.995  # multiply epsilon by this after every episode
LOG_EVERY = 100        # print a progress line every N episodes
SAVE_Q = os.path.join(BASE_DIR, "q_table.json")
SAVE_LOG = os.path.join(BASE_DIR, "training_log.json")

# The three phases every training run passes through. Which phase you
# are in depends ONLY on epsilon (the exploration rate):
#   EXPLORING  eps > 0.5    mostly random moves — looking around
#   LEARNING   eps > 0.05   mixing finds with cashing in
#   POLISHING  eps = floor  almost pure exploitation — tuning paths
STAGE_NAMES = ("EXPLORING", "LEARNING", "POLISHING")


class Trainer:
    """One training run, advanced from the outside.

    The plain train() function below just calls run_all() once —
    but play.py's LIVE mode calls run_episodes(n) every frame,
    which is what lets you watch the agent improve stage by stage.
    """

    def __init__(self, maze_name=None, episodes=EPISODES, alpha=ALPHA,
                 gamma=GAMMA, epsilon_start=EPSILON_START, quiet=False):
        """Prepare a fresh run on the ACTIVE maze (or a named one)."""
        if maze_name is None:
            maze_name = load_active_maze()[0]
        self.maze_name = maze_name
        self.env = GridWorld(maze_name)
        self.episodes = episodes
        self.quiet = quiet
        self.agent = QAgent(n_actions=4, alpha=alpha, gamma=gamma,
                            epsilon=epsilon_start)
        self.started = time.time()

        # history is a dict of parallel lists — one entry per episode —
        # ready to be turned into charts in step 5. The maze itself is
        # stored alongside so the policy map draws the RIGHT maze later.
        self.history = {
            "episode": [],     # episode number
            "reward": [],      # total reward earned that episode
            "outcome": [],     # "goal" / "pit" / "timeout"
            "epsilon": [],     # exploration chance at episode start
            "steps": [],       # how many moves it took
            "maze": maze_name,
            "maze_rows": list(self.env.maze_rows),
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
        """% of the last n episodes that ended at the goal."""
        tail = self.history["outcome"][-n:]
        if not tail:
            return 0.0
        return 100.0 * sum(o == "goal" for o in tail) / len(tail)

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
                self.agent.update(state, action, reward,          # 3. learn
                                  next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            # -------------------------------------------------------

            # Shrink exploration: early episodes are for looking around,
            # later ones for cashing in what we found. Never below MIN —
            # a 5% chance of silliness keeps the agent honest forever.
            self.agent.epsilon = max(EPSILON_MIN,
                                     self.agent.epsilon * EPSILON_DECAY)

            self.history["episode"].append(self.episode + 1)
            self.history["reward"].append(total_reward)
            self.history["outcome"].append(info["reason"])
            self.history["epsilon"].append(self.agent.epsilon)
            self.history["steps"].append(info["steps"])
            ran += 1

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

    def _report(self):
        recent = self.history["reward"][-LOG_EVERY:]
        wins = sum(1 for o in self.history["outcome"][-LOG_EVERY:]
                   if o == "goal")
        avg = sum(recent) / len(recent)
        print(f"ep {self.episode:>5}/{self.episodes} | "
              f"avg reward {avg:+6.2f} | "
              f"wins {wins:>3}/{LOG_EVERY} | "
              f"epsilon {self.agent.epsilon:.3f}")

    # -- saving -------------------------------------------------------
    def finish(self, q_path=None, log_path=None):
        """Save brain + training diary. Returns (q_path, log_path).

        LIVE mode calls this at every stage boundary and when you
        leave the mode, so the saved agent is never lost mid-run.
        """
        q_path = SAVE_Q if q_path is None else q_path
        log_path = SAVE_LOG if log_path is None else log_path
        self.agent.save(q_path)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=1)
        return q_path, log_path

    def summary(self):
        """The end-of-training printout (a.k.a. the report card)."""
        last100 = self.history["reward"][-100:]
        elapsed = time.time() - self.started
        return (f"\ntrained in {elapsed:.1f}s\n"
                f"last 100 episodes: avg reward {sum(last100)/len(last100):+.2f}, "
                f"win rate {self.win_rate():.0f}%\n"
                f"notebook filled: {len(self.agent.q_table)} positions visited")


def train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
          epsilon_start=EPSILON_START, quiet=False, maze_name=None):
    """Run a full training session. Returns (agent, history).

    Thin wrapper around Trainer — kept because it's the friendliest
    way to grab a trained agent in two lines of code.
    """
    trainer = Trainer(maze_name=maze_name, episodes=episodes, alpha=alpha,
                      gamma=gamma, epsilon_start=epsilon_start, quiet=quiet)
    trainer.run_all()
    if not quiet:
        print(trainer.summary())
    return trainer.agent, trainer.history


def evaluate(agent, episodes=100, maze_name=None):
    """Pure performance check: explore=False, no learning, just play.

    Training mixes in random moves on purpose (that's exploration);
    evaluation switches it off so the numbers show the REAL skill.
    Returns (avg_reward, win_rate).
    """
    if maze_name is None:
        maze_name = load_active_maze()[0]
    env = GridWorld(maze_name)
    rewards, wins = [], []
    for _ in range(episodes):
        state = env.reset()
        total = 0.0
        while True:
            action = agent.choose_action(state, explore=False)
            state, reward, done, info = env.step(action)
            total += reward
            if done:
                break
        rewards.append(total)
        wins.append(info["reason"] == "goal")
    return sum(rewards) / episodes, sum(wins) / episodes * 100


if __name__ == "__main__":
    trainer = Trainer()          # trains whichever maze is active
    trainer.run_all()
    trainer.finish()

    print(f"saved {SAVE_Q} and {SAVE_LOG} "
          f"(maze: {trainer.maze_name})")
    print(trainer.summary())

    # How does the GRADUATE perform with the training wheels off?
    avg, win_pct = evaluate(trainer.agent, episodes=100)
    print(f"\ngreedy evaluation (no exploration):")
    print(f"  avg reward {avg:+.2f}   win rate {win_pct:.0f}%")
