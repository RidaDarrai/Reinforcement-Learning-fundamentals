"""
train.py — THE SCHOOL (step 3 of the lunar lander project)
==========================================================

The recipe's third character: the loop that runs episodes, files
every tick into the DQN's replay diary, slowly replaces luck with
policy, and — every EXAM_EVERY episodes — sits the brain a GREEDY
EXAM. The best exam's weights are copied aside and saved as the
diploma (dqn_model.pt), because 02's and 03's hardest-won lesson
applies even harder here: NEVER TRUST THE LAST BRAIN; trust the
best exam. Curriculum:

    1. EXPLORE     epsilon 1.0  -> random buttons, chaos everywhere
    2. DECAY       epsilon  -> 0.05 linearly over the first 1,000
                   episodes; the world starts teaching its rules
    3. POLISH      epsilon stays 0.05; the net sharpens what it knows
    (Exams every 50 episodes track where the real progress is.)

THE METRIC IS NEW (vs cart-pole): reward is no longer "steps
survived". An episode's return is terminal + shaping + fuel — the
sum environment.py reports tick by tick. The train.py invariant
ret == sum(components) is asserted in test_train.py, and every
component travels SEPARATELY to training_log.json so step 5's
charts can decompose where the money comes from.

DIPLOMA (best-exam weights): saved to dqn_model.pt next to this
file; the raw training diary goes to training_log.json. Both are
git-ignored artifacts — recreate with `python train.py`.

Official run (seed 42): ~2,000-3,000 episodes, tens of minutes of
CPU, tens of thousands of gradient steps. Tune nothing during one
run: step 5 reads the curves, and the noise does the talking.

    python train.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import copy   # deep-copy the BEST exam's weights before training moves on
import json   # write the training diary (training_log.json)
import math   # isfinite guards, means over exam episodes
import os     # artifacts land next to this file, repo-portable
import time   # the official run's wall-clock bill (summary field)

# The world and the brain — both in THIS folder, no cross-project
# imports (project independence rule).
from environment import LunarLander, DEATH_REASONS
from agent import DQNAgent

# The two artifacts (both git-ignored; recreate with python train.py).
HERE = os.path.dirname(os.path.abspath(__file__))
DIPLOMA_PATH = os.path.join(HERE, "dqn_model.pt")
LOG_PATH = os.path.join(HERE, "training_log.json")

# ==============================================================
# THE SCHOOL'S DIALS (defaults = the official run; tests shrink them)
# ==============================================================
EPISODES = 3000              # one episode = one full attempt at landing
EPSILON_DECAY_EPISODES = 300  # 1.0 -> floor over this window (linear)
                             # RUN 3 POSTMORTEM: this was 1000, and the
                             # final brain DEGENERATED — greedy argmax
                             # (Q spread ~0.3, under the MLP's own
                             # fitting noise) locked into "push NO
                             # buttons ever": 96% IDLE, vy pinned at
                             # free-fall 14.9 m/s, tilt frozen at 0.0,
                             # 0 landings in 3000 episodes. The seed-7
                             # probe with THIS window (300) instead
                             # committed early and BUILT its own
                             # braking corpus: median fatal speed
                             # 14.7 -> 10.1 m/s, tilt down to 33 deg,
                             # 14% MAIN near the ground, hull crashes
                             # 46% -> 18%. Slow decay just filled the
                             # replay with random meteors for 1000
                             # episodes and then fit the danger map
                             # harder; fast decay gives the policy
                             # autonomy while it can still invent the
                             # gentle phase from its own experiences.
EPSILON_FLOOR = 0.05         # keeps a pinch of curiosity forever
EXAM_EVERY = 50              # greedy exam cadence (training episodes)
EXAM_EPISODES = 25           # exam length: 25 fresh spawns, no luck
SEED = 42                    # the official run's seed (twins repeat it)


class Trainer:
    """Runs episodes, files ticks into the DQN, holds the exams,
    tracks the BEST one, and writes the diploma + the diary."""

    def __init__(self, episodes=EPISODES,
                 epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
                 epsilon_floor=EPSILON_FLOOR,
                 exam_every=EXAM_EVERY, exam_episodes=EXAM_EPISODES,
                 seed=SEED, log_path=LOG_PATH, diploma_path=DIPLOMA_PATH):
        """Wire a fresh world + brain pair; nothing has flown yet.

        Args: the dials above. Tests shrink every one (episodes=24,
              exam_every=12...) — the loop must not care.
              log_path / diploma_path — redirect both artifacts
              (tests write them into a temp dir).
              seed — drives BOTH the world's spawn scatter (its
              private RNG) AND the brain (weight init, epsilon coin
              and buffer sampling via torch/python seeds). Same seed
              -> same school, everywhere.
        """
        self.episodes = episodes
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon_floor = epsilon_floor
        self.exam_every = exam_every
        self.exam_episodes = exam_episodes
        self.seed = seed

        self.env = LunarLander(seed=seed)
        self.agent = DQNAgent(seed=seed)

        # The diary that outlives training:
        self.log_episodes = []   # one record per training episode
        self.log_exams = []      # one record per greedy exam
        self.best = None         # best exam record (dict) so far
        self._best_weights = None  # copied-aside state at that exam
        self.log_path = log_path
        self.diploma_path = diploma_path

    # ----------------------------------------------------------
    # 0. epsilon schedule — 1.0 down to the floor, never back up
    # ----------------------------------------------------------
    def epsilon_at(self, episode):
        """Epsilon for a training episode INDEX (0-based).

        Linear 1.0 -> floor across the first epsilon_decay_episodes,
        then the floor forever. At the boundary it is EXACTLY the
        floor (episode == window), so decay has no loose ends.
        """
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_floor
        span = 1.0 - self.epsilon_floor
        return 1.0 - span * episode / self.epsilon_decay_episodes

    # ----------------------------------------------------------
    # 1. one training episode — the loop everyone came to see
    # ----------------------------------------------------------
    def train_episode(self):
        """Fly ONE episode with an exploring brain and record it.

        The canonical RL loop, unembellished: act (epsilon-greedy),
        step the world, file the tick into the replay diary (which
        takes ONE gradient step once warmed). Returns the episode's
        record dict — flat keys, so the JSON diary and the step-5
        charts can read it without unpacking nests.
        """
        state = self.env.reset()
        done = False
        ret = 0.0
        losses = []                      # gradient steps taken here
        terminal_sum = shaping_sum = fuel_sum = 0.0

        while not done:
            action = self.agent.choose_action(state, explore=True)
            next_state, reward, done, info = self.env.step(action)
            loss = self.agent.update(state, action, reward,
                                     next_state, done)
            if loss is not None:
                losses.append(loss)
            ret += reward
            comps = info["components"]   # the money, per currency
            terminal_sum += comps["terminal"]
            shaping_sum += comps["shaping"]
            fuel_sum += comps["fuel"]
            state = next_state

        return {
            "return": ret,
            "steps": self.env.steps,
            "outcome": info["reason"],               # landed/crashed/...
            "crash_type": info.get("crash_type"),    # impact/hull/missed
            "fuel": float(info["fuel_used"]),
            "terminal": terminal_sum,
            "shaping": shaping_sum,
            "fuel_cost": fuel_sum,
            "avg_loss": (sum(losses) / len(losses)) if losses else None,
            "buffer": len(self.agent.buffer),
        }

    # ----------------------------------------------------------
    # 2. the greedy exam — no coin flips allowed
    # ----------------------------------------------------------
    def greedy_exam(self):
        """Sit the CURRENT brain EXAM_EPISODES exam-spawns, greedy only.

        Training is paused (world + brain untouched); the exam flies
        fresh spawns with explore=False and measures: average return,
        landing rate, average steps. Nothing learns here — asserted
        in test_exam_is_greedy_and_frozen (weights do not move).
        """
        returns, steps_all = [], []
        landed = 0
        for _ in range(self.exam_episodes):
            state = self.env.reset()
            done = False
            ret = 0.0
            while not done:
                action = self.agent.choose_action(state, explore=False)
                state, reward, done, info = self.env.step(action)
                ret += reward
            returns.append(ret)
            steps_all.append(self.env.steps)
            landed += 1 if info["reason"] == "landed" else 0

        n = float(self.exam_episodes)
        return {
            "episode": len(self.log_episodes),  # exams report their
                                                # training-time position
            "avg_return": sum(returns) / n,
            "land_rate": landed / n,
            "avg_steps": sum(steps_all) / n,
        }

    def _remember_best(self, exam):
        """Keep the BEST exam so far — by average return, ties keep
        the EARLIER exam (stable, and kinder to replayability)."""
        if (self.best is None
                or exam["avg_return"] > self.best["avg_return"]):
            self.best = exam
            # Copy the weights NOW — training never stands still, and
            # the diploma must be the brain at its best moment.
            self._best_weights = {
                "online": copy.deepcopy(self.agent.online.state_dict()),
                "target": copy.deepcopy(self.agent.target.state_dict()),
            }

    # ----------------------------------------------------------
    # 3. run() — the whole school day
    # ----------------------------------------------------------
    def run(self):
        """episodes x (set epsilon, train, examine) -> artifacts.

        Writes training_log.json (every record) and dqn_model.pt
        (the BEST exam's weights), prints progress a human can
        follow, and returns the summary dict. Every 25 episodes a
        line — ASCII only, 02's console-portability lesson.
        """
        t0 = time.time()
        for ep in range(self.episodes):
            self.agent.epsilon = self.epsilon_at(ep)
            record = self.train_episode()
            record["episode"] = ep + 1
            self.log_episodes.append(record)

            if (ep + 1) % self.exam_every == 0:
                exam = self.greedy_exam()
                self.log_exams.append(exam)
                self._remember_best(exam)
                print(f"  [exam @ ep {ep + 1}] avg {exam['avg_return']:+8.2f}"
                      f"  land {exam['land_rate']:4.0%}"
                      f"  steps {exam['avg_steps']:6.1f}")
            if (ep + 1) % 25 == 0:
                r = record["return"]
                print(f"  ep {ep + 1:>5}  return {r:+9.2f}"
                      f"  eps {self.agent.epsilon:.2f}"
                      f"  out {record['outcome']}")

        # The final exam runs even if the count misses a multiple of
        # exam_every — a school always ends with a last look.
        final_exam = self.greedy_exam()
        final_exam["episode"] = self.episodes
        final_exam["final"] = True
        self.log_exams.append(final_exam)
        self._remember_best(final_exam)

        summary = self.finish(seconds=time.time() - t0)
        return summary

    # ----------------------------------------------------------
    # 4. finish() — the diploma and the diary
    # ----------------------------------------------------------
    def finish(self, seconds=None):
        """Write both artifacts; seal the diploma with BEST weights."""
        # Outcome tallies for the step-5 death chart:
        outcomes = {r: 0 for r in DEATH_REASONS}
        crash_types = {}
        for rec in self.log_episodes:
            outcomes[rec["outcome"]] += 1
            if rec["outcome"] == "crashed" and rec["crash_type"]:
                crash_types[rec["crash_type"]] = \
                    crash_types.get(rec["crash_type"], 0) + 1

        n_eps = len(self.log_episodes)
        window = min(100, n_eps)
        first100 = [r["return"] for r in self.log_episodes[:window]]
        last100 = [r["return"] for r in self.log_episodes[-window:]]

        summary = {
            "seed": self.seed,
            "episodes": n_eps,
            "seconds": round(seconds, 1) if seconds is not None else None,
            "exams": len(self.log_exams),
            "best_exam": self.best,
            "outcomes": outcomes,
            "crash_types": crash_types,
            "avg_return_first_100": sum(first100) / window,
            "avg_return_last_100": sum(last100) / window,
            "buffer_size": len(self.agent.buffer),
            "gradient_steps": self.agent.updates,
        }
        log = {
            "config": {
                "episodes": self.episodes,
                "epsilon_decay_episodes": self.epsilon_decay_episodes,
                "epsilon_floor": self.epsilon_floor,
                "exam_every": self.exam_every,
                "exam_episodes": self.exam_episodes,
                "seed": self.seed,
            },
            "episodes": self.log_episodes,
            "exams": self.log_exams,
            "best": self.best,
            "summary": summary,
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)

        # The diploma: restore the best moment, save it. If training
        # ended before ANY exam could run, the current weights are
        # the best we have (and self._remember_best handles that in
        # run(), so this restores _best_weights unconditionally).
        assert self._best_weights is not None, \
            "no exam ever ran - run() must always end with one"
        self.agent.online.load_state_dict(self._best_weights["online"])
        self.agent.target.load_state_dict(self._best_weights["target"])
        self.agent.save(self.diploma_path)
        return summary


# ==============================================================
# THE OFFICIAL RUN
# ==============================================================
if __name__ == "__main__":
    trainer = Trainer()
    print(f"=== the school: {EPISODES} episodes, seed {SEED}, "
          f"exams every {EXAM_EVERY} ===")
    summary = trainer.run()

    # The official report — numbers a human reads first, ASCII only.
    b = summary["best_exam"]
    print("\n=== report card ===")
    print(f"started     : {summary['avg_return_first_100']:+.2f} "
          f"(first {100} episodes)")
    print(f"finished    : {summary['avg_return_last_100']:+.2f} "
          f"(last {100} episodes)")
    print(f"best exam   : ep {b['episode']} -> avg {b['avg_return']:+.2f}, "
          f"land rate {b['land_rate']:.0%}, "
          f"steps {b['avg_steps']:.1f}")
    print(f"outcomes    : {summary['outcomes']}")
    print(f"crash types : {summary['crash_types']}")
    print(f"diary       : {LOG_PATH}")
    print(f"diploma     : {DIPLOMA_PATH}")
