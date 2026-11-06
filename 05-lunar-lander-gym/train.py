"""
train.py — THE SCHOOL (step 3 of the lunar-lander-gym project)
==============================================================

The loop everyone came to see, running the OTHER brain family:
the caller files one reward per tick, the policy networks sample
their own exploration, and ONE gradient step per episode teaches
the whole thing. The reference loop, ported verbatim in arithmetic
(reference: github.com/nikhilbarhate99/Actor-Critic-PyTorch):

    for every episode:
        state = env.reset()
        for every tick until done:
            action = policy.act(state)          # samples + remembers
            state, reward, done, info = env.step(action)
            policy.rewards.append(reward)       # the caller files it
        loss = policy.calculate_loss(gamma)     # whole-episode loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        policy.clear_memory()

THE SOLVE CHECK (the reference's exact, quirky arithmetic):
    one accumulator `running_reward` SYs up across episodes; every
    20th episode it is divided by 20 (printed as the window
    average) and RESET to zero. The solve test fires EVERY episode,
    not just at the window border:

        running_reward > 4000

    i.e. the SUM of the episodes of the current (partial!) window
    passes 4000 — with a full 20-episode window that is exactly the
    classic Gym bar "average return > 200", but a few hot episodes
    can tip it EARLY, mid-window. We keep the quirk slavishly: the
    reference's numbers are the target, and its bar is that loop's
    arithmetic, not a paraphrase of it.

THE DIALS — all of them the reference's own:
    gamma  0.99            the discount inside calculate_loss
    lr     0.02            Adam's step (100x a DQN's; policy
                           gradients get one noisy sum per episode
                           and need a big, well-conditioned step)
    betas  (0.9, 0.999)    Adam's momenta, the reference's pair
    seed   543             drives torch (init + sampling) AND the
                           world's first reset (the stream continues)
    max    10000 episodes  the reference's own loop cap; the solve
                           bar is expected ~1,500 in its README.

THE EXAMS (our addition, MEASUREMENT ONLY — zero training logic):
    every EXAM_EVERY episodes the CURRENT brain sits EXAM_EPISODES
    fresh spawns in GREEDY mode (act_greedy: argmax, no dice, no
    memory pollution, no RNG consumed — exams cannot wiggle the
    torch stream, pinned in test_train.py). The BEST exam's weights
    are kept aside as the diploma, because 02-04's hardest lesson
    stands: NEVER TRUST THE LAST BRAIN. (If the run SOLVES, the
    solved weights save as the reference's artifact instead —
    solver beats examiner.)

ARTIFACTS (both git-ignored; recreate with `python train.py`):
    training_log.json     the diary: config, every episode, exams
    actor_critic_model.pt the diploma (best-exam or solved weights)

Try it (a 40-episode smoke run, ~half a minute on CPU):

    python train.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import copy     # deep-copy the BEST exam's weights out of harm's way
import json     # the training diary
import os       # artifacts land next to this file, repo-portable
import time     # the run's wall-clock bill in the summary

import torch
import torch.optim as optim   # the reference constructs Adam OUTSIDE
                              # the net (model.py knows no optimizer)
                              # — the split is kept on purpose

from environment import LunarLanderEnv, DEATH_REASONS
from agent import ActorCritic, DEFAULT_GAMMA, DEFAULT_LR

# The two artifacts (both git-ignored; recreate with python train.py).
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "actor_critic_model.pt")
LOG_PATH = os.path.join(HERE, "training_log.json")

# ==============================================================
# THE SCHOOL'S DIALS (the reference's own numbers on our knobs)
# ==============================================================
EPISODES      = 10000        # the reference's loop ceiling
SOLVE_SUM_BAR = 4000.0       # running_reward > 4000: the SUM over
                             # the current (up to 20) episode window;
                             # full window == classic "avg > 200"
WINDOW        = 20           # its print/reset cadence
GAMMA         = DEFAULT_GAMMA   # 0.99 — single import, no magic
LR            = DEFAULT_LR      # 0.02
BETAS         = (0.9, 0.999)    # Adam's momenta (the reference's pair)
SEED          = 543             # torch init+sampling AND world reset
EXAM_EVERY    = 50              # greedy exam cadence (0 = exams off)
EXAM_EPISODES = 20              # exam length: 20 fresh spawns


class Trainer:
    """Runs episodes, files rewards, holds the exams, and writes the
    diploma + the diary. The training loop itself is the reference's,
    hand-copied (see the module docstring)."""

    def __init__(self, episodes=EPISODES, seed=SEED,
                 gamma=GAMMA, lr=LR, betas=BETAS,
                 solve_sum_bar=SOLVE_SUM_BAR, window=WINDOW,
                 exam_every=EXAM_EVERY, exam_episodes=EXAM_EPISODES,
                 log_path=LOG_PATH, model_path=MODEL_PATH):
        """Wire a world + brain + optimizer; nothing has flown yet.

        Args: episodes — max training episodes (solve may break
                earlier; 10,000 is the reference's own ceiling).
              seed — torch manual_seed AND the world's first
                reset seed. Same seed -> same school, everywhere.
              gamma / lr / betas — the net+optimizer dials (tests
                shrink nothing here; they pin their PRODUCTS).
              solve_sum_bar / window — the solve check's arithmetic
                (the reference's 4000 / 20; a test drives a fake
                hot window through the same code).
              exam_every / exam_episodes — measurement-only knobs
                (0 disables exams; the solve check still runs).
              log_path / model_path — redirect both artifacts
                (tests write them into a temp dir).
        """
        self.episodes = episodes
        self.seed = seed
        self.gamma = gamma
        self.lr = lr
        self.betas = tuple(betas)
        self.solve_sum_bar = solve_sum_bar
        self.window = window
        self.exam_every = exam_every
        self.exam_episodes = exam_episodes
        self.log_path = log_path
        self.model_path = model_path

        # Same seed -> same school. The world seeds its FIRST reset
        # (the stream continues); the net seeds torch's stream, which
        # drives weight init AND every Categorical.sample in act().
        self.env = LunarLanderEnv(seed=seed)
        self.policy = ActorCritic(gamma=gamma, seed=seed)

        # Adam lives up here, not in the net — the reference's split,
        # kept: calculate_loss only RETURNS a loss tensor; the loop
        # zero_grads/descends/steps. (An optimizer inside the net
        # would cut the loss away from being inspectable.)
        self.optimizer = optim.Adam(self.policy.parameters(),
                                    lr=lr, betas=self.betas)

        # The reference's running accumulator and the quiet state
        # the diary wants:
        self.running_reward = 0.0   # the window's episode-return SUM
        self.solved = False         # flipped by the solve check
        self.solved_episode = None  # where it flipped (1-based)

        # The diary that outlives training:
        self.log_episodes = []   # one record per training episode
        self.log_exams = []      # one record per greedy exam
        self.best = None         # best exam record (dict) so far
        self._best_weights = None  # copied-aside state at that exam

    # ----------------------------------------------------------
    # 1. one training episode — the loop everyone came to see
    # ----------------------------------------------------------
    def train_episode(self):
        """Fly ONE episode; return its record for the diary.

        The exact reference loop, commented at the seams:
            act -> step -> file the reward -> repeat,
            then ONE calculate_loss/zero_grad/backward/step and a
            clear_memory. Nothing else — no replay diary, no target
            net, no epsilon: sampling IS the exploration.
        """
        self.policy.clear_memory()          # fresh slate per episode
        state = self.env.reset()
        ret = 0.0
        while True:
            # act() samples from pi(a|s) AND remembers log-prob and
            # the critic's value (the two handles the loss needs):
            action = self.policy.act(state)
            state, reward, done, info = self.env.step(action)
            # the caller files the reward — the reference's
            # `policy.rewards.append(reward)`, verbatim:
            self.policy.rewards.append(reward)
            self.running_reward += reward    # the window accumulator
            ret += reward
            if done:
                break
        outcome = info["outcome"]

        # --- the ONE gradient step of the episode -----------------
        self.optimizer.zero_grad()
        loss = self.policy.calculate_loss(self.gamma)
        loss.backward()
        self.optimizer.step()
        self.policy.clear_memory()          # between-episodes hygiene

        return {
            "episode": len(self.log_episodes) + 1,
            "return": ret,
            "steps": self.env.steps,
            "outcome": outcome,
            "loss": float(loss.item()),
        }

    # ----------------------------------------------------------
    # 2. the greedy exam — no dice, no memory, no RNG spent
    # ----------------------------------------------------------
    def greedy_exam(self):
        """Sit the CURRENT brain exam_episodes fresh spawns, greedy.

        act_greedy: argmax of the policy's probabilities. Strictly
        a no-op for learning: no log-probs appended, no values
        appended, no_grad for the reads, and (because argmax rolls
        no dice) the torch sampler's stream is untouched — so an
        exam in the middle of training CANNOT change the episodes
        after it (pinned by test_train.py's stream-freeze check).
        """
        returns, steps_all = [], []
        landed = 0
        for _ in range(self.exam_episodes):
            state = self.env.reset()
            done = False
            ret = 0.0
            while not done:
                action = self.policy.act_greedy(state)
                state, reward, done, info = self.env.step(action)
                ret += reward
            returns.append(ret)
            steps_all.append(self.env.steps)
            landed += 1 if info["outcome"] == "landed" else 0

        n = float(self.exam_episodes)
        return {
            "episode": len(self.log_episodes),   # exams report their
            # training-time position (0-based index into the diary)
            "avg_return": sum(returns) / n,
            "land_rate": landed / n,
            "avg_steps": sum(steps_all) / n,
        }

    def _remember_best(self, exam):
        """Keep the BEST exam so far — by avg return, ties keep the
        EARLIER exam (stable, kinder to replayability; 04's rule)."""
        if (self.best is None
                or exam["avg_return"] > self.best["avg_return"]):
            self.best = exam
            # Copy the weights NOW — training never stands still.
            self._best_weights = copy.deepcopy(
                self.policy.state_dict())

    # ----------------------------------------------------------
    # 3. run() — the whole school day
    # ----------------------------------------------------------
    def run(self):
        """episodes x (train, exam, solve-check) -> the artifacts.

        Returns the summary dict. Prints a window-average line
        every 20 episodes (the reference's cadence, ASCII only).
        """
        t0 = time.time()
        for episode in range(1, self.episodes + 1):
            record = self.train_episode()
            self.log_episodes.append(record)

            # ---- the SOLVE CHECK (the reference's, verbatim in
            # arithmetic): tested EVERY episode against the sum
            # of the current window; the window prints/resets every
            # `window` episodes. The check precedes the reset —
            # that ordering IS the quirk we port on purpose.
            if self.running_reward > self.solve_sum_bar:
                self.solved = True
                self.solved_episode = episode
                print("########## SOLVED! ##########")
                print(f"window sum {self.running_reward:.2f} "
                      f"> bar {self.solve_sum_bar:.0f} "
                      f"at episode {episode}")
                break   # the reference breaks (and saves) — same

            # ---- the window's print/reset border ----------------
            if episode % self.window == 0:
                avg = self.running_reward / self.window
                last_rec = self.log_episodes[-1]
                print(f"ep {episode:>5}  window avg return {avg:+8.2f}"
                      f"  last {last_rec['outcome']}")
                self.running_reward = 0.0   # next window starts

            # ---- our measurement-only exam ----------------------
            if self.exam_every and episode % self.exam_every == 0:
                exam = self.greedy_exam()
                self.log_exams.append(exam)
                self._remember_best(exam)
                print(f"  [exam @ ep {episode}] "
                      f"avg {exam['avg_return']:+8.2f}  "
                      f"land {exam['land_rate']:4.0%}")

        # The final exam runs even if the count missed a multiple
        # of exam_every — a school always ends with a last look.
        final = self.greedy_exam()
        final["episode"] = len(self.log_episodes)
        final["final"] = True
        self.log_exams.append(final)
        self._remember_best(final)

        return self.finish(seconds=time.time() - t0)

    # ----------------------------------------------------------
    # 4. finish() — the diploma and the diary
    # ----------------------------------------------------------
    def finish(self, seconds=None):
        """Write both artifacts; seal the diploma with the best
        weights (solve beats exam; default exam beats none)."""
        # Outcome tallies for the charts (gym's ending mix):
        outcomes = {r: 0 for r in DEATH_REASONS}
        for rec in self.log_episodes:
            outcomes[rec["outcome"]] += 1

        n_eps = len(self.log_episodes)
        window_n = min(100, n_eps)
        first100 = [r["return"] for r in self.log_episodes[:window_n]]
        last100 = [r["return"] for r in self.log_episodes[-window_n:]]

        summary = {
            "seed": self.seed,
            "episodes": n_eps,
            "seconds": round(seconds, 1) if seconds is not None
                       else None,
            "exams": len(self.log_exams),
            "best_exam": self.best,
            "solved": self.solved,
            "solved_episode": self.solved_episode,
            "outcome_mix": outcomes,
            "avg_return_first_100": sum(first100) / window_n,
            "avg_return_last_100": sum(last100) / window_n,
        }
        log = {
            "config": {
                "episodes": self.episodes,
                "seed": self.seed,
                "gamma": self.gamma,
                "lr": self.lr,
                "betas": list(self.betas),
                "solve_sum_bar": self.solve_sum_bar,
                "window": self.window,
                "exam_every": self.exam_every,
                "exam_episodes": self.exam_episodes,
            },
            "episodes": self.log_episodes,
            "exams": self.log_exams,
            "best": self.best,
            "summary": summary,
        }
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)

        # The diploma: the SOLVED weights if the run solved (the
        # reference's artifact), else the BEST exam's (our never-
        # trust-the-last-brain lesson; a run that never solved and
        # never examined still saves CURRENT weights via run()'s
        # final exam, or the best-so-far exam). Either way the file
        # is save()'s envelope, so play.py/visualize.py load blind.
        if self.solved:
            self.policy.save(self.model_path)
        else:
            assert self._best_weights is not None, \
                "no exam ever ran - run() must always end with one"
            self.policy.load_state_dict(self._best_weights)
            self.policy.save(self.model_path)
        return summary


# ==============================================================
# THE OFFICIAL RUN
# ==============================================================
if __name__ == "__main__":
    trainer = Trainer()
    print(f"=== the school: {EPISODES} episodes max, seed {SEED}, "
          f"gamma {GAMMA}, lr {LR}, betas {BETAS} ===")
    summary = trainer.run()

    # The official report — ASCII only, numbers a human reads first.
    print("\n=== report card ===")
    print(f"episodes    : {summary['episodes']}")
    print(f"seconds     : {summary['seconds']}")
    print(f"solved      : {summary['solved']}"
          + (f" at ep {summary['solved_episode']}"
             if summary["solved"] else ""))
    print(f"started     : {summary['avg_return_first_100']:+.2f} "
          f"(first 100 episodes)")
    print(f"finished    : {summary['avg_return_last_100']:+.2f} "
          f"(last 100 episodes)")
    b = summary["best_exam"]
    if b is not None:
        print(f"best exam   : ep {b['episode']} -> "
              f"avg {b['avg_return']:+.2f}, "
              f"land {b['land_rate']:.0%}")
    print(f"outcome mix : {summary['outcome_mix']}")
    print(f"diary       : {LOG_PATH}")
    print(f"diploma     : {MODEL_PATH}")
