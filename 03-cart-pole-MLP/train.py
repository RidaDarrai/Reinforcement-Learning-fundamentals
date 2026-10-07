"""
train.py — THE SCHOOL (step 3 of the cart-pole MLP project)
============================================================

Step 1 gave the world (environment.py).
Step 2 gave an empty brain (agent.py: MLP + replay + target net).
Step 3 puts them together and fills the weights:

    for each episode:
        start with 4 noisy floats
        while not finished:
            action  = agent.choose_action(state)        # decide
            state', reward, done, info = env.step(action)   # act
            loss = agent.update(state, action, taught, state', done)
        remember how long we survived
        shrink epsilon a little (explore less as we know more)

That inner loop IS reinforcement learning. Everything else is
bookkeeping: logging, greedy exams, and the diploma.

THE METRIC (same as 02 — read twice)
-------------------------------------
Cart-pole pays +1.0 for every tick alive, so:

    total reward  ==  steps survived   (ALWAYS, by design)
    a "perfect" episode = the 500-step time limit
    falling at step 12 scores 12 — failure and progress are
    the SAME axis here

So THE number to watch is **avg steps survived** (over the last
100 episodes). The learning curve IS the survival curve: it should
climb from ~20 toward 500.

WHAT DQN CHANGES vs 02's SCHOOL
--------------------------------
1. NO binner, NO q_init optimism. Exploration comes from epsilon
   alone (the MLP answers for every state from step 0 — randomly
   at first, then smarter). Fresh brains guess small random Qs.
2. LEARNING IS BATCHED + DELAYED. 02's update() rewrote one table
   row instantly. Here update() files the transition in the replay
   buffer and takes ONE gradient step on a random batch of 64 —
   and returns None until the buffer is warm (tests assert this).
3. SOFTER FALL BILL. 02 taught -30 on the fatal tick to break its
   ~100-step plateau. A -30 TD-error through an MLP is a hammer
   blow (big gradient, unstable tail). DQN generalizes across
   states, so a gentler FALL_PENALTY = -10.0 sharpens the ending
   without destabilizing. History/charts still record raw +1/tick.
4. FEWER EPISODES. A table memorizes 7,680 rows one visit at a
   time (15,000 episodes in 02). An MLP generalizes — one gradient
   step moves ALL similar states — so 1,500 episodes is the new
   15,000. Exams come every 100 episodes instead of 500.
5. THE DIPLOMA IS WEIGHTS, not rows. Training oscillates (same as
   02: a peak at ep 900 can be a dip by ep 1,200), so greedy exams
   still pick the saved brain — but the snapshot is a state_dict
   copy, not a dict of rows.

Two files come out of training:

    dqn_model.pt        the trained brain (loaded by play.py later)
    training_log.json   every episode's score + outcome — raw material
                        for step 5's charts

    python train.py
"""

# Every import earns its comment (repo rule: explain every file well).
import copy    # deepcopy state_dicts for the diploma snapshot
import json    # training_log.json: the diary step 5's charts read
import os      # save paths anchored next to THIS file (not the cwd)
import random  # seeding reproducible training runs
import time    # elapsed-seconds in the graduation summary

# The world and the brain (same folder — play.py will reuse both).
from agent import DQNAgent
from environment import CartPole, MAX_STEPS

# Save next to THIS file (not whatever folder you launched python
# from) — play.py looks for dqn_model.pt in exactly this location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# THE KNOBS — hyperparameters (change these, change everything)
# ---------------------------------------------------------------
EPISODES = 1500       # practice runs (DQN generalizes: 1,500 here ≈
                      # 15,000 tabular in 02 — each gradient step moves
                      # ALL similar states, not one bucket row)
ALPHA = 1e-3          # Adam learning rate (passed as lr= to DQNAgent)
GAMMA = 0.99          # discount: rewards 10 ticks away still count ~90%
EPSILON_START = 1.0   # start as a pure explorer...
EPSILON_MIN = 0.05    # ...end as a 95% exploiter
EPSILON_DECAY = 0.995  # multiply epsilon after every episode (hits the
                       # floor around episode ~600: 1.0*0.995^600 ≈ 0.05)
LOG_EVERY = 50        # print a progress line every N episodes

# -- DQN knobs (passed straight into DQNAgent — see agent.py) -------
BUFFER_SIZE = 50_000  # replay diary capacity (recent ~100 episodes)
BATCH_SIZE = 64       # transitions per gradient step
TARGET_SYNC = 500     # gradient steps between target-net copies
HIDDEN = 64           # MLP width (both hidden layers share it)

# -- reward shaping: how endings PAY --------------------------------
# The raw game pays +1 every tick and nothing at the end. 02 needed
# FALL_PENALTY = -30 to break its plateau; for an MLP that sized bill
# is a destabilizing hammer (one -30 TD-error yanks every similar
# state). -10 proved enough in our sweeps: the fatal tick still
# screams "avoid", but gradients stay calm. Charts/history always
# store the RAW +1/tick score; only what the brain is TAUGHT changes.
FALL_PENALTY = -10.0
TIME_LIMIT_BONUS = 0.0  # extra payment for surviving all MAX_STEPS.
                        # OFF: changed nothing measurable in 02 either.

# -- the graduation exam (the "best diploma" trick) -----------------
# Training oscillates: the LAST brain is often in a dip while one
# from hundreds of episodes ago sat on a peak. So every EXAM_EVERY
# episodes the agent sits a short GREEDY exam (no learning, no
# exploration) and finish() saves whichever brain scored best.
EXAM_EVERY = 100     # how often the exam comes around (100 x 15 exams)
EXAM_EPISODES = 20   # how long each exam lasts (20 keeps exams cheap)

SAVE_MODEL = os.path.join(BASE_DIR, "dqn_model.pt")
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
    but step 4's play.py LIVE mode will call run_episodes(n) every
    frame, which is what lets you WATCH learning happen.
    """

    def __init__(self, episodes=EPISODES, alpha=ALPHA,
                 gamma=GAMMA, epsilon_start=EPSILON_START, quiet=False,
                 seed=None, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, target_sync=TARGET_SYNC,
                 hidden=HIDDEN):
        """Prepare a fresh run: one world, one empty MLP brain."""
        # Seed first so the WORLD's reset noise is reproducible too
        # (agent.py separately seeds torch + its own random stream).
        if seed is not None:
            random.seed(seed)
        self.env = CartPole()
        self.episodes = episodes
        self.quiet = quiet
        self.seed = seed
        self.agent = DQNAgent(n_actions=2, hidden=hidden, lr=alpha,
                             gamma=gamma, epsilon=epsilon_start,
                             buffer_size=buffer_size,
                             batch_size=batch_size,
                             target_sync=target_sync, seed=seed)
        self.started = time.time()

        # The diploma cabinet: best GREEDY-exam result so far, and a
        # snapshot of the WEIGHTS that earned it (see _exam). Weights
        # need deepcopy — state_dict() alone is a live view that keeps
        # training would silently overwrite.
        self.best_score = (-1.0, -1.0)   # (win %, avg steps)
        self.best_episode = 0
        self.best_online = None
        self.best_target = None

        # history is a dict of parallel lists — one entry per episode —
        # ready for step 5's charts. NOTE: "reward" and "steps" are the
        # same number (+1 per tick); both kept because the chart says
        # "reward" and the physics says "steps" — test_train.py checks
        # they never drift apart.
        self.history = {
            "episode": [],     # episode number
            "reward": [],      # total reward earned that episode
            "outcome": [],     # "pole_fell" / "cart_out" / "time_limit"
            "epsilon": [],     # exploration chance after this episode
            "steps": [],       # how many ticks it took
            "loss": [],        # mean gradient-step loss (None = warming)
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

        That is THIS game's definition of a win: survive the full
        time limit. (A fall at step 400 is a much better loss than
        a fall at step 12 — avg_steps tracks that, win_rate tracks
        perfection.)
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
        training does not improve in a straight line, and we want
        to remember the peak even if the tail cooled off."""
        steps = self.history["steps"]
        if len(steps) <= n:
            return self.avg_steps(n)
        return max(sum(steps[i:i + n]) / n
                   for i in range(len(steps) - n + 1))

    # -- advancing the run -------------------------------------------
    def run_episodes(self, n):
        """Run up to n more episodes. Returns how many actually ran.

        Each episode is the same inner loop you would write by hand:
        choose, act, learn, remember — then shrink epsilon a little.
        """
        ran = 0
        for _ in range(n):
            if self.done:
                break

            state = self.env.reset()
            total_reward = 0.0
            losses = []  # gradient-step losses this episode (for logs)

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
                        taught += FALL_PENALTY        # e.g. 1.0 -> -9.0

                loss = self.agent.update(state, action, taught,  # 3. learn
                                         next_state, done)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_reward += reward       # RAW +1/tick only: the
                if done:                     # history/charts stay honest
                    break
            # -------------------------------------------------------

            # Shrink exploration: early episodes shake the cart with
            # near-random pushes; later ones cash in what was found.
            # Never below MIN — a 5% chance of silliness keeps the
            # agent honest forever.
            self.agent.epsilon = max(EPSILON_MIN,
                                     self.agent.epsilon * EPSILON_DECAY)

            self.history["episode"].append(self.episode + 1)
            self.history["reward"].append(total_reward)
            self.history["outcome"].append(info["reason"])
            self.history["epsilon"].append(self.agent.epsilon)
            self.history["steps"].append(info["steps"])
            # Mean loss or None while the buffer was still warming up
            # (charts draw gaps, not fake zeros, for warmup episodes).
            self.history["loss"].append(
                sum(losses) / len(losses) if losses else None)
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
        this brain beats every previous one, its WEIGHTS become the
        diploma (finish() will save them instead of the final brain).

        Exams exist because training oscillates: the LAST brain of
        a run is often in a dip, while one from hundreds of episodes
        ago sat near a peak. (evaluate() lives below the class.)
        """
        avg, win = evaluate(self.agent, episodes=EXAM_EPISODES)
        score = (win, avg)
        if score > self.best_score:
            self.best_score = score
            self.best_episode = self.episode
            # Snapshot the WEIGHTS (deepcopy — the live nets keep
            # training the very next episode and must not drag the
            # diploma along with them).
            self.best_online = copy.deepcopy(
                self.agent.online.state_dict())
            self.best_target = copy.deepcopy(
                self.agent.target.state_dict())

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
    def finish(self, model_path=None, log_path=None):
        """Save brain + training diary. Returns (model_path, log_path).

        Saves the DIPLOMA (best-exam weights) whenever one exists —
        not necessarily the final brain, which may be caught in a
        dip. One exception runs first: if episodes ran since the last
        exam, take one final exam so nothing trained goes unjudged.

        LIVE mode (step 4) calls this at every stage boundary and when
        you leave the mode, so the saved agent is never lost mid-run.
        """
        if self.episode and self.episode % EXAM_EVERY != 0:
            self._exam()

        model_path = SAVE_MODEL if model_path is None else model_path
        log_path = SAVE_LOG if log_path is None else log_path

        # Temporarily swap in the diploma weights for saving, then
        # put the live ones back (LIVE mode keeps training!).
        if self.best_online is not None:
            live_online = copy.deepcopy(self.agent.online.state_dict())
            live_target = copy.deepcopy(self.agent.target.state_dict())
            self.agent.online.load_state_dict(self.best_online)
            self.agent.target.load_state_dict(self.best_target)
            self.agent.save(model_path)
            self.agent.online.load_state_dict(live_online)
            self.agent.target.load_state_dict(live_target)
        else:
            self.agent.save(model_path)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=1)
        return model_path, log_path

    def summary(self):
        """The end-of-training printout (a.k.a. the report card)."""
        elapsed = time.time() - self.started
        diploma = ""
        if self.best_online is not None:
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
                f"replay buffer: {len(self.agent.buffer):,} stored | "
                f"gradient steps: {self.agent.updates:,}")


def train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
          epsilon_start=EPSILON_START, quiet=False, seed=None):
    """Run a full training session. Returns (agent, history).

    Thin wrapper around Trainer — kept because it is the friendliest
    way to grab a trained agent in two lines of code (play.py's
    safety net calls this on a first run).
    """
    trainer = Trainer(episodes=episodes, alpha=alpha, gamma=gamma,
                      epsilon_start=epsilon_start, quiet=quiet, seed=seed)
    trainer.run_all()
    if not quiet:
        print(trainer.summary())
    return trainer.agent, trainer.history


def evaluate(agent, episodes=100):
    """Pure performance check: explore=False, no learning, just play.

    Training mixes in random moves on purpose (exploration);
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

    print(f"saved {SAVE_MODEL} and {SAVE_LOG}")
    print(trainer.summary())

    # How does the SAVED brain perform — the diploma play.py will
    # actually load, read back off the disk, training wheels off?
    graduate = DQNAgent.load(SAVE_MODEL)
    avg, win_pct = evaluate(graduate, episodes=100)
    print("\ngreedy evaluation of the saved brain (no exploration):")
    print(f"  avg steps {avg:.1f}   hit the {MAX_STEPS}-step "
          f"limit {win_pct:.0f}%")
