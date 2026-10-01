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

Two files come out of training:

    q_table.json        the trained brain (loaded later by play.py)
    training_log.json   every episode's score + outcome — the raw
                        material for step 5's charts

Run it:

    python train.py
"""

import json
import time

from agent import QAgent
from environment import GridWorld, MAX_STEPS

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
SAVE_Q = "q_table.json"
SAVE_LOG = "training_log.json"

# How a finished episode is labeled (from the environment's info dict).
OUTCOME_LABEL = {"goal": "GOAL", "pit": "PIT", "timeout": "TIME"}


def train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA,
          epsilon_start=EPSILON_START, quiet=False):
    """Run the training loop. Returns (agent, history).

    history is a dict of parallel lists — one entry per episode —
    ready to be turned into charts in step 5.
    """
    env = GridWorld()
    agent = QAgent(n_actions=4, alpha=alpha, gamma=gamma,
                   epsilon=epsilon_start)

    history = {
        "episode": [],     # episode number
        "reward": [],      # total reward earned that episode
        "outcome": [],     # "goal" / "pit" / "timeout"
        "epsilon": [],     # exploration chance at episode start
        "steps": [],       # how many moves it took
    }

    started = time.time()

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        # ---- THE LOOP (the actual reinforcement learning) ------
        while True:
            action = agent.choose_action(state)          # 1. decide
            next_state, reward, done, info = env.step(action)  # 2. act
            agent.update(state, action, reward,          # 3. learn
                         next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        # -------------------------------------------------------

        # Shrink exploration: early episodes are for looking around,
        # later ones for cashing in what we found. Never below MIN —
        # a 5% chance of silliness keeps the agent honest forever.
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)

        history["episode"].append(episode)
        history["reward"].append(total_reward)
        history["outcome"].append(info["reason"])
        history["epsilon"].append(agent.epsilon)
        history["steps"].append(info["steps"])

        # ---- progress report ------------------------------------
        if not quiet and episode % LOG_EVERY == 0:
            recent = history["reward"][-LOG_EVERY:]
            wins = sum(1 for o in history["outcome"][-LOG_EVERY:]
                       if o == "goal")
            avg = sum(recent) / len(recent)
            print(f"ep {episode:>5}/{episodes} | "
                  f"avg reward {avg:+6.2f} | "
                  f"wins {wins:>3}/{LOG_EVERY} | "
                  f"epsilon {agent.epsilon:.3f}")

    elapsed = time.time() - started

    # ---- final summary ----------------------------------------
    last100 = history["reward"][-100:]
    last100_wins = sum(1 for o in history["outcome"][-100:]
                       if o == "goal")
    if not quiet:
        print(f"\ntrained in {elapsed:.1f}s")
        print(f"last 100 episodes: avg reward {sum(last100)/len(last100):+.2f}, "
              f"win rate {last100_wins}%")
        print(f"notebook filled: {len(agent.q_table)} positions visited")

    return agent, history


def evaluate(agent, episodes=100):
    """Pure performance check: explore=False, no learning, just play.

    Training mixes in random moves on purpose (that's exploration);
    evaluation switches it off so the numbers show the REAL skill.
    Returns (avg_reward, win_rate).
    """
    env = GridWorld()
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
    agent, history = train()

    # Save brain + training diary.
    agent.save(SAVE_Q)
    with open(SAVE_LOG, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=1)
    print(f"saved {SAVE_Q} and {SAVE_LOG}")

    # How does the GRADUATE perform with the training wheels off?
    avg, win_pct = evaluate(agent, episodes=100)
    print(f"\ngreedy evaluation (no exploration):")
    print(f"  avg reward {avg:+.2f}   win rate {win_pct:.0f}%")
