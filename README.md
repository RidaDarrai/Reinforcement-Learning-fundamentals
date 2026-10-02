# Reinforcement Learning fundamental projects

As the title suggests this project's purpose is to learn RL, they are a bunch of beginner-friendly projects, starting with the custom grid world maze navigation, then the cart pole inverted pendulum, the lunar lander, classic arcade, Atari Breakout or Pong, and the lastly an algorithmic trading agent. We will start with the first one, custom grid world maze navigation. I have my environment set up as you can see.

![HUMAN mode](01-gridworld-maze/screenshots/human.gif)

## Why I built this

I kept reading about reinforcement learning and getting lost in the
vocabulary: *policy, value function, Bellman equation, exploration…*
Every tutorial showed me a wall of math, but I learn best by **building
the smallest possible thing myself**.

So my goal for this project was:

1. **Write my own environment** — a tiny 7×7 maze, so the "world" is
   something I can see and reason about.
2. **Write my own agent** — a Q-learning table, ~100 lines, no library
   hides the update rule from me.
3. **Train it and WATCH it learn** — a game window where I can play the
   maze myself, watch a random agent fail, and then watch my trained
   agent win.
4. **Prove it learned** — charts of the training run and a map of what
   the agent believes about every cell.


## Important concepts

### 1. The agent–environment loop

Everything in RL is one loop:

```
    AGENT  --acts-->  ENVIRONMENT
      ^                    |
      |--- (next state, reward, done) ---|
```

In my project the **environment** is `environment.py` (the maze, the
walls, the scoring rules) and the **agent** is `agent.py` (the brain).
They only talk through two functions — this is the "Gym API" that every
RL library in the world uses:

```python
state = env.reset()                          # start a new episode
state, reward, done, info = env.step(action)  # take one action
```

The thing that finally clicked for me: **the agent never sees the maze
internals** — it only sees numbers (its position, rewards). The maze
could be anything.

### 2. State, action, reward

- **State** = where I am: just `(row, col)`. My maze is *fully
  observable* — the agent knows everything about its situation.
- **Action** = one of 4 integers: `UP=0, DOWN=1, LEFT=2, RIGHT=3`.
  RL algorithms need numbers, not words like "north".
- **Reward** = the only teaching signal. I chose mine carefully:

| Event      | Reward | Why I picked it            |
|------------|--------|----------------------------|
| Reach goal | **+10** | the big win to chase       |
| Fall in pit| **−10** | the big mistake to avoid   |
| Every step | **−0.1** | a "hurry up" tax, so the agent learns *short* paths, not just any path |
| 100 steps  | timeout | ends hopeless episodes     |

The step penalty was my first real lesson in **reward design**: without
it, the agent happily wanders in circles forever — any path to the goal
is "equally good". With it, the shortest path wins. One number changed
the whole behavior.

### 3. Episodes

One full attempt (start → win/lose/timeout) is an **episode**. Training
= running thousands of episodes. My timeout is 100 steps so that a
hopeless episode can't run forever.

### 4. Q-learning: teaching with a scorecard

This is the core idea I came to build. The agent keeps a **Q-table** —
a scorecard for every (position, action) pair:

```
Q[(2, 3)][DOWN] = 8.2     "moving down from row 2, col 3 is worth ~8.2"
```

Q means *"how much total reward do I expect from here if I take this
move, then play reasonably after?"* After every step, I nudge the score
I just used toward a better estimate — the **temporal-difference
update**:

```
target = r                         if the episode ended here
       = r + gamma * max Q(s', a') if it continues

Q(s, a) += alpha * (target - Q(s, a))
           |_____|    |___________|
           learning    how wrong we were
             rate       (the error we fix)
```

The three knobs I now keep forever:

- **alpha (learning rate, 0.1)** — rewrite 10% toward each new truth.
  Too high = my table flip-flops wildly; too low = I learn in slow motion.
- **gamma (discount, 0.99)** — rewards 12 steps away still count ~89%.
  This is what lets a +10 at the goal teach the move from *ten steps
  back*. Without gamma, learning could never propagate upstream.
- **epsilon (exploration, 1.0 → 0.05)** — see next point.

### 5. Explore vs exploit — my favorite RL concept

Sometimes pick the **current best guess** (exploit), sometimes pick a
**random move** (explore). That's ε-greedy:

- ε = 1.0 at the start: pure wandering — you can't find treasure you
  never stumbled into.
- ε shrinks every episode toward 0.05: mostly exploiting now, but 5%
  silliness forever so the agent never goes fully blind.

**The trap I worried about**: if the agent is never allowed to be dumb,
it explores its first lucky habit and never discovers the better path.

### 6. Training vs evaluating

During training, randomness is *on purpose* (that's exploration) — so
training scores look messy. To see real skill I switch it off
(`explore=False`) and run pure greedy evaluation. That's exactly what
the **TRAINED** mode in the game window does.

### 7. Why I'll need deep RL later (teaser)

My Q-table works because there are only 49 cells. Give this maze 10
million states and the table won't fit in memory — that's when a neural
network replaces the table (Deep Q-Learning). I haven't gone there yet;
this project is the tabular foundation first.

## What's in the project

```
01-gridworld-maze/
├── environment.py      # THE WORLD: maze, moves, rewards (Gym-style API)
├── agent.py            # THE BRAIN: Q-table + ε-greedy + TD update
├── train.py            # THE SCHOOL: fills the Q-table, saves results
├── play.py             # THE WINDOW: HUMAN / WATCH / LIVE / TRAINED
├── visualize.py        # THE REPORT CARD: learning curves + policy map
├── capture_media.py    # the camera that recorded the GIFs below
├── test_*.py           # 45 checks: every claim above is tested
├── q_table.json        # the trained brain   (after python train.py)
├── training_log.json   # the training diary  (after python train.py)
└── maze_random.json    # my random maze     (after pressing N, once)
```

## How I run it

```bash
pip install -r requirements.txt

python train.py          # trains 2000 episodes in ~0.1 s, saves the brain
python play.py           # the game window (see below)
python visualize.py      # writes charts/... (or press V inside play.py)

python test_environment.py   # I run these whenever I change something
python test_agent.py
python test_train.py
python test_play.py
python test_visualize.py
```

Training results **persist**: `train.py` writes `q_table.json` to disk,
so I can close everything and `play.py` still loads the same trained
agent tomorrow. The maze persists the same way — `maze_random.json`
remembers the random maze I generated until I generate another one.

## How to play the game

```bash
python play.py
```

| Key            | What it does                                  |
|----------------|-----------------------------------------------|
| **Arrow keys** | move (only in HUMAN mode)                     |
| **SPACE**      | cycle the mode: HUMAN → WATCH → LIVE → TRAINED |
| **+ / −**      | speed up / slow down (WATCH, LIVE & TRAINED)  |
| **N**          | generate a NEW random maze (it retrains)      |
| **TAB**        | skip to the next training stage (LIVE)        |
| **V**          | open the charts of the trained agent          |
| **R**          | restart the episode                           |
| **ESC / Q**    | quit                                          |

### Which maze am I on?

The **classic maze** on my first ever run. Press **N** and a freshly
generated random maze becomes *the* maze: it's saved to
`maze_random.json`, loaded again on every future launch, and replaced
only when I press N again — I never get classic back (by design: one
active maze, always the newest). The generator guarantees every maze
is fair: solvable from S to G with a way around the pit, and still
possible to blunder *into* the pit.

Because the one agent was trained for the old maze, N shows a banner
and **retrains it automatically** (~0.1 s) — the brain always matches
the maze I'm playing.

### Mode 1 — HUMAN: me against the maze

The badge says **HUMAN**, and I drive with the arrow keys. I play this
first on purpose: feeling how easy it is to bump walls and walk into
the pit makes me appreciate what the agent has to figure out. (Bumping
still costs a step — watch the reward tick down.)

![I play the maze myself — note the honest wall bump](01-gridworld-maze/screenshots/human.gif)

### Mode 2 — WATCH: the random agent (the baseline)

Press SPACE. Now a mindless agent picks a random move every tick — no
thinking, just luck. This is the **baseline to beat**, and the GIF
below is why: after 69 hopeless wanders it falls into the pit (reward
−16.8). Randomness never learns; my rewards show every mistake.

![Random agent wanders 69 steps and falls into the pit](01-gridworld-maze/screenshots/watch.gif)

### Mode 3 — LIVE: watching the brain fill itself

SPACE again. This mode doesn't play the finished agent — it **runs a
fresh training session in front of me**. The HUD counts episodes
(EP 437/2000), shows which of the three stages I'm in, the live
exploration rate ε, and the rolling win rate:

| Stage        | ε range        | What the agent is doing          |
|--------------|----------------|----------------------------------|
| EXPLORING    | ε > 0.5        | mostly random moves — wandering  |
| LEARNING     | 0.05 < ε ≤ 0.5 | finds getting cashed in          |
| POLISHING    | ε = 0.05       | near-greedy, tuning the paths    |

The board meanwhile animates the agent's **current best play** —
rebuilt every few seconds, so it visibly gets smarter as the numbers
climb. `+ / −` sets how fast episodes fly by (4 to 500 per second),
**TAB** fast-forwards to the next stage, **R** replays the demo.

And the safety net: if I enter LIVE without ever training, a warning
banner is dumped on screen and training starts **automatically**
behind it — I can't forget to launch it.

### Mode 4 — TRAINED: my graduate

SPACE again. The trained agent plays **greedily** — no random moves,
no learning, pure performance. It takes the optimal 12-step route past
the pit and wins with +8.9 every single time. The `+ / −` keys change
how fast you watch it. If `q_table.json` is missing, the same safety
net kicks in: warning banner first, then training runs by itself
(~0.1 s) before the agent starts playing.

![Trained agent takes the optimal 12-step path](01-gridworld-maze/screenshots/trained.gif)

**V** (in any mode) opens the report card — learning curves and policy
map of the *current* agent — in its own window, training first if it
was missing.

Side by side, that's the whole story of this project:

| Mode     | Who decides      | Result                |
|----------|------------------|-----------------------|
| HUMAN    | me (arrow keys)  | depends on my day :)  |
| WATCH    | `random`         | −16.8, fell in pit    |
| LIVE     | the learner      | watch ε fall, wins climb |
| TRAINED  | the Q-table      | **+8.9, every time**  |

## How I check that it actually learned

`python visualize.py` — or **V** inside the game window — turns the
saved files into two charts.

**The learning curves** — raw episode scores are noisy, so I trust the
bold moving-average line. I look for three things moving together:
reward climbing, win rate reaching 100%, and ε decaying (the handover
from exploring to exploiting). Steps-per-episode dropping shows it
found *shorter* paths, thanks to the −0.1/step tax.

![My training run: reward, win rate, epsilon, steps](01-gridworld-maze/charts/learning_curves.png)

**The policy map** — my favorite picture. Each cell shows the best
arrow Q-learning chose there, colored by how good that cell is (red →
green). The green corridor from the start to the goal *is* the learned
policy; the dark cells are walls, and the red zone around the pit is
the agent's learned fear. (The maze drawn comes from the training log,
so charting a random maze draws *that* maze, not the classic one.)

![The Q-table drawn on the maze: arrows = best moves, color = value](01-gridworld-maze/charts/policy_map.png)

## Things that surprised me (my honest notes)

- **Learning is fast.** ~200 episodes is when it visibly "gets it".
  I expected hours; tabular Q-learning on 49 cells takes 0.1 seconds.
- **The first 100 episodes look hopeless.** Rewards stuck at −10 while
  ε is still high — I almost thought it was broken. The curve said
  otherwise later.
- **Reward design IS the job.** Half my "bugs" were really the
  environment rewarding the wrong thing.
- **The pit is surrounded by red even on cells it never fell from** —
  the fear propagated backwards through γ, exactly like the theory
  says. Seeing the math show up in a picture was the best moment of
  the project.
- **Everything is testable.** 45 checks cover walls, rewards, the TD
  update by hand, ε-greedy, training quality, the random-maze
  generator, LIVE-mode key handling, HUD layout, and charts — so when
  I change a constant, I find out immediately.

## What I want to do next

- **Step 7 — hyperparameter experiments**: vary alpha/gamma/epsilon and
  compare learning curves side by side.
- A **second environment** (maybe a slippery-ice maze) to prove the
  agent doesn't secretly depend on this one maze.
- Then, eventually: **Deep Q-Learning** for something too big for a table.

---

*This is a learning project. If you're also starting out: build the
world first, break it deliberately, test everything, and watch the
curves — that's where the theory stops being scary.*
