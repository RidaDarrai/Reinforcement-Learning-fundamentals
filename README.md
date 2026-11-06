# Reinforcement Learning fundamental projects

As the title suggests this project's purpose is to learn RL, they are a bunch of beginner-friendly projects, starting with the custom grid world maze navigation, then the cart pole inverted pendulum, the lunar lander, classic arcade, Atari Breakout or Pong, and finaly an algorithmic trading agent.


## The projects

Every project lives in its own folder with its own README, tests, and
screenshots — this file is just the map.

| # | Project |
|---|---------|
| 01 | [Grid World Maze Navigation](01-gridworld-maze/README.md) |
| 02 | [Cart-Pole Inverted Pendulum](02-cart-pole/README.md) |
| 03 | [Cart-Pole MLP (DQN)](03-cart-pole-MLP/README.md) |
| 04 | [Lunar Lander (DQN)](04-lunar-lander/README.md) |
| 05 | Classic arcade game |
| 06 | Atari (Breakout or Pong) |
| 07 | Algorithmic trading agent |

## How this repo is organized

```
RL fundimentals/
├── README.md             # this map
├── requirements.txt      # shared dependencies (pygame, numpy, matplotlib, pillow, torch)
├── 01-gridworld-maze/    # project 1: environment, agent, training, window, tests
├── 02-cart-pole/         # project 2: the inverted pendulum, built the same way
├── 03-cart-pole-MLP/     # project 3: same concept as 02, but the brain is an MLP (DQN)
└── 04-lunar-lander/      # project 4: a 3-DOF lander on the Moon, plus Double DQN
```

All these projects follow the same step-by-step guide. no Gym, no
Stable-Baselines : 

Step 1 : the 'envirenment.py' file, defines the world physics, contact rules, reward components.  <br>
Step 2 : the 'agent.py' file, defines the brain. <br>
step 3 : the 'train.py' file, takes the world (environment.py) and the empty brain (agent.py) and puts them together. <br>
step 4 : the 'play.py' file, displays the results of the training and the environment in a PyGame window. These results are often saved in JSON files. <br>


```bash
pip install -r requirements.txt   # once, at the top level
```

---

