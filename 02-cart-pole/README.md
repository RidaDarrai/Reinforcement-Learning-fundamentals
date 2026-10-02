# Project 2 — Cart-Pole Inverted Pendulum

Status: just started.

## The goal

Balance a pole on top of a cart by pushing the cart left or right.
Compared to project 1's maze, everything gets one level harder — and
that's the point:

- **The state is continuous numbers**, not a grid cell: cart position
  `x`, cart speed `ẋ`, pole angle `θ`, pole angular speed `θ̇`. A
  Q-table can't index "position 0.31728…" — I'll have to learn how to
  turn continuous numbers into something a table can store.
- **The physics happen whether I like it or not**: every step the pole
  falls a little; the agent's only choice is which way to shove the
  cart. The environment comes from Newton's equations, written by me.
- **The reward is almost too simple**: +1 for every step the pole
  stays up. Winning = surviving 500 steps. No pits, no goals — just
  staying alive.

## The plan (same recipe as project 1)

1. **The world from scratch** — `environment.py`: the pendulum
   equations, `reset() / step(action)`, done when the pole tilts too
   far or the cart hits the edge. Gym-style API, zero dependencies.
2. **The brain** — `agent.py`: start with the Q-table I already know,
   but first *bin* the four continuous numbers into buckets. State
   `(x, ẋ, θ, θ̇)` → a bucket index the table can hold.
3. **The school** — `train.py`: the same ε-greedy loop, now scoring
   "how long did it stay up".
4. **The window** — `play.py`: watch the pole wobble, balance, and
   fall — with the same modes trick where they make sense.
5. **Tests + charts** — physics checked by hand, and a learning curve
   that must climb from ~10 steps toward 500.

Notes will fill in here as I go, project-1 style.
