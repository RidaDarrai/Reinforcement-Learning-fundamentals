# 05 - Lunar Lander on the real Gym (Actor-Critic)

The lunar lander AGAIN - but this time the world is a real
environment instead of ours, and the brain is the other classical
family. Project 04's DQN story (six runs, two retunes, zero
landings, a greedy policy that blasted straight up) demanded a
clean restart: same problem, real physics, different algorithm.

```
environment.py     THE WORLD - gymnasium's LunarLander-v3 behind a
                   classic reset()/step() wrapper (our 4-tuple API)
agent.py           THE BRAIN - episodic Actor-Critic: policy net +
                   value baseline, one gradient step per episode
train.py           THE SCHOOL - the reference's loop, solve bar
                   and dials, + our diary/exams/diploma conventions
play.py            THE COCKPIT - watch the TRAINED brain fly on
                   gym's own rendering, HUD on top
visualize.py       THE REPORT CARD - learning curves + policy
                   slices from the diary
test_*.py          the contracts, pinned by tests (34 checks)
```

## Why TWO families of brain

| | 04's DQN | 05's Actor-Critic |
|---|---|---|
| What the net predicts | Q(s, a), "how good is each button" | pi(a \| s), "which button to pick" + a value baseline |
| Behavior source | argmax of the Q | **samples its own exploration** (no epsilon dial) |
| Learning trigger | per-tick TD error off a replay diary | one gradient step PER EPISODE (whole-trajectory credit) |
| Known disease (observed in 04) | delusional values -> argmax ties -> streak policies | high variance -> controlled by reward normalization |

The reference implementation this port keeps its hands locked to:
[Actor-Critic-PyTorch](https://github.com/nikhilbarhate99/Actor-Critic-PyTorch)
(vanilla episodic A2C on LunarLander-v2 - same 8-float state, same
4 buttons; the port maps it onto gymnasium's maintained v3).

## The dials (all the reference's own)

gamma 0.99, lr 0.02, Adam betas (0.9, 0.999), seed 543. One policy
update per episode (no replay buffer, no target net, no epsilon).
The solve check is the reference's quirky arithmetic, kept
slavishly: an accumulator sums episode returns, every episode it
is tested `> 4000` (a 20-episode FULL window at that sum is the
classic gym bar "average return > 200"), and every 20th episode it
is divided by 20 and reset.

## Run it

```bash
python train.py        # the official school (see the bar above)
python play.py         # watch the graduate fly (R restart, S toggle)
python visualize.py    # the report card (or press V in play.py)
```

## Status

> **Run 1 (official)**: seed 543, the reference's own dials, bar
> solved at running-sum > 4000 (avg 200 over the 20-window),
> reference README claims **converges within 1,500 episodes** -
> that is the bar this run has to clear.
