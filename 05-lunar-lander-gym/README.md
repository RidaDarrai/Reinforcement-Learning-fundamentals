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
                   gym's own rendering, HUD + six toggleable
                   SLIDING telemetry graphs on the side
visualize.py       THE REPORT CARD - learning curves + policy
                   slices from the diary
test_*.py          the contracts, pinned by tests (43 checks)
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
python visualize.py    # the report card (or press V in play.py)
```

```bash
python play.py         # the cockpit (screenshot: screenshots/cockpit.png)
```

The cockpit manual: `R` restart, `S` greedy <-> sampled policy,
`+`/`-` playback speed (0.5x..4x, 1.0x is real time), `1`..`6` or a
mouse-click on a chip toggles each of the six telemetry graphs
(x, y, vx, vy, tilt, leg contact), `V` opens the charts, `ESC` quits.
Every graph is SLIDING ONLY — a rolling window of the last ~6 s
(300 ticks) that travels with the replay; whole-run curves belong
to `visualize.py`. The cockpit's colors are gym's own — sampled
from a rendered `LunarLander-v3` frame and pinned by tests: black
sky, white terrain, the lander's purple `(128, 102, 230)` (+ its
darker outline `(77, 77, 128)`), the flags' yellow `(204, 204, 0)`,
and the smoke plume's red `(255, 127, 127)` -> grey `(129, 129, 129)`
fade. Note that this cockpit is a REPLAY player (04's real-time
HUMAN/TRAINED modes stay 04's, by design: the world is gym's and
the brain is the show).

## Status

> **Run 1 (official): SOLVED at episode 6,999** (4570 s wall clock,
> seed 543, the reference's own dials). The reference's solve
> arithmetic crossed: window sum **4156.74 > 4000** with the closing
> 20-episode average at ~+208 — the classic gym bar "avg > 200".
>
> | verdict | number |
> |---|---|
> | reference README claims | converges within **~1,500 episodes** |
> | this port's solve-bar crossing | **6,999 episodes** (sampled policy) |
> | greedy-policy capability | exam avg **+157.9** (70% land) by **ep 900**; best exam **+194.8** (55% land) at **ep 2450** |
> | outcome mix over the run | 705 landed / 2,579 crashed / 195 out_of_bounds / 3,520 timeout |
>
> The honest read: the graduate could land long before the bar
> counted it (70% landing by ep 900, best exam +194.8, vs the
> sampled last-100 average of +168.9). The reference's solve check
> tests the SAMPLED training policy every episode, and this run's
> sampled policy spent its middle ages hovering into timeouts
> (3,520 of them) — the bar had to wait for the sampler to stop
> stalling. Between gymnasium-v3's retuned physics (vs the v2 the
> reference flew) and torch-version luck on seed 543, "4x late on
> the sum bar while the greedy brain is on schedule" is a good
> port, honestly audited. Charts in `charts/` (`python visualize.py`).
