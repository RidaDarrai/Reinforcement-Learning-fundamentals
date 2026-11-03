# Project 4 — Lunar Lander with a Double DQN

Status: done — environment → agent → train → play → charts → tests,
72 checks green (26 world + 16 brain + 9 school + 12 window +
9 report card). Four official runs landed ZERO times; the run-4
retune (below) rebuilt the reward anatomy and the attitude
authority, Gym-style, with the brain FROZEN.

## The goal

The first THREE-DEGREE-OF-FREEDOM world in the repo: a lander
falls 70 m through the Moon's sky and must arrive at a 30 m pad
**slow, level, and legs-first — all at once**. Cart-pole had ONE
control axis (push the cart); this has THREE independent ones
(the lander slides x/y AND rotates):

- **The state is eight honest floats**: `x, y, vx, vy, θ, ω, legL,
  legR` — position, speeds, tilt, spin, and the two leg-contact
  flags. No binner anywhere (03's lesson kept): the floats go
  straight into `MLP(8 → 64 → 64 → 4)`, one Q-value per button.
- **The physics are written from scratch, no Box2D** — contacts are
  DETECTED, not solved: every tick we ask "did anything poke below
  the ground line?" and the touchdown rules decide. Gravity is the
  real Moon's 1.62 m/s² and the main engine is 4.9 m/s² (T/W ≈ 3).
- **Four discrete buttons, forever**: `IDLE / MAIN / ROT_LEFT /
  ROT_RIGHT`. Like every project here, the agent picks a BUTTON,
  never a strength.
- **The teacher is DQN — Double DQN after run 1 failed**: replay
  buffer 100k (10× cart-pole's diary, because lander episodes can
  run 1,000 ticks), frozen target net synced every 500 gradient
  steps, ε-greedy 1.0 → 0.05, Adam 1e-3, γ = 0.99, best-exam
  diploma.

## What I built (the plan, done)

1. ✅ **The world from scratch** — `environment.py`: semi-implicit
   Euler at dt = 0.02 s, gravity + engine accelerations, ground
   line with a 30 m pad between two flags, and the touchdown rule
   ladder (see below).
2. ✅ **The brain** — `agent.py`: `QNetwork` + `ReplayBuffer` +
   `DQNAgent`. The state scales live in `environment.py` and the
   brain IMPORTS them (one source of truth — 03 kept a private
   copy; this project has the stricter habit).
3. ✅ **The school** — `train.py`: ε decay 1.0 → 0.05 over 1,000
   episodes, greedy exams every 50 (25 episodes each), the
   best-exam diploma, and a diary with every reward component
   separated.
4. ✅ **The window** — `play.py`: **HUMAN / TRAINED only** (no
   random-flailing WATCH, no in-window training — a 20-second
   lander episode makes both useless). Left panel: six live
   scrolling telemetry plots (x, y, vx, vy, tilt, leg contact,
   ~10 s window). Right: the Moon at 5 px/m, drawn from the SAME
   `geometry()` the physics rules read. Top-right: the keypad, the
   last action lamp-lit. Bottom: the HUD with the death reason's
   flavor.
5. ✅ **Tests + charts** — every claim pinned: 26 world checks
   (hand-computed physics), 16 brain checks (the TD loss recomputed
   BY HAND), 9 school checks, 12 headless window checks, 9 report
   card checks. `visualize.py` writes three charts.

## The touchdown rule ladder (the rules of dying here)

| Rule | Result |
|------|--------|
| hull corner below ground | crash, instantly, however gently |
| a leg touch where \|x\| > 15 m (off the pad) | crash (`missed_pad`) |
| a leg touch faster than 2.0 m/s | crash (`impact`) |
| touch between 1.5 and 2.0 m/s | **survivable** — burn and pull out |
| both legs, \|vx\|, \|vy\| ≤ 1.5, \|θ\| ≤ 15° | **landed, +100** |

The 1.5–2 m/s "recovery band" is the neat bit: a too-hard-ish touch
leaves the lander SINKING (contacts are detected, not solved), one
meter of hull clearance in reserve — a quick main-engine burn can
still arrest it and land. The agent can DISCOVER "touch, then
settle". (Run-4 retune: the band's scoring mouth widened — V_LAND
1.0 → 1.5, landing angle 10° → 15° — and a 1.53 m/s touch is now
ONE main burn from a scored landing, verified by hand in
test_recovery_band_save. V_IMPACT stayed 2.0: the DEATH band never
widens.)

## The reward anatomy (three currencies)

`reward = terminal + shaping + fuel`, reported separately every
tick so the charts can never lie about money.

- **TERMINAL**: landed **+100**; every death pays a **graded
  bill** (run 2's postmortem below): `−(5 + 2.5·speed + 4·tilt)`
  — a gentle 1 m/s graze ≈ −8, a 15 m/s meteor ≈ −50, spinning
  extra on top; hovering out the 1,000-tick clock now pays the
  **TIMEOUT_BILL = −20** (run-4 retune — it used to be free, and
  hover-the-clock was the best mediocre life).
- **SHAPING**: potential-based on the FULL state (run-4 retune —
  formerly distance-only):
  `Φ(s) = −1.5·dist/100 − 15·speed − 10·|wrapped tilt|`, paid as
  DIFFERENCES that **telescope**: the whole episode's shaping is
  exactly `Φ(end) − Φ(start)` — one hand-checkable number. Spawn
  at rest holds the old pin exactly: Φ = −1.5·70 = **−105.0**.
  Why speed/tilt joined: four runs' probes showed the marginal
  value of ONE braking tick (~0.1–0.3 Q) sitting inside the MLP's
  fitting noise — so braking was never marked for itself. Now a
  brake tick's Δv of 0.0656 m/s pays ~1.0 THAT tick, and hovering
  at constant speed pays 0: only CHANGES of speed/tilt cost or
  pay.
- **FUEL**: −0.30 per main-burn tick, −0.03 per side tick — Gym's
  real prices (run-4 retune; was 0.02/0.005 decoration). Money,
  not oxygen — no tank to run dry.

With the retuned anatomy the ladder finally runs ALL THE WAY DOWN:
**land ≈ +215** (terminal +100 + ~105 banked shaping, fuel
nibbling) **> meteor ≈ −165** (bill ~−45 + speed blood:
Φ builds −15 per m/s during the fall) **≈ hover-the-clock ≈
−150..−170** (holding 70 m costs ~⅓-duty mains × 0.30/tick +
the −20 timeout bill). The escape hatch is priced out; two club
members were expelled. See the run-4 verdict below.

## Important concepts (the new ones this project taught me)

### 1. 3 DOF is a different KIND of hard, not just harder

Balancing (02/03) asked "which push, now?". Landing asks
"descend this fast fall, level out, drift to the pad, and touch
down inside three windows at once" — speed ≤ 1.5 m/s on BOTH axes
AND level (≤ 15°) AND between the flags. Almost-correct inputs
produce almost-correct trajectories that still end either ≈ +215
or a death bill: the outcome boundary is a cliff, which shapes
what learning must do (see the DQN notes).

### 2. Potential shaping that telescopes

Φ-differences (not flat fees) mean mid-episode fiddling cancels
out; only the endpoints matter. The funnel converts "distance"
into money every tick, so APPROACHING the pad is profitable even
in a doomed episode — the gradient exists at every tick, not just
at the terminal one.

### 3. DOUBLE DQN — added because vanilla failed on this world

`.max()` over four Q-values *both picks and values* the next
action with the same frozen net. With four near-tied action
outputs the max operator systematically overestimates — and run 1
showed the extreme version (below). The fix is one split:
**the online net CHOOSES the next button; the frozen net VALUES
it.** Same bootstrap math, half the target bias. The hand-recompute
test was rewritten for the new rule (which net picks vs which net
values).

### 4. Terminal bills can drown a reward landscape — and a FLAT
###    bill can starve it (runs 1 AND 2's lesson)

Run 1: `CRASH_PENALTY = −100`, every training episode ended in a
bill (3,000 episodes, ZERO landings). The replay corpus never
contained one positive terminal event, so the Q surface it fitted
was a smooth "danger map" with a nearly FLAT action dimension.
Run 2: shrunk to a **flat −30**, land rate STILL 0% — this time
improved but not fixed: all-positive exams (+40..+58), yet the
probe showed the graduate hitting the ground at a median **14.7
m/s** (free-fall from 70 m ≈ 15.1: it stopped braking) with a
median wrapped tilt of **869°**. Why: a flat bill prices a 15 m/s
meteor EXACTLY like a 1 m/s graze — nothing pays for being gentle
— and rotation (0.005/tick vs IDLE's free) drifted into 59% of
actions. Run 3's fix: the death bill is now **graded**:
`−(5 + 2.5·speed + 4·tilt_folded_to_[0,π])`. A graze −8, a meteor
≈ −50, all paid ONCE at the fatal tick — the gentleness gradient
retrofitted exactly where the game decides life or death.

### 5. The hover trap, documented before... and priced out in run 4

Old anatomy: hovering out the clock banked the full potential and
cost ~10 fuel: ≈ +90, level with a gentle crash and far above a
meteor — a scaffold to climb through, in theory. In PRACTICE four
runs climbed the ladder sideways and settled there: run 4's exams
(+49 best) were braking snapshots with ZERO touchdowns, and just
1 timeout in ~3,000 episodes meant the policy never even needed
the hatch — mediocre-but-alive was its equilibrium. The run-4
retune closed it with data instead of hope: fuel at Gym's real
prices (hover-the-clock THERE costs ≈ −170) plus a
TIMEOUT_BILL −20, and the environment docstring's promise —
"tuned with data, not guessed" — is finally kept.

### 6. Greedy exams + best-exam diploma, one more time

The best exam's weights are deep-copied the instant they happen
and saved as `dqn_model.pt`; training never stands still and the
diploma must be the brain at its best moment. Four runs' worth of
diplomas (run 4's best: +49.05 at ep 1450, land 0%) live in the
Temp backups — none of them can land on the old world, which is
exactly why the world changed.

## What's in the project

```
04-lunar-lander/
├── environment.py       # THE WORLD: 3-DOF physics, contact rules, reward
├── agent.py             # THE BRAIN: MLP + Double DQN
├── train.py             # THE SCHOOL: exams, diploma, the diary
├── play.py              # THE COCKPIT: telemetry plots, keypad, HUD
├── visualize.py         # THE REPORT CARD: curves, anatomy, slices
├── test_*.py            # 72 checks — every claim above is tested
├── charts/              # generated report cards (python visualize.py)
├── dqn_model.pt         # the trained brain   (after python train.py)
└── training_log.json    # the training diary  (after python train.py)
```

(`dqn_model.pt` / `training_log.json` / `training_console.txt` are
git-ignored — recreate with `python train.py`.)

## How I run it

```bash
pip install -r ../requirements.txt   # torch already there from 03

python train.py          # 3,000 episodes, ~1 h CPU, saves the brain
python play.py           # the cockpit (see below)
python visualize.py      # writes charts/... (or press V inside play.py)

python test_environment.py   # I run these whenever I change something
python test_agent.py
python test_train.py
python test_play.py
python test_visualize.py
```

## How to play the game

```bash
python play.py
```

| Key          | What it does                                       |
|--------------|----------------------------------------------------|
| **↑**        | main engine burn (HUMAN mode — hold it!)           |
| **← / →**    | rotate the lander (HUMAN mode)                     |
| **SPACE**    | switch HUMAN ↔ TRAINED                             |
| **+ / −**    | slow down / speed up the physics (TRAINED)         |
| **R**        | retry this attempt                                 |
| **V**        | open the charts (launches visualize.py)            |
| **ESC / Q**  | quit                                               |

Two modes, on purpose: **HUMAN** (you, in real time — doing
nothing is a choice, IDLE free-falls) and **TRAINED** (the
graduate, greedy, auto-reloading `dqn_model.pt`; a 300-episode
quick course trains behind a warning banner if no diploma
exists). No random-flailing WATCH, no in-window LIVE school — a
lander episode is 20 s long; both of those modes were built for
episodes that last seconds.

Missing `dqn_model.pt`? TRAINED trains a quick course right
there (~1 min, writes `quick_course.pt`, never the official
run's artifacts).

> **Run status**: the official-`dqn_model.pt` slot is currently
> EMPTY of a working graduate — four 3,000-episode runs (seed 42)
> all landed 0%. Their logs and brains are backed up
> (`training_log_run1..4.json` / `dqn_model_run1..4.pt`); run 5
> will be the first to train on the retuned world.

## How I check that it actually learned

`python visualize.py` — or **V** inside the window — makes three
charts.

**The learning curves** — return per episode (raw + rolling mean +
greedy exam dots), landing rate, ε decaying, and how episodes
end. Trust the bold line and the exam dots.

![Learning curves](charts/learning_curves.png)

**The reward anatomy** — the three currencies as moving averages
(their stack should equal the raw return), and the crash-flavor
split among crashes (hull / impact / missed-pad — different
diseases, different cures).

![Reward anatomy](charts/reward_anatomy.png)

**The policy slices** — four flat cuts of the 8-D policy:
APPROACH (x, vx), DESCENT (y, vy), ATTITUDE (θ, ω), FLIGHT PLAN
(x, y). One color per button, no gray anywhere (an MLP answers
everywhere — 02's blind-spot disease is impossible here).

![Policy slices](charts/policy_slices.png)

## Run 1 postmortem (the honest part)

The first official run (vanilla DQN, crash bill −100, seed 42,
3,000 episodes, ~62 min) **landed zero times and timed out
almost zero**; rolls of greedy exams swung −15…−92 on one
policy. What the diary showed:

- greedy exam episodes lasted ≈ 460 ticks = the free-fall time
  from 70 m — the settled policy just *fell*, then died on
  landing;
- crashes piled up **~13–28 m from the pad center**, with ~40%
  hull / ~27% impact / ~27% missed-pad;
- Q(s) read like a danger map (≈ +29 at spawn, ≈ −53 near the
  ground) whose four action outputs differed by ≈ 0.3 points
  — everywhere, at every altitude.

The fix was two knobs, each motivated by that data: the crash
bill −100 → −30 (so corpus episodes stop being one-sided −100
bills and hover becomes the Tier-2 skill), and vanilla →
Double DQN (the max-over-tied-actions bias). Both are run-1
"documented in advance" knobs: `environment.py`'s trap note and
`agent.py`'s trick list said what to do if the charts showed
exactly this — and they did.

### Run 2 (Double DQN, flat −30, seed 42, 3,000 episodes): better,
### not good

Every exam from episode 50 to 3,000: land 0%, returns all
positive (+40..+58), zero wild swings — Double DQN had tamed the
overestimation, and the smaller bill let episodes bank money. But
the graduate was still a meteor. A 400-episode greedy probe of
the diploma's brain, before the next fix:

- **median |v| at the fatal tick: 14.7 m/s** — free-fall from 70
  m is ≈ 15.1. It essentially stopped braking, because a flat −30
  prices a 15 m/s crash exactly like a 1 m/s graze;
- **median wrapped tilt: 869°** (two full spins) and 59% of the
  last-60-ticks actions were ROT — rotation costs 0.005/tick vs
  IDLE's free, and with a flat action dimension the argmax drifts;
- 46% hull crashes (spinning = belly first), 36% impact, 18%
  missed pad;
- the action dimension of Q near the ground: still ≈ 0.3 spread —
  the gentleness gradient simply did not exist in the corpus.

The next fix, decided from that data: the death bill is now
**graded** (`-(5 + 2.5*speed + 4*tilt)` — see concept 4), turning
"arrive gently, arrive level" into per-tick money instead of a
hope.

### Run 3 (Double DQN, graded bill, decay window 1,000): the brain
### stopped acting

The graded bill created the gentleness gradient — and the slow ε
decay suffocated it: 3,000 episodes still never saw ε below ≈ 0.68
of greedy mass, so the corpus stayed mostly noise and the archived
brain collapsed into "do nothing": 96% IDLE, vy pinned at the
free-fall buildup, tilt 0.0, landings 0%. Medicine for run 4
(agent.py FROZEN otherwise): compress the exploration schedule,
decay window 1,000 → 300, probe-validated first — 300 episodes of
fast decay produced measurable braking attempts.

### Run 4 (decay 300, all else frozen): it fights now — and still
### can't land. Verdict: the WORLD is the problem.

The best run yet: best exam +49.05 (ep 1450), exams positive and
stable, every one of ~3,000 episodes ended in a fight (bills
graded, some near-misses). But: **0 landings, again** — crash
types 1,223 impact / 1,131 hull / 610 missed-pad, 35 OOB,
1 timeout. The mechanism is now clear. Under distance-only
shaping, the marginal value of ONE braking tick (~0.1–0.3 Q) is
inside the MLP's fitting noise (~0.2–0.3) — argmax can't see it —
and gentleness is priced only POSTHUMOUSLY, on the fatal tick.
Meanwhile the side thrusters' authority (1.5 rad/s², 0.03 rad/s
per tap) is ~10× weaker than the game demands, and the free
timeout made hover-the-clock the least-bad life. Four runs on the
same defect; a fifth would repeat it.

## The run-4 retune (Gym's recipe, brain untouched)

Run 5's training dial set is IDENTICAL to run 4's (train.py and
agent.py frozen) — the reward change is the single variable:

| Knob | Was | Now | Why |
|------|-----|-----|-----|
| shaping potential Φ | distance-only | full-state: distance − 15·speed − 10·wrapped tilt | pay gentle arrivals EVERY tick, not posthumously |
| V_LAND / landing angle | 1.0 m/s / 10° | 1.5 m/s / 15° | widen the SCORING band; the death band (V_IMPACT 2.0) stays |
| side thrusters | 1.5 rad/s² | 6.0 rad/s² | ~10× faster attitude response (ω state scale 4 → 8 to match) |
| fuel prices | 0.02 / 0.005 | 0.30 / 0.03 | Gym's real prices; hover-the-clock THERE costs ≈ −170 |
| timeout bill | 0 | −20 | the escape hatch, priced out |

Spawn scatter deliberately UNCHANGED (70 m up, x in ±30):
navigation keeps its difficulty, and those are the documented
next knobs if run 5 still stalls. Every constant carries its WHY
comment and a hand-recomputed pin in `test_environment.py`
(26 world checks; the spawn pin Φ(start, at rest) = −105.0
survives unchanged). The new outcome ladder, hand-checked:
**land ≈ +215** > deaths (meteor ≈ −165, gentle graze ≈ −15) >
**hover-the-clock ≈ −150..−170** — for the first time the ladder
runs all the way down, so "live poorly forever" pays worst.

## Things that surprised me

- **The exam-steps fingerprint.** 460-tick exams were readable
  BEFORE opening the diary: 465 ticks is free-fall from 70 m.
  A number that matches a physical constant is a diagnosis.
- **A retry with a shaping LADDER works better than a bigger
  hammer.** Every ordering land > hover > crash makes a subset
  of the problem learnable; one giant cliff doesn't.
- **Paper is 10× faster than a replay buffer.** The flat-action
  Q spread (0.3 points) explained the whole failure in one probe
  episode; no amount of "wait and see" would have.

## What I could do next with this project

- **Dueling heads** across the same MLP: state value + advantage
  split is the natural next retune.
- **Prioritized replay**: importance-sampling the rare
  landing-adjacent transitions rather than hoping the uniform
  sampler meets them.
- **Curriculum on gravity**: 1.62 → 2.5 → 3.2 m/s², then back to
  the real Moon with the agility learned there.
- **The recovery band as a named skill**: measure how often the
  agent uses the 1.5–2 m/s band ("touch, then settle") vs treating
  it as a death sentence.

---

*This is a learning project. If you're also starting out: build
the world first, break it deliberately, test everything, and let
an honest postmortem rewrite the knobs — that's where RL theory
stops being scary and starts being data.*
