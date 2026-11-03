"""
visualize.py — THE REPORT CARD (step 5 of the lunar lander project)
====================================================================

Steps 1-4 built the world, the brain, the school, and the cockpit.
Step 5 reads the receipts the school's diary wrote: WHEN it learned,
WHERE the money went, and WHAT the graduate would do mid-flight.

Three charts, saved next to this file in charts/:

    learning_curves.png   "did it learn, and WHEN?"
    reward_anatomy.png    "where did every point of return come from?"
    policy_slices.png     "what would it do from HERE?"

HOW TO READ THEM (the actual lesson):

learning_curves.png — a 2x2 grid:
  top-left     RETURN per episode. Unlike cart-pole ("reward IS
               steps"), a landing is worth ~+200 while deaths pay
               a GRADED bill (~-8 for a gentle graze, ~-50 for a
               meteor) — the shaping made distance money. Raw
               episodes are noisy; trust the
               bold average AND the exam dots (greedy policy, no
               luck allowed in).
  top-right    LANDING rate, rolling, plus the exams' land rates as
               dots. This is the single number that matters.
  bottom-left  exploration rate epsilon, RECOMPUTED from the log's
               own config (the diary does not store it per episode —
               it is derived, derived honestly, from the same knobs
               train.py used).
  bottom-right HOW episodes end: landed / crashed / out_of_bounds /
               timeout. A timeout-heavy policy means the known
               hover trap won; a crash-heavy one means "close, but
               not gentle".

reward_anatomy.png — two panels:
  left         the three currencies, per episode (moving averages):
               TERMINAL (a +100 landing, graded death bills), SHAPING (the banked
               distance potential), FUEL (the burn bill). Their sum
               IS the return — train.py asserts that identity, and
               this chart shows it: check that the blue/orange/green
               lines stack up to the faint raw return.
  right        among the CRASHES only: rolling split of hull flop /
               hard impact / missed pad. "You crash at the pad"
               (impact) and "you die far away" (out of...) are
               different diseases with different cures.

policy_slices.png — the policy has EIGHT input dimensions, so the
report card cuts four informative flat slices (like CT scans), legs
pinned at 0 (in the air) and the other floats fixed per panel:
  approach     (x, vx) at y=40: how it channels the drift home
  descent      (y, vy) at x=0: when it burns against the fall
  attitude     (theta, omega) at x=0, y=40: the balancing act
  flight plan  (x, y) at zero speeds: the map-level strategy
  ONE color per button: gray IDLE, orange MAIN, blue LEFT,
  violet RIGHT — an MLP answers EVERYWHERE, so no gray gaps
  (that was 02's table disease); read the boundaries instead.

Axis ranges come from environment.py's own scales (STATE_SCALE and
the world constants) — imported, never retyped.

Run it (after python train.py — or press V inside play.py):

    python visualize.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import json     # training_log.json: the diary step 3's Trainer wrote
import math     # radians() for the attitude axis, isfinite guards
import os       # paths anchored next to THIS file (not the cwd)
import sys      # exit code for CI (0 = charts built, 1 = train first)

import matplotlib.pyplot as plt    # Agg backend in tests — no windows
from matplotlib.patches import Patch, Rectangle  # legend swatches

# The brain (its get_q answers every slice cell) and the world's
# canonical scales (the slice axes — one source of truth).
from agent import DQNAgent
from environment import (
    STATE_SCALE, ACTION_NAMES, IDLE, MAIN, ROT_LEFT, ROT_RIGHT,
    WORLD_HALF_WIDTH, CEILING_Y, MAX_STEPS,
)
# Reuse train.py's EXACT artifact paths — one source of truth, so the
# charts can never point at files training doesn't actually write.
from train import LOG_PATH, DIPLOMA_PATH

# All output lives next to this file, whatever folder you ran from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------
# CHART COLORS — one per line / per button
# ---------------------------------------------------------------
C_RETURN  = "steelblue"     # per-episode return
C_EXAM    = "black"         # greedy-exam markers (no luck allowed)
C_LAND    = "seagreen"      # rolling landing rate
C_EPS     = "darkorange"    # exploration rate
C_OUT     = {               # outcome colors (the death/life mix)
    "landed": "seagreen",
    "crashed": "#e2574c",
    "out_of_bounds": "#8d93a1",
    "timeout": "#b8860b",
}
C_TERMINAL = "#3b82f6"      # reward anatomy: +100 land / graded bills
C_SHAPING  = "#f59e0b"      # the banked distance potential
C_FUEL     = "seagreen"     # the burn bill
C_CRASH    = {              # crash flavors among crashes
    "hull": "#e2574c",
    "impact": "#d97706",
    "missed_pad": "#8d93a1",
}

# Slice cell colors: ONE per button (an MLP answers everywhere —
# four colors per cell, no gray "never visited" holes possible).
ACTION_COLORS = {
    IDLE: "#9aa3ad",        # gray — fall / coast
    MAIN: "#f59e0b",        # orange — burn
    ROT_LEFT: "#3b82f6",    # blue — lean one way
    ROT_RIGHT: "#8b5cf6",   # violet — lean the other
}

# Slice resolution: 24 x 16 cells per panel (384 forward passes
# each — fine enough to see the boundary, cheap enough to run fast).
SLICE_GRID = (24, 16)

# Canonical slice axes, straight from the world's own scales.
SLICE_X      = (-WORLD_HALF_WIDTH, WORLD_HALF_WIDTH)     # m; +/-50
SLICE_Y      = (0.0, CEILING_Y)                          # m; 0..80
SLICE_VX     = (-STATE_SCALE[2], STATE_SCALE[2])         # m/s; +/-6
SLICE_VY     = (-STATE_SCALE[3], STATE_SCALE[3])         # m/s; +/-8
SLICE_TILT   = (-math.pi, math.pi)                       # rad; drawn as degrees
SLICE_OMEGA  = (-STATE_SCALE[5], STATE_SCALE[5])         # rad/s; +/-4


# ==============================================================
# 1. THE SMOOTHER — raw episodes are noise; trends are not
# ==============================================================
def moving_average(values, window=100):
    """Rolling mean: for each position, the average of the `window`
    values ending there (fewer at the very start — honest, not padded).

    This one line is why all four charts are readable at all: single
    lander episodes swing from -180 to +220, but the average of 100
    tells the trend.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)          # growing at first
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def load_history(path=LOG_PATH):
    """Read training_log.json (what train.py wrote)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def epsilon_schedule(n_episodes, decay_episodes, floor):
    """Recompute the school's exploration dial from the log's config.

    The diary does not store epsilon per episode; it stores the three
    knobs (decay window + floor) — the SAME formula Trainer used:
    linear 1.0 -> floor, then floor forever.
    """
    return [floor if ep >= decay_episodes
            else 1.0 - (1.0 - floor) * ep / decay_episodes
            for ep in range(n_episodes)]


def rolling_share(episodes, predicate, window=100):
    """Rolling percentage of episodes satisfying `predicate`.

    Episode records are dicts (the diary's flat rows); a predicate
    like `lambda r: r["outcome"] == "landed"` picks a slice of them.
    """
    out = []
    for i in range(len(episodes)):
        chunk = episodes[max(0, i - window + 1):i + 1]
        hits = sum(1 for r in chunk if predicate(r))
        out.append(100.0 * hits / len(chunk))
    return out


def crash_flavor_share(episodes, flavor, window=100):
    """Rolling split WITHIN crashes: `flavor` as a % of the crashes
    in the window (None where no crash happened — the line breaks
    instead of lying)."""
    out = []
    for i in range(len(episodes)):
        chunk = episodes[max(0, i - window + 1):i + 1]
        crashed = [r for r in chunk if r["outcome"] == "crashed"]
        if not crashed:
            out.append(None)
            continue
        share = 100.0 * sum(1 for r in crashed
                            if r.get("crash_type") == flavor) / len(crashed)
        out.append(share)
    return out


# ==============================================================
# 2. CHART ONE — did it learn, and when?
# ==============================================================
def plot_learning_curves(history, window=100, save_path=None):
    """Turn the training diary into a 2x2 figure. Returns the figure
    (tests grab it; the command line just shows it).
    """
    eps_rows = history["episodes"]
    required = ("return", "outcome")
    missing = [k for k in required if k not in (eps_rows[0] if eps_rows
                                                else {})]
    if missing:
        raise ValueError(f"training log is missing {missing} "
                         f"— re-run: python train.py")

    n = len(eps_rows)
    ep = list(range(1, n + 1))
    returns = [r["return"] for r in eps_rows]
    exams = history.get("exams", [])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    (ax_ret, ax_land), (ax_eps, ax_death) = axes

    # --- top-left: return, raw + average + the exams ---------------
    ax_ret.plot(ep, returns, lw=1, alpha=0.25, color=C_RETURN,
                label="each episode")
    ax_ret.plot(ep, moving_average(returns, window), lw=2.4,
                color=C_RETURN, label=f"{window}-episode average")
    if exams:
        exam_ep = [e["episode"] for e in exams]
        exam_ret = [e["avg_return"] for e in exams]
        ax_ret.plot(exam_ep, exam_ret, lw=1.6, color=C_EXAM, marker="o",
                    ms=4, label="greedy exams (no luck)")
    ax_ret.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax_ret.set_title("Return per episode (landing ~ +200; deaths graded)")
    ax_ret.set_ylabel("return")
    ax_ret.legend(loc="lower right", fontsize=8)

    # Report-card corner: the last `window` episodes, summarized.
    tail = list(zip(returns[-window:],
                    [r["outcome"] for r in eps_rows[-window:]]))
    if tail:
        tail_land = sum(1 for _, o in tail if o == "landed")
        ax_ret.text(
            0.98, 0.05,
            f"last {len(tail)} episodes:\n"
            f"avg {sum(r for r, _ in tail) / len(tail):+.1f}   "
            f"{tail_land}/{len(tail)} landed",
            transform=ax_ret.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="gray", alpha=0.9),
        )

    # --- top-right: landing rate, rolling + exam dots ---------------
    land_pct = rolling_share(eps_rows, lambda r: r["outcome"] == "landed",
                            window)
    ax_land.plot(ep, land_pct, lw=2.4, color=C_LAND,
                 label="rolling land rate")
    if exams:
        ax_land.plot([e["episode"] for e in exams],
                     [100.0 * e["land_rate"] for e in exams],
                     lw=0, marker="o", ms=4, color=C_EXAM,
                     label="greedy exams")
    ax_land.set_ylim(-3, 103)
    ax_land.set_title("Landing rate (rolling 100 episodes)")
    ax_land.set_ylabel("% landed")
    ax_land.legend(loc="upper left", fontsize=8)

    # --- bottom-left: exploration, recomputed honestly --------------
    cfg = history["config"]
    eps_line = epsilon_schedule(n, cfg["epsilon_decay_episodes"],
                                cfg["epsilon_floor"])
    ax_eps.plot(ep, eps_line, lw=2, color=C_EPS)
    ax_eps.set_ylim(0, 1.05)
    ax_eps.set_title("Exploration rate (recomputed from the log)")
    ax_eps.set_ylabel("epsilon")

    # --- bottom-right: HOW episodes end -----------------------------
    for outcome, color in C_OUT.items():
        pct = rolling_share(eps_rows, lambda r, o=outcome: r["outcome"] == o,
                            window)
        ax_death.plot(ep, pct, lw=2, color=color,
                      label={"landed": "landed",
                             "crashed": "crashed (any way)",
                             "out_of_bounds": "left the world",
                             "timeout": "timeout (hovered?)"}[outcome])
    ax_death.set_ylim(-3, 103)
    ax_death.set_title("How episodes end (rolling 100)")
    ax_death.set_ylabel("% of episodes")
    ax_death.set_xlabel("training episode")
    ax_death.legend(fontsize=8)
    ax_land.set_xlabel("training episode")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle("Lunar lander DQN — did it learn?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 3. CHART TWO — where did the money come from?
# ==============================================================
def plot_reward_anatomy(history, window=100, save_path=None):
    """The reward's three currencies + the crash-flavor split."""
    eps_rows = history["episodes"]
    if not eps_rows:
        raise ValueError("training log has no episodes to chart")
    n = len(eps_rows)
    ep = list(range(1, n + 1))

    terminal = [r["terminal"] for r in eps_rows]
    shaping = [r["shaping"] for r in eps_rows]
    fuel = [r["fuel_cost"] for r in eps_rows]

    fig, (ax_money, ax_flavor) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- left: the three currencies, moving averages ---------------
    for series, color, label in (
        (terminal, C_TERMINAL, "terminal (+100 land / graded death)"),
        (shaping, C_SHAPING, "shaping (banked distance)"),
        (fuel, C_FUEL, "fuel (the burn bill)"),
    ):
        ax_money.plot(ep, moving_average(series, window), lw=2.2,
                      color=color, label=label)
    # The raw return underneath: the three bold lines should STACK
    # UP to it (the log's money identity — train.py asserts it).
    ax_money.plot(ep, [r["return"] for r in eps_rows], lw=1, alpha=0.2,
                  color="black", label="raw return (their sum)")
    ax_money.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax_money.set_title("Where the return comes from (rolling 100)")
    ax_money.set_xlabel("training episode")
    ax_money.set_ylabel("points")
    ax_money.legend(fontsize=8)

    # --- right: among the crashes, WHICH way did we die? -----------
    for flavor, color in C_CRASH.items():
        share = crash_flavor_share(eps_rows, flavor, window)
        nan_share = [float(x) if x is not None else float("nan")
                     for x in share]
        label = {"hull": "hull flop (belly hit dirt)",
                 "impact": "hard impact (touch too fast)",
                 "missed_pad": "missed pad (a foot off)"}[flavor]
        ax_flavor.plot(ep, nan_share, lw=2, color=color, label=label)
    ax_flavor.set_ylim(-3, 103)
    ax_flavor.set_title("Crash flavors among crashed episodes\n"
                        "(blank = no crashes in the window)")
    ax_flavor.set_xlabel("training episode")
    ax_flavor.set_ylabel("% of crashes")
    ax_flavor.legend(fontsize=8)

    for ax in (ax_money, ax_flavor):
        ax.grid(alpha=0.3)

    fig.suptitle("Lunar lander DQN — reward anatomy", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 4. CHART THREE — what would it do from HERE?
# ==============================================================
def peek_action(agent, state):
    """The brain's chosen action for a raw 8-float state (read-only)."""
    q = agent.get_q(state)
    return max(range(len(q)), key=lambda a: q[a])


def plot_policy_slices(agent, save_path=None, grid=SLICE_GRID):
    """Draw four 2-D slices of the 8-dimensional DQN policy.

    Every panel pins its 6 other floats: legs at 0 (still in the
    air) and the drift speeds/attitude named in the titles. The
    attitude axis lives in DEGREES end to end (ticks included) but
    the brain reads RADIANS — converted back before probing.
    """
    nx, ny = grid
    legs_off = (0, 0)

    # (x_range, y_range, template state, x_dim, y_dim, xlabel, title,
    #  x_is_degrees)
    panels = (
        (SLICE_X, SLICE_VX,
         (0.0, 40.0, 0.0, 0.0, 0.0, 0.0) + legs_off,
         0, 2, "sideways position x (m)",
         "APPROACH: channel the drift home (y = 40)", False),
        (SLICE_Y, SLICE_VY,
         (0.0, 40.0, 0.0, 0.0, 0.0, 0.0) + legs_off,
         1, 3, "height above ground y (m)",
         "DESCENT: when to burn against the fall (x = 0)", False),
        (SLICE_TILT, SLICE_OMEGA,
         (0.0, 40.0, 0.0, 0.0, 0.0, 0.0) + legs_off,
         4, 5, "tilt theta (degrees)",
         "ATTITUDE: the balancing act (x = 0, y = 40)", True),
        (SLICE_X, SLICE_Y,
         (0.0, 40.0, 0.0, 0.0, 0.0, 0.0) + legs_off,
         0, 1, "sideways position x (m)",
         "FLIGHT PLAN: the map-level strategy (driftless)", False),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

    for ax, (xr, yr, template, x_dim, y_dim, xlabel, title,
             x_is_degrees) in zip(axes.flat, panels):
        (xl, xh), (yl, yh) = xr, yr
        dx, dy = (xh - xl) / nx, (yh - yl) / ny
        for i in range(nx):
            for j in range(ny):
                probe = list(template)
                xv = xl + (i + 0.5) * dx          # x-axis value
                probe[x_dim] = math.radians(xv) if x_is_degrees else xv
                probe[y_dim] = yl + (j + 0.5) * dy  # y-axis value
                action = peek_action(agent, probe)
                ax.add_patch(Rectangle(
                    (xl + i * dx, yl + j * dy), dx, dy,
                    facecolor=ACTION_COLORS[action],
                    edgecolor="white", lw=0.6))
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_xlabel(xlabel)
        if x_is_degrees:
            ticks = [xl + (xh - xl) * k / 4 for k in range(5)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.0f}°" for t in ticks])
            ax.set_xlim(xl, xh)   # set_xticklabels rescales: pin back
        ax.set_title(title, fontsize=10)

    legend = [Patch(facecolor=ACTION_COLORS[a], label=ACTION_NAMES[a])
              for a in (IDLE, MAIN, ROT_LEFT, ROT_RIGHT)]
    fig.legend(handles=legend, loc="lower center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Learned DQN policy in four slices", fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 5. MAIN — build all three charts, save them, open them
# ==============================================================
def main(log_path=LOG_PATH, model_path=DIPLOMA_PATH, out_dir=CHART_DIR,
         show=True):
    """Generate the charts. Returns 0 on success, 1 on missing input
    (so `echo %errorlevel%` and CI can both see failures)."""
    missing = [p for p in (log_path, model_path) if not os.path.exists(p)]
    if missing:
        print(f"missing {os.path.basename(missing[0])} "
              f"- train first:\n    python train.py")
        return 1

    history = load_history(log_path)
    agent = DQNAgent.load(model_path)

    os.makedirs(out_dir, exist_ok=True)
    curves_png = os.path.join(out_dir, "learning_curves.png")
    anatomy_png = os.path.join(out_dir, "reward_anatomy.png")
    slices_png = os.path.join(out_dir, "policy_slices.png")

    plot_learning_curves(history, save_path=curves_png)
    plot_reward_anatomy(history, save_path=anatomy_png)
    plot_policy_slices(agent, save_path=slices_png)

    print(f"saved {curves_png}")
    print(f"saved {anatomy_png}")
    print(f"saved {slices_png}")
    if show:
        print("opening the charts - close the windows to finish")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
