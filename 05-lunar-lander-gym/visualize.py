"""
visualize.py — THE REPORT CARD (step 5 of the lunar-lander-gym project)
=======================================================================

Steps 1-4 built the gym world, the actor-critic brain, the school,
and the cockpit. Step 5 reads the training diary and answers the
question this whole project exists to answer:

    DID THE POLICY-NET FAMILY (A2C) get where the DQN family
    could not? (04's tribute: six runs, two retunes, zero landings.)

Two charts next to this file in charts/:

    learning_curves.png   "did it learn, and WHEN?"
        return per episode  raw (faint), 100-run average (bold), and
             the GREEDY exams as dots — plus the gym solved bar
             (average return 200 over the reference's 20-episode
             window) as a dashed line: crossing it with the
             window-avg is what the reference calls "solved"
        steps per episode   deaths land in ~90 ticks, a stall uses
             the whole 1,000 — where the curve settles tells
             patience from panic
        loss per episode    the A2C loss (one per episode) — it
             tells SPIKES, not goodness (a bigger loss just means
             the episode's discounted math got harder)
        outcome mix         landed / crashed / out_of_bounds /
             timeout rolling share: the policy migration map

    policy_slices.png     "what would the policy do from HERE?"
        the actor's argmax over four flat cuts of the 8-D state
        (legs pinned at 0); one color per button, like 04's DQN
        slices — but the net answers in PROBABILITIES now, so a
        cell's color is ITS WINNER (argmax), not a Q-score.

No epsilon chart this time ON PURPOSE: the actor-critic has NO
epsilon dial — exploration is sampling, which decays only as the
software learns to concentrate its own probabilities.

Run it (after python train.py — or press V inside play.py):

    python visualize.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import json     # training_log.json: the diary step 3's Trainer wrote
import math     # radians() for the attitude slice axes, isfinite
import os       # paths anchored next to THIS file (not the cwd)
import sys      # exit code for CI (0 = charts built, 1 = train first)

import matplotlib
matplotlib.use("Agg")           # no windows — files only (tests work
                                # headless, the 03+ convention)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # legend swatches

# The brain's greedy head answers the slice cells; the artifacts are
# train.py's EXACT paths — one source of truth (reuse, don't retype).
from agent import ActorCritic
from train import LOG_PATH, MODEL_PATH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------
# CHART COLORS — one per button / per outcome (04's palette, kept)
# ---------------------------------------------------------------
C_RETURN = "steelblue"      # per-episode return
C_EXAM   = "black"          # greedy-exam dots
C_STEPS  = "seagreen"       # episode length
C_LOSS   = "firebrick"      # the A2C loss curve
C_BAR    = "seagreen"       # the gym solved bar (avg 200)
C_OUT = {                   # how episodes end (rolling share)
    "landed": "seagreen",
    "crashed": "#e2574c",
    "out_of_bounds": "#8d93a1",
    "timeout": "#b8860b",
}
ACTION_COLORS = {           # the actor's four possible winners
    0: "#9aa3ad",           # gray   — IDLE
    1: "#f59e0b",           # orange — MAIN
    2: "#3b82f6",           # blue   — LEFT
    3: "#8b5cf6",           # violet — RIGHT
}
ACTION_LABELS = ("IDLE", "MAIN", "LEFT", "RIGHT")

# Policy-slice resolution (cells per panel axis — 384 argmax calls
# per panel across four panels is instant).
SLICE_GRID = (24, 16)

# The gym solved bar, named: the reference's solve arithmetic is
# "running sum over 20 episodes passes 4000", == average 200.
SOLVED_BAR = 200.0


# ==============================================================
# 1. THE SMOOTHER — raw episodes are noise; trends are not
# ==============================================================
def moving_average(values, window=100):
    """Rolling mean: the average of the `window` values ENDING at
    each position (fewer at the start — honest, not padded).

    Hand-checkable everywhere; test_visualize.py pins it.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def load_history(path=LOG_PATH):
    """Read + validate the diary (train.py's schema, echoed here).

    Missing keys SCREAM with the exact name — a silent KeyError
    three charts later is not a stack trace you can act on.
    """
    with open(path, encoding="utf-8") as f:
        history = json.load(f)
    for key in ("config", "episodes", "exams", "summary"):
        if key not in history:
            raise KeyError(f"training_log.json missing '{key}' - "
                           "retrain (python train.py)")
    for rec in history["episodes"]:
        for key in ("episode", "return", "steps", "outcome", "loss"):
            if key not in rec:
                raise KeyError(f"episode record missing '{key}': {rec}")
    return history


def rolling_share(episodes, outcome, window=100):
    """Share of episodes ending in `outcome`, per rolling window.

    Hand-checkable: window 1 degenerates to 1.0/0.0; a window of
    two crashed episodes reads 1.0 crashed.
    """
    shares = []
    for i in range(len(episodes)):
        chunk = episodes[max(0, i - window + 1):i + 1]
        count = sum(1 for e in chunk if e["outcome"] == outcome)
        shares.append(count / len(chunk))
    return shares


# ==============================================================
# 2. LEARNING CURVES — did it learn, and WHEN?
# ==============================================================
def plot_learning_curves(history, window=100, save_path=None):
    """The 2x2: return / steps / loss / outcome mix."""
    episodes = history["episodes"]
    exams = history["exams"]
    returns = [e["return"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    losses = [e["loss"] for e in episodes]
    idx = [e["episode"] for e in episodes]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("05 lunar-lander-gym - Actor-Critic learning curves")

    # (1) return + the gym solved bar
    ax = axes[0][0]
    ax.plot(idx, returns, color=C_RETURN, alpha=0.25, linewidth=0.8,
            label="return / episode")
    ax.plot(idx, moving_average(returns, window), color=C_RETURN,
            linewidth=2.0, label=f"avg({window})")
    for exam in exams:
        ax.plot(exam["episode"], exam["avg_return"], "o",
                color=C_EXAM, markersize=4,
                label="greedy exam" if exam is exams[0] else None)
    ax.axhline(SOLVED_BAR, color=C_BAR, linestyle="--", linewidth=1.2,
               label="gym solved bar (avg 200)")
    ax.set_xlabel("episode"); ax.set_ylabel("return")
    ax.legend(loc="lower right", fontsize=8)

    # (2) steps per episode
    ax = axes[0][1]
    ax.plot(idx, steps, color=C_STEPS, alpha=0.25, linewidth=0.8)
    ax.plot(idx, moving_average(steps, window), color=C_STEPS,
            linewidth=2.0)
    ax.plot([], [], color=C_STEPS, label=f"steps avg({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("steps per episode")
    ax.legend(fontsize=8)

    # (3) the A2C loss
    ax = axes[1][0]
    ax.plot(idx, losses, color=C_LOSS, alpha=0.25, linewidth=0.8,
            label="loss / episode")
    ax.plot(idx, moving_average(losses, window), color=C_LOSS,
            linewidth=2.0, label=f"loss avg({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode loss")
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.99, 0.02, "loss magnitude is NOT model quality",
            transform=ax.transAxes, ha="right", fontsize=8,
            color="gray")

    # (4) outcome mix, rolling
    ax = axes[1][1]
    for outcome, color in C_OUT.items():
        share = rolling_share(episodes, outcome, window)
        ax.plot(idx, share, color=color, linewidth=1.4, label=outcome)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"share of endings (avg {window})")
    ax.legend(loc="center left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


# ==============================================================
# 3. POLICY SLICES — what would the policy do from HERE?
# ==============================================================
SLICES = (
    # (title, x-label, y-label, slice dim A, dim B, FIXED STATE)
    # fixed is a FULL 8-float state; the two slice dims sit as None
    # placeholders that peek_action swaps per cell (legs are always
    # the last two: 0 = in the air).
    ("approach: (x, vx) at y 0.4, level, slow",
     "x (pad-normalized)", "vx",
     0, 2, (None, 0.4, None, 0.0, 0.0, 0.0, 0, 0)),
    ("descent: (y, vy) at x 0, level, slow",
     "y (pad-normalized)", "vy",
     1, 3, (0.0, None, 0.0, None, 0.0, 0.0, 0, 0)),
    ("attitude: (angle, omega) at x 0, y 0.4, slow",
     "angle (rad)", "omega",
     4, 5, (0.0, 0.4, 0.0, 0.0, None, None, 0, 0)),
    ("approach map: (x, y) level, slow",
     "x (pad-normalized)", "y",
     0, 1, (None, None, 0.0, 0.0, 0.0, 0.0, 0, 0)),
)

# Slice ranges, gym's own state scales (pad-normalized, +-1 walls):
SLICE_X     = (-1.0, 1.0)
SLICE_Y     = (-0.4, 1.2)      # kill-floor to spawn ceiling
SLICE_VV    = (-4.0, 4.0)
SLICE_TILT  = (-math.pi, math.pi)
SLICE_OMEGA = (-4.0, 4.0)


def peek_action(policy, a_dim, b_dim, a_lo, a_hi, b_lo, b_hi,
                grid=SLICE_GRID, fixed=None):
    """The actor's greedy winner for ONE cell of a slice panel.

    Builds a state by swapping the two slice dimensions through the
    fixed others (legs pinned at 0 = in the air); every cell gets
    its own argmax call — 24x16 calls per panel, instant for a
    net this size.
    """
    if fixed is None:
        raise ValueError("peek_action needs the 8-dim fixed state")
    a_values = [a_lo + (a_hi - a_lo) * j / (grid[0] - 1)
                for j in range(grid[0])]
    b_values = [b_lo + (b_hi - b_lo) * i / (grid[1] - 1)
                for i in range(grid[1])]
    winners = {}
    for i, b in enumerate(b_values):
        for j, a in enumerate(a_values):
            state = list(fixed)
            state[a_dim] = a
            state[b_dim] = b
            winners[(j, i)] = policy.act_greedy(tuple(state))
    return winners


def hex_to_rgb(color):
    """'#rrggbb' -> (r, g, b) floats 0..1 (imshow wants numeric
    planes; hex strings would need object dtype and protests)."""
    color = color.lstrip("#")
    return tuple(int(color[k:k + 2], 16) / 255.0
                 for k in (0, 2, 4))


def plot_policy_slices(policy, save_path=None, grid=SLICE_GRID):
    """Four cuts, one color per winner."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Actor's argmax over four state cuts")

    for (title, xlabel, ylabel, ai, bi, fixed), ax in zip(SLICES,
                                                          axes.flat):
        a_lo, a_hi = _slice_range(ai)
        b_lo, b_hi = _slice_range(bi)
        winners = peek_action(policy, ai, bi,
                              a_lo, a_hi, b_lo, b_hi,
                              grid=grid, fixed=fixed)
        xs = [a_lo + (a_hi - a_lo) * j / (grid[0] - 1)
              for j in range(grid[0])]
        ys = [b_lo + (b_hi - b_lo) * i / (grid[1] - 1)
              for i in range(grid[1])]
        # Color grid as numeric (H, W, 3): row i (the b/slave axis,
        # imshow rows), column j (the a/slave axis):
        import numpy as np
        color_arr = np.zeros((len(ys), len(xs), 3))
        for i in range(len(ys)):
            for j in range(len(xs)):
                color_arr[i][j] = hex_to_rgb(ACTION_COLORS[winners[(j, i)]])
        ax.imshow(color_arr,
                  extent=[min(xs), max(xs), min(ys), max(ys)],
                  origin="lower", aspect="auto",
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    handles = [Patch(facecolor=ACTION_COLORS[a],
                     label=ACTION_LABELS[a]) for a in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


def _slice_range(dimension):
    """The named axis ranges above, read back by code (no magic)."""
    return {
        0: SLICE_X,
        1: SLICE_Y,
        2: SLICE_VV,
        3: SLICE_VV,
        4: SLICE_TILT,
        5: SLICE_OMEGA,
    }[dimension]


# ==============================================================
# 4. main() — the whole report card
# ==============================================================
def main(log_path=LOG_PATH, model_path=MODEL_PATH, out_dir=CHART_DIR,
         window=100):
    """Build both charts from the diary+diploma; exit 0 / 1 for CI."""
    if not os.path.exists(log_path):
        print(f"missing {log_path} - train first:")
        print("    python train.py")
        return 1
    history = load_history(log_path)

    os.makedirs(out_dir, exist_ok=True)
    curves_png = os.path.join(out_dir, "learning_curves.png")
    slices_png = os.path.join(out_dir, "policy_slices.png")

    plot_learning_curves(history, window=window, save_path=curves_png)
    print(f"saved {curves_png}")

    if os.path.exists(model_path):
        policy = ActorCritic.load(model_path)
        plot_policy_slices(policy, save_path=slices_png)
        print(f"saved {slices_png}")
    else:
        print(f"no brain at {model_path} - skips policy slices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
