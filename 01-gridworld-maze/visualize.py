"""
visualize.py — THE REPORT CARD (step 5 of our RL project)
==========================================================

Steps 1-4 built the world, the brain, the school, and the show
window. Step 5 reads the receipts: WHERE and HOW the agent learned.

Two charts, saved next to this file in charts/ and opened as windows:

    learning_curves.png   "did it learn, and WHEN?"
    policy_map.png        "what exactly did it learn?"

HOW TO READ THEM (the actual lesson):

learning_curves.png — a 2x2 grid
  top-left     reward per episode. Raw episode scores are NOISY —
               always trust the bold moving-average line; a rising
               trend means learning is happening.
  top-right    rolling win rate (%). The single number that matters:
               how often the last 100 episodes ended in the goal.
  bottom-left  exploration rate (epsilon). Starts at 1.0 (pure
               wandering), decays toward 0.05. Watch it FALL while
               the win rate CLIMBS — that handover from exploring
               to exploiting is the heart of epsilon-greedy.
  bottom-right steps per episode. Falls when the agent finds
               SHORTER paths — the -0.1/step penalty teaching speed.

policy_map.png — the Q-table drawn on top of the maze
  arrow in a cell  = the best action Q-learning picked there
  color of cell    = how good the cell is (red bad -> green good)
  gray cell        = never visited (the agent's blind spots)
  green / red cell = goal / pit,  blue outline = start

Run it (after python train.py):

    python visualize.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, Rectangle

from agent import QAgent
from environment import (
    MAZE_MAP, MAZE_CHARS, START_CHAR,
    UP, DOWN, LEFT, RIGHT, WALL, GOAL, PIT, MAX_STEPS,
)
# Reuse train.py's EXACT output paths — one source of truth, so the
# charts can never point at a file training doesn't actually write.
from train import SAVE_LOG, SAVE_Q

# All output lives next to this file, whatever folder you ran from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------
# CHART COLORS — one per line on the learning curves
# ---------------------------------------------------------------
C_REWARD = "steelblue"    # total reward per episode
C_WIN = "seagreen"        # rolling win rate
C_EPS = "darkorange"      # exploration rate
C_STEPS = "dimgray"       # steps per episode

# Policy-map cell colors (hex, so they match what you see on screen)
C_WALL = "#31363f"
C_GOAL = "#50dc78"
C_PIT = "#ff5f55"
C_NEVER = "#e6e6e6"       # visited by nobody
C_START = "#3b82f6"       # blue outline around the start cell

# One unicode arrow per action — the policy map's alphabet.
ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}


# ==============================================================
# 1. THE SMOOTHER — raw episode scores are noisy; trends are not
# ==============================================================
def moving_average(values, window=100):
    """Rolling mean: for each position, the average of the `window`
    values ending there (fewer at the very start — honest, not padded).

    This one line is why charts are readable at all: single episodes
    swing from -10 to +10, but the AVERAGE of 100 tells the trend.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)          # growing at first
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def load_history(path=SAVE_LOG):
    """Read training_log.json (what train.py wrote)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ==============================================================
# 2. CHART ONE — did it learn, and when?
# ==============================================================
def plot_learning_curves(history, window=100, save_path=None):
    """Turn the training diary into a 2x2 figure. Returns the figure
    (tests grab it; the command line just shows it)."""
    required = ("episode", "reward", "outcome", "epsilon", "steps")
    missing = [key for key in required if key not in history]
    if missing:
        raise ValueError(f"training log is missing {missing} "
                         f"— re-run: python train.py")

    ep = history["episode"]
    rew = history["reward"]
    out = history["outcome"]
    eps = history["epsilon"]
    steps = history["steps"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    (ax_rew, ax_win), (ax_eps, ax_steps) = axes

    # --- top-left: reward, raw + moving average -------------------
    ax_rew.plot(ep, rew, lw=1, alpha=0.3, color=C_REWARD,
                label="each episode")
    ax_rew.plot(ep, moving_average(rew, window), lw=2.4,
                color=C_REWARD, label=f"{window}-episode average")
    ax_rew.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax_rew.set_title("Total reward per episode")
    ax_rew.set_ylabel("reward")
    ax_rew.legend(loc="upper left", fontsize=8)

    # Report card in the corner: the last 100 episodes, summarized.
    tail = rew[-100:]
    if tail:
        tail_wins = sum(1 for o in out[-100:] if o == "goal")
        ax_rew.text(
            0.99, 0.05,
            f"last {len(tail)} episodes:\n"
            f"avg {sum(tail) / len(tail):+.2f}   "
            f"{tail_wins}/{len(tail)} won",
            transform=ax_rew.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="gray", alpha=0.9),
        )

    # --- top-right: rolling win rate ------------------------------
    win_pct = [100.0 if o == "goal" else 0.0 for o in out]
    ax_win.plot(ep, moving_average(win_pct, window), lw=2.4, color=C_WIN)
    ax_win.set_ylim(-3, 103)
    ax_win.set_title("Win rate (rolling 100 episodes)")
    ax_win.set_ylabel("% won")

    # --- bottom-left: exploration decaying ------------------------
    ax_eps.plot(ep, eps, lw=2, color=C_EPS)
    ax_eps.set_ylim(0, 1.05)
    ax_eps.set_title("Exploration rate ε (wander vs exploit)")
    ax_eps.set_ylabel("ε")

    # --- bottom-right: fewer steps = shorter paths ----------------
    ax_steps.plot(ep, steps, lw=1, alpha=0.3, color=C_STEPS,
                  label="each episode")
    ax_steps.plot(ep, moving_average(steps, window), lw=2.4,
                  color=C_STEPS, label=f"{window}-episode average")
    ax_steps.axhline(MAX_STEPS, color="red", lw=1, ls=":",
                     label="timeout")
    ax_steps.set_title("Steps per episode (lower = shorter path)")
    ax_steps.set_ylabel("steps")
    ax_steps.set_xlabel("episode")
    ax_steps.legend(fontsize=8)
    ax_win.set_xlabel("episode")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle("Training progress — did it learn?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 3. CHART TWO — what exactly did it learn?
# ==============================================================
def plot_policy_map(agent, save_path=None):
    """Draw the agent's Q-table on top of the maze itself.

    Every cell gets a face color (its best Q-value through the
    red->green heat scale) and, if visited, an arrow showing the
    action the agent would take standing there.
    """
    rows, cols = len(MAZE_MAP), len(MAZE_MAP[0])

    # Best Q per visited cell = "how good is it to BE here".
    values = {state: max(q) for state, q in agent.q_table.items()}
    # Degenerate scales (no visits, or all equal) still need limits.
    vmin = min(values.values(), default=-1.0)
    vmax = max(values.values(), default=1.0)
    norm = plt.Normalize(vmin, vmax)
    heat = colormaps["RdYlGn"]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # --- 1. paint every cell -------------------------------------
    for r in range(rows):
        for c in range(cols):
            cell = MAZE_CHARS[MAZE_MAP[r][c]]
            if cell == WALL:
                face = C_WALL
            elif cell == GOAL:
                face = C_GOAL
            elif cell == PIT:
                face = C_PIT
            elif (r, c) in values:
                face = heat(norm(values[(r, c)]))
            else:
                face = C_NEVER          # never stood here: no data
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=face, edgecolor="white",
                                   lw=1.5))

    # --- 2. arrows + values on top of the paint -------------------
    for (r, c), q in agent.q_table.items():
        action = max(range(len(q)), key=lambda a: q[a])
        ax.text(c, r - 0.12, ARROWS[action], ha="center", va="center",
                fontsize=22, fontweight="bold", color="#1b1b1b",
                zorder=4)
        ax.text(c, r + 0.33, f"{max(q):+.1f}", ha="center", va="center",
                fontsize=8, color="#333333", zorder=4)

    # --- 3. name the terminals, ring the start --------------------
    for r in range(rows):
        for c in range(cols):
            cell = MAZE_CHARS[MAZE_MAP[r][c]]
            if cell in (GOAL, PIT):
                ax.text(c, r, MAZE_MAP[r][c], ha="center", va="center",
                        fontsize=16, fontweight="bold", color="white",
                        zorder=5)
            elif MAZE_MAP[r][c] == START_CHAR:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                       facecolor="none",
                                       edgecolor=C_START, lw=3,
                                       zorder=5))

    # --- 4. axis cosmetics ---------------------------------------
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)       # flipped: row 0 stays on top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Learned policy — arrow = best move, "
                 "color = cell value", fontsize=12, pad=12)

    legend = [
        Patch(facecolor=C_WALL, label="wall"),
        Patch(facecolor=C_GOAL, label="goal (+10)"),
        Patch(facecolor=C_PIT, label="pit (-10)"),
        Patch(facecolor=C_NEVER, label="never visited"),
        Patch(facecolor="none", edgecolor=C_START, lw=2, label="start"),
    ]
    ax.legend(handles=legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False,
              fontsize=9)

    # Heat legend: a color strip from "worst visited cell" to best.
    mappable = ScalarMappable(norm=norm, cmap=heat)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02,
                 label="cell value (best Q)")

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 4. MAIN — build both charts, save them, open them
# ==============================================================
def main(log_path=SAVE_LOG, q_path=SAVE_Q, out_dir=CHART_DIR, show=True):
    """Generate the charts. Returns 0 on success, 1 on missing input
    (so `echo %errorlevel%` and CI can both see failures)."""
    missing = [p for p in (log_path, q_path) if not os.path.exists(p)]
    if missing:
        print(f"missing {os.path.basename(missing[0])} "
              f"- train first:\n    python train.py")
        return 1

    history = load_history(log_path)
    agent = QAgent.load(q_path)

    os.makedirs(out_dir, exist_ok=True)
    curves_png = os.path.join(out_dir, "learning_curves.png")
    policy_png = os.path.join(out_dir, "policy_map.png")

    plot_learning_curves(history, save_path=curves_png)
    plot_policy_map(agent, save_path=policy_png)

    print(f"saved {curves_png}")
    print(f"saved {policy_png}")
    if show:
        print("opening the charts — close the windows to finish")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
