"""
visualize.py — THE REPORT CARD (step 5 of our cart-pole project)
=================================================================

Steps 1-4 built the world, the brain, the school, and the show
window. Step 5 reads the receipts: WHEN it learned, and WHAT it
learned to do about a falling pole.

Two charts, saved next to this file in charts/ and opened as windows:

    learning_curves.png   "did it learn, and WHEN?"
    policy_slice.png      "what would it do from HERE?"

HOW TO READ THEM (the actual lesson):

learning_curves.png — a 2x2 grid
  top-left     survival STEPS per episode. Remember: in this game
               reward IS steps (+1 per tick alive), so they are the
               SAME number — and here LONGER is better, the exact
               opposite of project 1's "find the shortest path"
               lesson. Raw episodes are noisy; trust the bold
               100-episode average line.
  top-right    rolling win rate. A "win" = surviving all 500 ticks
               to the time limit. The single number that matters.
  bottom-left  exploration rate (epsilon). Starts at 1.0 (pure
               wandering), decays to 0.05. Watch it FALL while the
               win rate CLIMBS — that handover from exploring to
               exploiting is the heart of epsilon-greedy.
  bottom-right how episodes DIE. Two failure modes: cart_out (the
               flailing beginner — cart flies off the track) and
               pole_fell (close! cart stayed on track but the pole
               tipped past 12 degrees). WHICH one dominates tells
               you what the brain still gets wrong.

policy_slice.png — the Q-table has FOUR dimensions, so we cut it
into two flat slices (like a CT scan of the policy):
  left panel   RECOVERY map: cart pinned at x=0, speed 0, showing
               the brain's choice over (pole tilt, pole spin).
               This is the heart of balancing: lean right -> push
               right to run under the falling tip.
  right panel  POSITION map: pole pinned perfectly upright at
               rest, over (cart position, cart speed) — how the
               brain herds the cart back to the center.

  blue cell    = the brain would push LEFT there
  orange cell  = the brain would push RIGHT there
  gray cell    = never visited during training (blind spots —
                 exactly the buckets the fall-penalty lesson had
                 to rescue)

Run it (after python train.py — or press V inside play.py):

    python visualize.py
"""

import json
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from agent import QAgent
from environment import MAX_STEPS
# Reuse train.py's EXACT output paths — one source of truth, so the
# charts can never point at a file training doesn't actually write.
from train import SAVE_LOG, SAVE_Q

# All output lives next to this file, whatever folder you ran from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, "charts")

# ---------------------------------------------------------------
# CHART COLORS — one per line on the learning curves
# ---------------------------------------------------------------
C_STEPS = "steelblue"      # survival steps per episode
C_WIN = "seagreen"         # rolling win rate
C_EPS = "darkorange"       # exploration rate
C_FALL = "#e2574c"         # rolling % ended by the pole falling
C_OUT = "#8d93a1"          # rolling % ended by the cart leaving

# Policy-slice cell colors (hex, so they match what you see on screen)
C_LEFT = "#3b82f6"         # push LEFT
C_RIGHT = "#f59e0b"        # push RIGHT
C_NEVER = "#e6e6e6"        # never visited: the brain's blind spots

# Slice resolution: 24 x 16 cells per panel (768 little rectangles
# total — fine enough to see the decision boundary, coarse enough
# that each cell still corresponds to a real bucket region).
SLICE_GRID = (24, 16)


# ==============================================================
# 1. THE SMOOTHER — raw episode scores are noisy; trends are not
# ==============================================================
def moving_average(values, window=100):
    """Rolling mean: for each position, the average of the `window`
    values ending there (fewer at the very start — honest, not padded).

    This one line is why charts are readable at all: single episodes
    swing from 3 ticks to 500, but the AVERAGE of 100 tells the trend.
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
    out = history["outcome"]
    eps = history["epsilon"]
    steps = history["steps"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    (ax_steps, ax_win), (ax_eps, ax_fail) = axes

    # --- top-left: survival steps, raw + moving average -------------
    # (reward would draw the SAME line — it equals steps here.)
    ax_steps.plot(ep, steps, lw=1, alpha=0.3, color=C_STEPS,
                  label="each episode")
    ax_steps.plot(ep, moving_average(steps, window), lw=2.4,
                  color=C_STEPS, label=f"{window}-episode average")
    ax_steps.axhline(MAX_STEPS, color="red", lw=1, ls=":",
                     label="the 500-tick limit")
    ax_steps.set_title("Survival steps (higher = better)")
    ax_steps.set_ylabel("steps")
    ax_steps.legend(loc="upper left", fontsize=8)

    # Report card in the corner: the last 100 episodes, summarized.
    tail = steps[-100:]
    if tail:
        tail_wins = sum(1 for o in out[-100:] if o == "time_limit")
        ax_steps.text(
            0.99, 0.05,
            f"last {len(tail)} episodes:\n"
            f"avg {sum(tail) / len(tail):.1f} steps   "
            f"{tail_wins}/{len(tail)} hit the limit",
            transform=ax_steps.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="gray", alpha=0.9),
        )

    # --- top-right: rolling win rate -------------------------------
    win_pct = [100.0 if o == "time_limit" else 0.0 for o in out]
    ax_win.plot(ep, moving_average(win_pct, window), lw=2.4,
                color=C_WIN)
    ax_win.set_ylim(-3, 103)
    ax_win.set_title("Win rate (rolling 100 episodes)")
    ax_win.set_ylabel("% hit limit")

    # --- bottom-left: exploration decaying -------------------------
    ax_eps.plot(ep, eps, lw=2, color=C_EPS)
    ax_eps.set_ylim(0, 1.05)
    ax_eps.set_title("Exploration rate ε (wander vs exploit)")
    ax_eps.set_ylabel("ε")

    # --- bottom-right: WHICH death? --------------------------------
    # Two failure modes tell two different stories: cart_out =
    # "can't steer yet", pole_fell = "steers but can't balance".
    fall_pct = [100.0 if o == "pole_fell" else 0.0 for o in out]
    out_pct = [100.0 if o == "cart_out" else 0.0 for o in out]
    ax_fail.plot(ep, moving_average(fall_pct, window), lw=2,
                 color=C_FALL, label="pole fell (tipped past 12°)")
    ax_fail.plot(ep, moving_average(out_pct, window), lw=2,
                 color=C_OUT, label="cart left the track")
    ax_fail.set_ylim(-3, 103)
    ax_fail.set_title("How episodes die (rolling 100)")
    ax_fail.set_ylabel("% of episodes")
    ax_fail.set_xlabel("episode")
    ax_fail.legend(fontsize=8)
    ax_win.set_xlabel("episode")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle("Cart-pole training — did it learn?", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# ==============================================================
# 3. CHART TWO — what would it do from HERE?
# ==============================================================
def peek_action(agent, state):
    """The brain's chosen action for a raw state — WITHOUT creating
    a new row.

    get_q() would happily write fresh optimism rows for every cell
    we sample; then every gray "never visited" cell would turn blue
    or orange and the chart would LIE about what was learned.
    Read-only peek: missing row -> None (draw it gray instead).
    """
    q = agent.q_table.get(agent.binner(state))
    if q is None:
        return None
    return max(range(len(q)), key=lambda a: q[a])


def plot_policy_slice(agent, save_path=None):
    """Draw two 2D slices of the four-dimensional Q-table.

    Slice A: cart frozen at the center (x=0, x_dot=0); the plane of
             (theta, theta_dot) — the recovery decision.
    Slice B: pole frozen upright at rest (theta=0, theta_dot=0);
             the plane of (x, x_dot) — the position decision.

    Axis ranges come from THE AGENT'S OWN BINNER — the chart always
    shows the world as this brain actually sees it, even if it was
    trained with different bucket lines.
    """
    ranges = agent.binner.ranges          # ((x), (x_dot), (th), (thd))
    nx, ny = SLICE_GRID

    fig, (ax_rec, ax_pos) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    def draw_panel(ax, x_range, y_range, state, x_dim, y_dim, xlabel,
                   ylabel, title, x_is_degrees=False):
        """Fill `ax` with a grid of cells; each cell shows what the
        brain would DO at its center (LEFT / RIGHT / never-visited).

        state: the full four-float starting point — the two chosen
        dimensions get overwritten by the grid, the other two stay
        fixed. That's what makes this a "slice" of the 4D policy.
        x_dim / y_dim: which of (x, x_dot, theta, theta_dot) the
        axes' values are written into (0, 1, 2 or 3).
        x_is_degrees: the tilt axis is DRAWN in degrees (the unit
        the environment talks in) but the brain reads radians —
        convert back before probing, or every cell lands in the
        wrong bucket.
        """
        (xl, xh), (yl, yh) = x_range, y_range
        dx, dy = (xh - xl) / nx, (yh - yl) / ny
        for i in range(nx):
            for j in range(ny):
                probe = list(state)
                xv = xl + (i + 0.5) * dx        # x-axis value
                probe[x_dim] = math.radians(xv) if x_is_degrees else xv
                probe[y_dim] = yl + (j + 0.5) * dy   # y-axis value
                action = peek_action(agent, probe)
                face = (C_NEVER if action is None else
                        C_LEFT if action == 0 else C_RIGHT)
                ax.add_patch(Rectangle(
                    (xl + i * dx, yl + j * dy), dx, dy,
                    facecolor=face, edgecolor="white", lw=0.6))
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    # --- left panel: RECOVERY (x fixed at 0, cart speed fixed 0) ----
    # The tilt axis lives in DEGREES end to end (ticks included) —
    # "12° or the pole dies" is the sentence worth reading.
    tilt_lo, tilt_hi = (math.degrees(v) for v in ranges[2])
    draw_panel(
        ax_rec, (tilt_lo, tilt_hi), ranges[3],
        state=(0.0, 0.0, 0.0, 0.0),
        x_dim=2, y_dim=3,          # the grid owns theta + theta_dot
        xlabel="pole tilt θ (degrees)",
        ylabel="pole spin θ̇ (rad/s)",
        title="Recovery: catch the falling pole (x = 0)",
        x_is_degrees=True,
    )
    ax_rec.set_xticks([tilt_lo + (tilt_hi - tilt_lo) * k / 4
                       for k in range(5)])
    ax_rec.set_xticklabels(
        [f"{tilt_lo + (tilt_hi - tilt_lo) * k / 4:.0f}°"
         for k in range(5)])
    # matplotlib >= 3.11: set_xticks rescales the axis to the tick
    # span — put the data limits back as the very last word.
    ax_rec.set_xlim(tilt_lo, tilt_hi)

    # --- right panel: POSITION (pole pinned upright at rest) --------
    draw_panel(
        ax_pos, ranges[0], ranges[1],
        state=(0.0, 0.0, 0.0, 0.0),
        x_dim=0, y_dim=1,          # the grid owns x + x_dot
        xlabel="cart position x (m)",
        ylabel="cart speed ẋ (m/s)",
        title="Position: herd the cart home (θ = 0)",
    )

    legend = [
        Patch(facecolor=C_LEFT, label="would push LEFT"),
        Patch(facecolor=C_RIGHT, label="would push RIGHT"),
        Patch(facecolor=C_NEVER, label="never visited"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learned policy in two slices", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

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
    policy_png = os.path.join(out_dir, "policy_slice.png")

    plot_learning_curves(history, save_path=curves_png)
    plot_policy_slice(agent, save_path=policy_png)

    print(f"saved {curves_png}")
    print(f"saved {policy_png}")
    if show:
        print("opening the charts — close the windows to finish")
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
