"""
play.py — THE WINDOW (part C of our RL project)
===============================================

Until now the world lived in the terminal. This file gives it a real
game window: colored tiles, a HUD, and four ways to interact.

Run it (from this folder):

    python play.py

Which maze?  The classic one on your FIRST run. After you press N,
a freshly generated random maze becomes THE maze — saved to
maze_random.json, loaded again on every future launch, replaced only
when you press N again.

Controls
--------
    Arrow keys ... move the agent        (HUMAN mode)
    + / - ........ speed up / slow down  (WATCH / LIVE / TRAINED)
    N ............ generate a NEW random maze (the agent retrains)
    R ............ restart the episode
    SPACE ........ cycle to the next mode
    TAB .......... skip to the next training stage   (LIVE mode)
    V ............ open the charts of the trained agent
    ESC or Q ..... quit

Modes
-----
    HUMAN   : you play the maze yourself — the best way to FEEL what
              the task demands before asking an agent to learn it.
    WATCH   : a mindless random agent stumbles around by itself —
              the baseline to beat.
    LIVE    : watch the agent LEARN. A fresh training run advances in
              the background while the board shows its current best
              play; the HUD counts episodes and walks through the
              three stages EXPLORING -> LEARNING -> POLISHING.
              + / - changes the training rate, TAB fast-forwards to
              the next stage. Forgot to launch training? A warning
              is dumped on screen and training starts by itself.
    TRAINED : the GRADUATE plays greedily (no random moves, no
              learning — pure performance). Needs q_table.json;
              if it's missing, a warning appears and training is
              launched automatically (~0.1 s).

The window talks to the world only through the same two calls
everything else uses (reset / step) — exactly how Gymnasium's own
environments render themselves. Nothing here knows the maze rules;
environment.py remains the single source of truth.
"""

import os
import random
import subprocess
import sys

import pygame

from agent import QAgent
from environment import (
    GridWorld, UP, DOWN, LEFT, RIGHT, N_ACTIONS, MAX_STEPS,
    EMPTY, WALL, GOAL, PIT,
    load_active_maze, generate_random_maze, save_random_maze,
)
from train import Trainer, STAGE_NAMES

# Everything lives next to this file, whatever folder you ran from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The window sizes itself from the actual maze, so the drawing code
# can never drift out of sync with environment.py.
_probe = GridWorld()
ROWS, COLS = _probe.rows, _probe.cols
del _probe

# ---------------------------------------------------------------
# LAYOUT — pure geometry: how big are cells, padding, the window?
# ---------------------------------------------------------------
CELL = 72           # pixels per maze cell
PAD = 16            # border around the board
GAP = 14            # space between board and HUD
HUD_H = 76          # height of the info bar

WIDTH = PAD * 2 + COLS * CELL          # 536
BOARD_H = ROWS * CELL                  # 504
HEIGHT = PAD + BOARD_H + GAP + HUD_H + PAD

FPS = 60            # redraws per second (the window itself)

# ---------------------------------------------------------------
# WATCH MODE SPEED — the +- keys walk up/down this ladder
# ---------------------------------------------------------------
# Each entry is milliseconds between random moves (slow -> fast).
# The window shows the current level as a multiplier relative to the
# default 260ms:  800ms = 0.3x ... 260ms = 1.0x ... 50ms = 5.2x
SPEED_LEVELS_MS = [800, 500, 260, 150, 90, 50]
SPEED_DEFAULT_I = 2                        # the 260ms entry
SPEED_BASE_MS = SPEED_LEVELS_MS[SPEED_DEFAULT_I]   # "1.0x" reference

# ---------------------------------------------------------------
# LIVE MODE SPEED — how many training episodes fly by per second
# ---------------------------------------------------------------
# The + / - keys walk this ladder while in LIVE mode (same speed_i
# index as WATCH/TRAINED — both ladders have six rungs, default 30).
# At 500 ep/s the whole 2000-episode session flashes past in four
# seconds; at 4 ep/s you can watch epsilon decay in real time.
LIVE_EPS_PER_SEC = [4, 10, 30, 80, 200, 500]

# How fast the "current best play" demo animates on the board.
DEMO_STEP_MS = 160

WATCH_PAUSE_MS = 1400   # watch mode: admire the result before restart

# ---------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------
C_BG       = (28, 30, 36)      # window background
C_FLOOR    = (236, 230, 214)   # open cell (light)
C_FLOOR_2  = (224, 217, 198)   # open cell, checkerboard shade
C_WALL     = (86, 96, 122)     # wall
C_GOAL     = (52, 168, 83)     # goal cell (green)
C_PIT      = (158, 47, 47)     # pit cell (red)
C_AGENT    = (47, 108, 212)    # the agent (blue)
C_AGENT_RIM = (235, 241, 255)  # outline around the agent
C_START    = (140, 132, 112)   # the 'S' start marker
C_HUD_BG   = (40, 44, 54)      # HUD background
C_TEXT     = (235, 235, 235)   # HUD text
C_TEXT_DIM = (170, 175, 185)   # secondary HUD text
C_MODE_HUMAN = (90, 160, 255)  # "HUMAN" badge color
C_MODE_WATCH = (255, 170, 60)  # "WATCH" badge color
C_MODE_LIVE = (190, 140, 255)  # "LIVE" badge color (training = purple)
C_MODE_TRAINED = (110, 225, 150)  # "TRAINED" badge color
C_WIN  = (80, 220, 120)        # win banner
C_LOSE = (255, 95, 85)         # loss banner
C_TIME = (255, 210, 90)        # timeout banner

# The mode cycle that SPACE walks through, and each mode's badge color.
MODES = ("HUMAN", "WATCH", "LIVE", "TRAINED")
MODE_COLORS = {
    "HUMAN": C_MODE_HUMAN,
    "WATCH": C_MODE_WATCH,
    "LIVE": C_MODE_LIVE,
    "TRAINED": C_MODE_TRAINED,
}


def next_mode(current):
    """The mode SPACE switches to: HUMAN -> WATCH -> LIVE -> TRAINED."""
    return MODES[(MODES.index(current) + 1) % len(MODES)]


# The trained brain lives next to this file (train.py saves it there),
# so it works no matter which folder you launched python from.
Q_TABLE_PATH = os.path.join(BASE_DIR, "q_table.json")


def load_trained_agent(path=None):
    """Load the graduate from disk, or None if there isn't one yet.

    Returning None (instead of raising) lets the window degrade
    gracefully: no q_table.json -> friendly hint banner, no crash.
    """
    try:
        return QAgent.load(Q_TABLE_PATH if path is None else path)
    except (OSError, ValueError, KeyError):
        return None        # missing file / half-written / wrong shape


def retrain():
    """Silently train a fresh agent for the ACTIVE maze (~0.1 s).

    This is the engine behind the safety net: whenever something
    needs a trained agent and none exists (or the maze changed),
    we don't ask the user to open a terminal — we just do it.
    """
    trainer = Trainer(quiet=True)
    trainer.run_all()
    trainer.finish()
    return trainer.agent


def ensure_trained(ui):
    """Return a trained agent, warning first if training was missing.

    If q_table.json exists we're done (normal case). If not, dump a
    warning banner, launch the full training right here, and return
    the result — the banner stays until a key acknowledges it, so
    the message can't be missed.
    """
    agent = load_trained_agent()
    if agent is not None:
        return agent
    ui["notice"] = "WARNING: no trained agent!"
    ui["notice_sub"] = "training was not launched - starting it now"
    return retrain()


def launch_visualizer(runner=None):
    """Open the charts of the CURRENT agent in its own process.

    A separate process keeps pygame and matplotlib from fighting
    over the windowing system (on Windows they really do). Returns
    True if the charts are on their way.
    """
    if runner is None:
        runner = subprocess.Popen
    try:
        runner([sys.executable, "visualize.py"], cwd=BASE_DIR)
        return True
    except OSError:
        return False


def new_random_maze():
    """Generate a brand-new random maze, save it as THE maze, build it.

    The saved file replaces the previous one — this is the "another
    maze is created" moment from environment.py's docs. Returns the
    new GridWorld (its start position comes from the maze's own 'S').
    """
    rows = generate_random_maze()
    save_random_maze(rows)          # written to environment's file
    return GridWorld("random")      # ...which is what this reads


def build_demo(agent, maze_name):
    """Run one greedy episode and return (positions, cum_rewards, reason).

    LIVE mode animates this on the board: instead of showing the
    messy episode currently being TRAINED, we show the agent's
    current BEST behaviour — rebuilt after every demo, so you watch
    it get better as training progresses.
    """
    env = GridWorld(maze_name)
    state = env.reset()
    positions, cum, total = [state], [], 0.0
    while True:
        action = agent.choose_action(state, explore=False)
        state, reward, done, info = env.step(action)
        total += reward
        positions.append(state)
        cum.append(total)
        if done:
            return positions, cum, info["reason"]

# Map each of our integer actions to a keyboard key.
KEY_TO_ACTION = {
    pygame.K_UP: UP,
    pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT,
    pygame.K_RIGHT: RIGHT,
}

# Fonts are cached: creating a font object is expensive, drawing
# is not. pygame.font must be initialised first (run() does that).
_FONTS = {}


def font(size):
    """A monospace font of the given pixel size (cached)."""
    if size not in _FONTS:
        _FONTS[size] = pygame.font.SysFont("consolas", size)
    return _FONTS[size]


def cell_rect(row, col):
    """The screen rectangle (x, y, w, h) of one maze cell."""
    return pygame.Rect(PAD + col * CELL, PAD + row * CELL, CELL, CELL)


def centered_text(surface, text, font_obj, color, rect, offset_y=0):
    """Draw text centered inside `rect` (with an optional vertical nudge)."""
    img = font_obj.render(text, True, color)
    pos = img.get_rect(center=rect.center)
    pos.centery += offset_y
    surface.blit(img, pos)


# ---------------------------------------------------------------
# DRAWING — one frame of the game
# ---------------------------------------------------------------
def draw_board(screen, env):
    """Paint every maze cell, then the agent on top."""
    for r in range(env.rows):
        for c in range(env.cols):
            rect = cell_rect(r, c)
            cell = env.grid[r][c]
            is_agent = (r, c) == env.agent_pos

            # 1. Floor first (checkerboard pattern under everything).
            floor = C_FLOOR if (r + c) % 2 == 0 else C_FLOOR_2
            pygame.draw.rect(screen, floor, rect)

            # 2. Cell contents — same legend as the ASCII renderer.
            if cell == WALL:
                pygame.draw.rect(screen, C_WALL, rect, border_radius=6)
            elif cell == GOAL:
                pygame.draw.rect(screen, C_GOAL, rect, border_radius=6)
                centered_text(screen, "G", font(34), C_TEXT, rect)
            elif cell == PIT:
                pygame.draw.rect(screen, C_PIT, rect, border_radius=6)
                centered_text(screen, "P", font(34), C_TEXT, rect)
            elif cell == EMPTY and (r, c) == env.start_pos and not is_agent:
                # Start cell: keep showing 'S' when the agent is away,
                # exactly like render() does in the terminal.
                centered_text(screen, "S", font(28), C_START, rect)

    # 3. The agent last, so it's never hidden behind a cell.
    if env.agent_pos is not None:
        a_rect = cell_rect(*env.agent_pos).inflate(-14, -14)
        pygame.draw.rect(screen, C_AGENT, a_rect, border_radius=12)
        pygame.draw.rect(screen, C_AGENT_RIM, a_rect, 3, border_radius=12)


def draw_hud(screen, env, ui):
    """The info bar: mode, progress, score, and the control hints."""
    hud_rect = pygame.Rect(PAD, PAD + BOARD_H + GAP, WIDTH - 2 * PAD, HUD_H)
    pygame.draw.rect(screen, C_HUD_BG, hud_rect, border_radius=10)

    # --- line 1: mode badge + the numbers that change per mode ------
    # Badge: text on a colored patch sized to fit (12px total margin).
    # Every width below is chosen to keep the worst case inside the
    # window (test_play.py proves it frame by frame).
    mode = ui["mode"]
    badge_text = font(22).render(mode, True, C_BG)
    badge_bg = pygame.Surface((badge_text.get_width() + 12,
                               badge_text.get_height()))
    badge_bg.fill(MODE_COLORS[mode])
    badge_x = hud_rect.x + 12
    screen.blit(badge_bg, (badge_x, hud_rect.y + 10))
    screen.blit(badge_text, (badge_x + 6, hud_rect.y + 10))
    info_x = badge_x + badge_bg.get_width() + 6

    if mode == "LIVE" and ui.get("live"):
        # Training dashboard: episode counter, stage number, the
        # exploration rate, and the rolling win rate. Smaller font
        # (20 not 22) so the worst case still fits next to the badge.
        live = ui["live"]
        info = font(20).render(
            f"EP {live['episode']}/{live['total']} "
            f"{live['stage'] + 1}/3 \u03b5{live['epsilon']:.2f} "
            f"W{live['win']:.0f}%",
            True, C_TEXT,
        )
        screen.blit(info, (info_x, hud_rect.y + 12))
        speed = font(17).render(
            f" {LIVE_EPS_PER_SEC[ui['speed_i']]} ep/s",
            True, MODE_COLORS["LIVE"],
        )
        screen.blit(speed,
                    (info_x + info.get_width() + 4, hud_rect.y + 14))
    else:
        info = font(22).render(
            f"  STEP {env.steps}/{MAX_STEPS}   REWARD {ui['reward']:+.1f}",
            True, C_TEXT,
        )
        screen.blit(info, (info_x, hud_rect.y + 10))

        # Timer modes (WATCH / TRAINED) show the current speed as a
        # multiplier — small font so the TRAINED badge + worst-case
        # numbers (100/100, -10.0) still fit the window.
        if mode in ("WATCH", "TRAINED"):
            speed = SPEED_BASE_MS / SPEED_LEVELS_MS[ui["speed_i"]]
            speed_img = font(17).render(f" {speed:.1f}x", True,
                                        MODE_COLORS[mode])
            screen.blit(speed_img,
                        (info_x + info.get_width() + 4, hud_rect.y + 14))

    # --- line 2: control hints (they change with the mode) -----------
    # Kept short and at font 14: Consolas is ~9px/char at font 16, so
    # even these would clip (test_play.py proves it frame by frame).
    if mode == "LIVE":
        stage_name = STAGE_NAMES[ui["live"]["stage"]] if ui.get("live") \
            else STAGE_NAMES[0]
        hint = (f"{stage_name}  [TAB] stage  [N] maze  [V] charts  "
                f"[SPACE] mode")
    elif mode == "HUMAN":
        hint = ("[arrows] move  [SPACE] modes  [N] maze  "
                "[V] charts  [R] restart")
    else:
        hint = ("[+/-] speed  [SPACE] modes  [N] maze  "
                "[V] charts  [R] restart")
    screen.blit(font(14).render(hint, True, C_TEXT_DIM),
                (hud_rect.x + 12, hud_rect.y + 48))


def draw_outcome(screen, env, ui):
    """Big banner across the middle: episode result, or a notice
    (e.g. "you picked TRAINED but there's no trained agent yet")."""
    if ui["outcome"] is None:
        # No episode result -> show the notice, if the mode set one.
        # Notices come as one or two lines (big warning + sub-line),
        # e.g. "WARNING: no trained agent!" / "training ... now".
        notice = ui.get("notice")
        if not notice:
            return
        band = pygame.Surface((WIDTH, 100), pygame.SRCALPHA)
        band.fill((10, 12, 16, 190))
        screen.blit(band, (0, (PAD + BOARD_H) // 2 - 50))
        big = font(24).render(notice, True, C_TIME)
        if ui.get("notice_sub"):
            big_y = HEIGHT // 2 - 70
            sub = font(16).render(ui["notice_sub"], True, C_TEXT_DIM)
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2,
                                                   HEIGHT // 2 - 40)))
        else:
            big_y = HEIGHT // 2 - 60
        screen.blit(big, big.get_rect(center=(WIDTH // 2, big_y)))
        return

    banner = {
        "goal":    ("GOAL REACHED!", C_WIN),
        "pit":     ("FELL INTO THE PIT!", C_LOSE),
        "timeout": ("OUT OF TIME", C_TIME),
    }
    text, color = banner[ui["outcome"]]

    band = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    band.fill((10, 12, 16, 190))
    screen.blit(band, (0, (PAD + BOARD_H) // 2 - 60))

    big = font(38).render(text, True, color)
    screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 78)))

    sub_text = "press R to restart" if ui["mode"] == "HUMAN" else "restarting..."
    sub = font(20).render(sub_text, True, C_TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 34)))


def draw(screen, env, ui):
    """Render one complete frame (board -> HUD -> banner)."""
    screen.fill(C_BG)
    draw_board(screen, env)
    draw_hud(screen, env, ui)
    draw_outcome(screen, env, ui)


# ---------------------------------------------------------------
# EPISODE PLUMBING — thin wrappers around the environment API
# ---------------------------------------------------------------
def apply_step(env, ui, action):
    """Take one action through the standard step() contract and keep
    a running score for the HUD (the environment itself doesn't
    remember cumulative reward — that's the caller's bookkeeping)."""
    if env.done:
        return                      # finished episode: must reset first
    _, reward, done, info = env.step(action)
    ui["reward"] += reward
    if done:
        ui["outcome"] = info["reason"]   # "goal" / "pit" / "timeout"


def restart(env, ui):
    """Fresh episode: back to start, score zeroed, banner cleared."""
    env.reset()
    ui["reward"] = 0.0
    ui["outcome"] = None


# ---------------------------------------------------------------
# THE MAIN LOOP — events in, frames out
# ---------------------------------------------------------------
# Keys that acknowledge a warning banner before doing their job
# (so the message is read once, then gets out of the way).
DISMISS_KEYS = (
    pygame.K_SPACE, pygame.K_r, pygame.K_TAB,
    pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
    pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
    pygame.K_MINUS, pygame.K_KP_MINUS,
)


def run():
    """Open the window and play until the user quits."""
    pygame.init()
    _FONTS.clear()   # a previous pygame.quit() poisoned the cache
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Start on whichever maze is ACTIVE: classic on the first ever
    # run, your saved random maze after that (see environment.py).
    name, _rows = load_active_maze()
    env = GridWorld(name)
    pygame.display.set_caption(f"GridWorld Maze [{name}] - RL Fundamentals")

    ui = {"mode": "HUMAN", "reward": 0.0, "outcome": None,
          "speed_i": SPEED_DEFAULT_I, "notice": None, "notice_sub": None,
          "maze_name": name, "live": None}
    restart(env, ui)
    trained = None   # loaded lazily when TRAINED mode is entered

    # LIVE mode's state (None / unused outside that mode).
    live = None            # the Trainer driving the dashboard
    live_budget = 0.0      # saved-up fraction of an episode
    live_saved_stage = 0   # last stage that was written to disk
    live_finished = False  # has the completed run been saved yet?
    demo = None            # the animated "current best play"
    demo_env = None        # the board the demo walks through
    demo_timer = 0.0       # ms until the demo takes its next step

    watch_timer = 0.0   # counts ms until the next auto-move/pause

    def start_live():
        """Begin a FRESH training session (episode 1) for LIVE mode.

        Called when SPACE enters the mode and when N swaps the maze
        mid-session. If nothing was ever trained, a warning is dumped
        on screen — training still starts immediately behind it.
        """
        nonlocal live, live_saved_stage, live_finished, live_budget
        nonlocal demo_env, demo, demo_timer
        live = Trainer(quiet=True)
        live_saved_stage = live.stage
        live_finished = False
        live_budget = 0.0
        demo_timer = 0.0
        demo_env = GridWorld(ui["maze_name"])
        demo = {"phase": "build"}
        ui["live"] = {"episode": 0, "total": live.episodes,
                      "stage": live.stage,
                      "epsilon": live.agent.epsilon, "win": 0.0}
        if load_trained_agent() is None:
            ui["notice"] = "WARNING: no trained agent!"
            ui["notice_sub"] = "training was not launched - starting now"

    def save_live_progress():
        """Write the session to disk at stage boundaries and at the end.

        Not every frame (that would be pointless disk churn) and not
        on a plain exit (an unfinished run shouldn't replace a good
        brain) — just at the moments worth keeping.
        """
        nonlocal live_saved_stage, live_finished
        if live is None or live.episode == 0:
            return
        if live.done and not live_finished:
            live.finish()
            live_finished = True
        elif live.stage != live_saved_stage:
            live.finish()
            live_saved_stage = live.stage

    running = True
    while running:
        # tick() both caps the frame rate AND tells us how many
        # milliseconds passed since the last frame (for watch timing).
        elapsed = clock.tick(FPS)

        # ---- 1. EVENTS (keyboard / closing the window) ---------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Read the warning first — then this key does its job.
                if ui["notice"] and event.key in DISMISS_KEYS:
                    ui["notice"] = None
                    ui["notice_sub"] = None
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    if ui["mode"] == "LIVE" and demo is not None:
                        demo["phase"] = "build"   # replay the demo
                        ui["reward"] = 0.0
                        ui["outcome"] = None
                        demo_timer = 0.0
                    else:
                        restart(env, ui)
                    watch_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    # Leaving LIVE? The session already saved itself at
                    # its stage boundaries — just tear it down.
                    if ui["mode"] == "LIVE":
                        live = None
                        demo = None
                        demo_env = None
                        ui["live"] = None
                    ui["mode"] = next_mode(ui["mode"])
                    watch_timer = 0.0
                    if ui["mode"] == "TRAINED":
                        # (Re)load so a fresh `python train.py` run is
                        # picked up — and train right here if there is
                        # nothing to load yet (the safety net).
                        trained = ensure_trained(ui)
                    elif ui["mode"] == "LIVE":
                        start_live()
                elif event.key == pygame.K_n:
                    # A NEW random maze replaces the current one and
                    # becomes the permanent maze from now on...
                    env = new_random_maze()
                    ui["maze_name"] = env.maze_name
                    restart(env, ui)
                    watch_timer = 0.0
                    pygame.display.set_caption(
                        f"GridWorld Maze [{ui['maze_name']}] - RL "
                        f"Fundamentals")
                    # ...which orphans the one agent: retrain at once
                    # so the brain always matches the maze you play.
                    fresh = retrain()
                    if ui["mode"] == "TRAINED":
                        trained = fresh
                    elif ui["mode"] == "LIVE":
                        start_live()
                    ui["notice"] = "NEW MAZE GENERATED!"
                    ui["notice_sub"] = "retraining the agent for it..."
                elif (event.key == pygame.K_TAB and ui["mode"] == "LIVE"
                      and live is not None):
                    # Fast-forward: the rest of this stage runs at
                    # once, then the demo shows a smarter agent.
                    live.skip_stage()
                    save_live_progress()
                    if demo is not None:
                        demo["phase"] = "build"
                        demo_timer = 0.0
                        ui["reward"] = 0.0
                        ui["outcome"] = None
                elif event.key == pygame.K_v:
                    # Charts of the CURRENT agent — training is
                    # launched first if it was forgotten.
                    ensure_trained(ui)
                    if not launch_visualizer():
                        ui["notice"] = "Could not open the visualizer"
                        ui["notice_sub"] = "run: python visualize.py"
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                   pygame.K_KP_PLUS):
                    # +: one step up the speed ladder (clamped at max)
                    ui["speed_i"] = min(
                        ui["speed_i"] + 1, len(SPEED_LEVELS_MS) - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    # -: one step down (clamped at min)
                    ui["speed_i"] = max(ui["speed_i"] - 1, 0)
                elif ui["mode"] == "HUMAN" and event.key in KEY_TO_ACTION:
                    # Your keypress IS the policy in human mode.
                    apply_step(env, ui, KEY_TO_ACTION[event.key])

        # ---- 2a. LIVE: train behind the scenes, demo on stage -------
        if ui["mode"] == "LIVE" and live is not None:
            # Training is NEVER paused — not even by the warning
            # banner: "launching automatically" means it happens now.
            live_budget += (elapsed
                            * LIVE_EPS_PER_SEC[ui["speed_i"]] / 1000.0)
            n_eps = int(live_budget)
            if n_eps:
                live_budget -= n_eps
                live.run_episodes(n_eps)
            save_live_progress()
            ui["live"] = {"episode": live.episode, "total": live.episodes,
                          "stage": live.stage,
                          "epsilon": live.agent.epsilon,
                          "win": live.win_rate()}

            # The board shows the agent's CURRENT BEST play — one
            # step every DEMO_STEP_MS, rebuilt (smarter) after each run.
            if demo is not None:
                demo_timer += elapsed
                if demo["phase"] == "build":
                    path, cum, reason = build_demo(live.agent,
                                                   ui["maze_name"])
                    demo.update(phase="play", path=path, cum=cum,
                                reason=reason, i=0)
                    demo_env.agent_pos = path[0]
                    demo_env.steps = 0
                    ui["reward"] = 0.0
                    ui["outcome"] = None
                    demo_timer = 0.0
                elif demo["phase"] == "play" and demo_timer >= DEMO_STEP_MS:
                    demo_timer = 0.0
                    demo["i"] += 1
                    i = demo["i"]
                    demo_env.agent_pos = demo["path"][i]
                    demo_env.steps = i
                    ui["reward"] = demo["cum"][i - 1]
                    if i >= len(demo["path"]) - 1:
                        ui["outcome"] = demo["reason"]
                        demo["phase"] = "hold"
                elif demo["phase"] == "hold" and demo_timer >= WATCH_PAUSE_MS:
                    demo["phase"] = "build"

        # ---- 2b. TIMER MODES act on a timer, not on events ----------
        # WATCH and TRAINED share the clock, speed ladder, and
        # auto-restart; they differ only in WHO picks the moves.
        # A waiting warning banner pauses them (LIVE, above, never
        # pauses — its training IS the thing being launched).
        if ui["mode"] in ("WATCH", "TRAINED") and not ui["notice"]:
            watch_timer += elapsed
            step_ms = SPEED_LEVELS_MS[ui["speed_i"]]   # current speed
            if not env.done and watch_timer >= step_ms:
                watch_timer = 0.0
                if ui["mode"] == "WATCH":
                    # A random policy: no thinking, just luck.
                    action = random.randrange(N_ACTIONS)
                else:
                    # The graduate: pure exploitation (explore=False).
                    action = trained.choose_action(env.agent_pos,
                                                   explore=False)
                apply_step(env, ui, action)
            elif env.done and watch_timer >= WATCH_PAUSE_MS:
                watch_timer = 0.0
                restart(env, ui)   # auto-start the next attempt

        # ---- 3. PAINT the frame --------------------------------------
        # In LIVE mode the painted board is the demo's, not the
        # (messy, in-progress) training environment's.
        screen_env = demo_env if (ui["mode"] == "LIVE"
                                  and demo_env is not None) else env
        draw(screen, screen_env, ui)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
