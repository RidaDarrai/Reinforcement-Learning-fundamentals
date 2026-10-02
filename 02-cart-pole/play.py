"""
play.py — THE WINDOW (part C of our RL project)
===============================================

Until now the world lived in the terminal. This file gives it a real
game window: a physics scene (track, cart, pole), a HUD, and four
ways to interact.

Run it (from this folder):

    python play.py

Controls
--------
    Left / Right ... push the cart        (HUMAN mode)
    + / - ........... speed up / slow down (WATCH / LIVE / TRAINED)
    R ............... restart the episode
    SPACE ........... cycle to the next mode
    TAB ............. skip to the next training stage  (LIVE mode)
    V ............... open the charts of the trained agent
    ESC or Q ........ quit

Modes
-----
    HUMAN   : you push the cart yourself — the best way to FEEL how
              hard balancing is before asking an agent to learn it.
              Every key press = one physics tick (0.02 s), so the
              world waits for you between presses.
    WATCH   : a mindless random agent flails by itself — the
              baseline to beat (~10-15 ticks).
    LIVE    : watch the agent LEARN. A fresh training run advances
              in the background while the scene replays its current
              best greedy episode; the HUD counts episodes, ε, and
              walks the stages EXPLORING -> LEARNING -> POLISHING.
              + / - changes the training rate, TAB fast-forwards.
    TRAINED : the GRADUATE plays greedily (no random moves, no
              learning — pure performance). Needs q_table.json;
              if it's missing, a warning appears and a QUICK
              retrain (~3 s) launches automatically — run
              `python train.py` for the full diploma session.

The window talks to the world only through the same calls
everything else uses (reset / step / state) — environment.py
remains the single source of truth; nothing here knows the physics.
"""

import math
import os
import random
import subprocess
import sys

import pygame

from agent import QAgent
from environment import (
    CartPole, LEFT, RIGHT, N_ACTIONS, MAX_STEPS, X_LIMIT, ANGLE_LIMIT,
)
from train import Trainer, STAGE_NAMES

# Everything lives next to this file, whatever folder you ran from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# LAYOUT — pure geometry: how big is the scene, the window?
# ---------------------------------------------------------------
WIDTH = 760           # window width
PAD = 16              # border around everything
GAP = 14              # space between scene and HUD
SCENE_H = 300         # the physics stage (track + pole room above)
HUD_H = 76            # height of the info bar
HEIGHT = PAD + SCENE_H + GAP + HUD_H + PAD      # 422

FPS = 60              # redraws per second (the window itself)

# The track: +/-2.4 m maps onto this stretch of pixels, with margins
# so the cart never hides under the window border.
TRACK_LEFT = PAD + 40
TRACK_RIGHT = WIDTH - PAD - 40
TRACK_Y = PAD + 240            # the ground line the wheels roll on
CENTER_PX = WIDTH // 2

CART_W, CART_H = 60, 28        # cart body
POLE_LEN = 170                 # pole length in pixels (visual only)
POLE_W = 7                     # pole thickness

# ---------------------------------------------------------------
# SPEED — the + / - keys walk up/down these ladders
# ---------------------------------------------------------------
# WATCH/TRAINED: milliseconds between physics ticks (slow -> fast).
# Default 40 ms = 25 ticks/s, roughly watchable live motion for a
# 0.02 s-dynamics world. Displayed as a multiplier of the default.
SPEED_LEVELS_MS = [120, 70, 40, 24, 14, 8]
SPEED_DEFAULT_I = 2                        # the 40 ms entry
SPEED_BASE_MS = SPEED_LEVELS_MS[SPEED_DEFAULT_I]   # "1.0x" reference

# LIVE: how many training episodes fly by per second.
LIVE_EPS_PER_SEC = [4, 10, 30, 80, 200, 500]

# LIVE's "current best play" replay: greedy episodes can run the
# full 500 ticks, so every other state is shown, 40 ms each —
# worst case 250 * 40ms = 10 s of ballet before the next rebuild.
DEMO_STEP_MS = 40
DEMO_SUBSAMPLE = 2
DEMO_HOLD_MS = 1400        # admire the episode before rebuilding

WATCH_PAUSE_MS = 1400      # admire a finished episode before restart

# The safety net trains this many episodes when q_table.json is
# missing (~3 s). The FULL session — 15,000 episodes with
# graduation exams — is what `python train.py` runs.
RETRAIN_EPISODES = 2000

# ---------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------
C_BG       = (28, 30, 36)      # window background
C_SCENE    = (22, 24, 28)      # ground band below the track
C_TRACK    = (210, 210, 200)   # the track line
C_CENTER   = (90, 96, 110)     # dotted x=0 marker
C_WALL     = (158, 47, 47)     # the +/-2.4 m end stops
C_GUIDE    = (120, 70, 70)     # faint +/-12 deg guide lines
C_CART     = (47, 108, 212)    # the cart (blue)
C_CART_RIM = (235, 241, 255)   # cart outline
C_POLE     = (240, 240, 240)   # the pole, safely upright
C_POLE_HOT = (255, 130, 90)    # the pole, near the 12 deg cliff
C_ARROW    = (255, 210, 90)    # last push direction
C_HUD_BG   = (40, 44, 54)      # HUD background
C_TEXT     = (235, 235, 235)   # HUD text
C_TEXT_DIM = (170, 175, 185)   # secondary HUD text
C_MODE_HUMAN = (90, 160, 255)  # "HUMAN" badge color
C_MODE_WATCH = (255, 170, 60)  # "WATCH" badge color
C_MODE_LIVE = (190, 140, 255)  # "LIVE" badge color (training = purple)
C_MODE_TRAINED = (110, 225, 150)  # "TRAINED" badge color
C_WIN  = (80, 220, 120)        # survived the full 500 (the good end)
C_LOSE = (255, 95, 85)         # fell / cart off track

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
    """Quick safety-net training (~3 s, RETRAIN_EPISODES).

    Whenever something needs a trained agent and none exists, we
    don't ask the user to open a terminal — we just do it. The
    quick course is enough to demo TRAINED mode; the full 15,000-
    episode diploma session is one `python train.py` away.
    """
    trainer = Trainer(episodes=RETRAIN_EPISODES, quiet=True)
    trainer.run_all()
    trainer.finish()
    return trainer.agent


def ensure_trained(ui):
    """Return a trained agent, warning first if training was missing.

    If q_table.json exists we're done (normal case). If not, dump a
    warning banner, launch the quick training right here, and
    return the result — the banner stays until a key acknowledges
    it, so the message can't be missed.
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


def build_demo(agent):
    """Run one greedy episode; return (states, reason).

    states = [(tick, x, theta, action), ...], already subsampled so
    LIVE mode's replay lasts at most ~10 s even when the agent
    survives all 500 ticks. The last state is always kept — you
    want to SEE how the episode ends.
    """
    env = CartPole()
    state = env.reset()
    states = [(0, state[0], state[2], None)]
    while True:
        action = agent.choose_action(state, explore=False)
        state, _reward, done, info = env.step(action)
        states.append((info["steps"], state[0], state[2], action))
        if done:
            break
    sampled = states[::DEMO_SUBSAMPLE]
    if sampled[-1] != states[-1]:
        sampled.append(states[-1])
    return sampled, info["reason"]


# Map each of our integer actions to a keyboard key.
KEY_TO_ACTION = {
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


def x_to_px(x):
    """Map world meters (-2.4 .. +2.4) onto screen pixels."""
    return CENTER_PX + (x / X_LIMIT) * ((TRACK_RIGHT - TRACK_LEFT) // 2)


# ---------------------------------------------------------------
# DRAWING — one frame of the game
# ---------------------------------------------------------------
def draw_scene(screen, env, ui):
    """Paint the physics stage: track, limits, cart, pole."""
    scene = pygame.Rect(0, 0, WIDTH, PAD + SCENE_H)
    pygame.draw.rect(screen, C_SCENE,
                     pygame.Rect(0, TRACK_Y, WIDTH, scene.bottom - TRACK_Y))

    # The +/-2.4 m end stops (the cart_out cliff, drawn as walls).
    for x_px in (TRACK_LEFT, TRACK_RIGHT):
        pygame.draw.rect(screen, C_WALL,
                         pygame.Rect(x_px - 4, TRACK_Y - 46, 8, 50),
                         border_radius=3)

    # The track itself.
    pygame.draw.line(screen, C_TRACK,
                     (TRACK_LEFT, TRACK_Y), (TRACK_RIGHT, TRACK_Y), 4)

    # Dotted center line: "where a balanced cart wants to be".
    for y in range(PAD + 20, TRACK_Y - 40, 16):
        pygame.draw.line(screen, C_CENTER, (CENTER_PX, y),
                         (CENTER_PX, y + 7), 2)

    # Faint +/-12 deg guides from the pivot: the cliff the pole may
    # never cross (the same ANGLE_LIMIT environment.py enforces).
    pivot_guess = (CENTER_PX, TRACK_Y - 6 - CART_H)
    for sign in (-1, 1):
        tip = (pivot_guess[0] + sign * POLE_LEN * math.sin(ANGLE_LIMIT),
               pivot_guess[1] - POLE_LEN * math.cos(ANGLE_LIMIT))
        pygame.draw.line(screen, C_GUIDE, pivot_guess, tip, 2)

    # The cart: body on little wheels, wherever the world says it is.
    cx = x_to_px(env.x)
    body = pygame.Rect(0, 0, CART_W, CART_H)
    body.center = (cx, TRACK_Y - 6 - CART_H // 2)
    pygame.draw.circle(screen, C_CART_RIM, (cx - 18, TRACK_Y), 7)
    pygame.draw.circle(screen, C_CART_RIM, (cx + 18, TRACK_Y), 7)
    pygame.draw.circle(screen, (60, 66, 80), (cx - 18, TRACK_Y), 4)
    pygame.draw.circle(screen, (60, 66, 80), (cx + 18, TRACK_Y), 4)
    pygame.draw.rect(screen, C_CART, body, border_radius=6)
    pygame.draw.rect(screen, C_CART_RIM, body, 2, border_radius=6)

    # Which way did we just push? The lesson of cart-pole in one
    # arrow: pushing the cart one way tips the pole the OTHER way.
    if ui.get("last_action") is not None:
        ay = body.centery
        if ui["last_action"] == RIGHT:
            pts = [(body.right + 4, ay), (body.right + 16, ay - 8),
                   (body.right + 16, ay + 8)]
        else:
            pts = [(body.left - 4, ay), (body.left - 16, ay - 8),
                   (body.left - 16, ay + 8)]
        pygame.draw.polygon(screen, C_ARROW, pts)

    # The pole: from the cart's pivot, tilted by theta from vertical.
    pivot = (cx, body.top)
    tip = (pivot[0] + POLE_LEN * math.sin(env.theta),
           pivot[1] - POLE_LEN * math.cos(env.theta))
    danger = abs(env.theta) >= ANGLE_LIMIT * 0.75   # 75% of the cliff
    pygame.draw.line(screen, C_POLE_HOT if danger else C_POLE,
                     pivot, tip, POLE_W)
    pygame.draw.circle(screen, (120, 126, 140), pivot, 6)   # the joint


def draw_hud(screen, env, ui):
    """The info bar: mode, progress, score, and the control hints."""
    hud_rect = pygame.Rect(PAD, PAD + SCENE_H + GAP, WIDTH - 2 * PAD,
                           HUD_H)
    pygame.draw.rect(screen, C_HUD_BG, hud_rect, border_radius=10)

    # --- line 1: mode badge + the numbers that change per mode ------
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
        # Training dashboard: episodes, stage, exploration rate,
        # rolling win rate, and the BEST EXAM so far (the diploma
        # that finish() would save right now).
        live = ui["live"]
        info = font(20).render(
            f"EP {live['episode']}/{live['total']} "
            f"{live['stage'] + 1}/3 \u03b5{live['epsilon']:.2f} "
            f"W{live['win']:.0f}% B{live['best']:.0f}%",
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
        # reward == steps ALWAYS in this game — the HUD shows both so
        # you can watch the two numbers agree tick after tick.
        info = font(22).render(
            f"  STEP {env.steps}/{MAX_STEPS}   REWARD {ui['reward']:+.1f}",
            True, C_TEXT,
        )
        screen.blit(info, (info_x, hud_rect.y + 10))

        if mode in ("WATCH", "TRAINED"):
            speed = SPEED_BASE_MS / SPEED_LEVELS_MS[ui["speed_i"]]
            speed_img = font(17).render(f" {speed:.1f}x", True,
                                        MODE_COLORS[mode])
            screen.blit(speed_img,
                        (info_x + info.get_width() + 4, hud_rect.y + 14))

    # --- line 2: control hints (they change with the mode) -----------
    if mode == "LIVE":
        stage_name = STAGE_NAMES[ui["live"]["stage"]] if ui.get("live") \
            else STAGE_NAMES[0]
        hint = (f"{stage_name}  [TAB] stage  [V] charts  "
                f"[SPACE] mode  [R] replay")
    elif mode == "HUMAN":
        hint = ("[<- ->] push  [SPACE] modes  [V] charts  [R] restart")
    else:
        hint = ("[+/-] speed  [SPACE] modes  [V] charts  [R] restart")
    screen.blit(font(14).render(hint, True, C_TEXT_DIM),
                (hud_rect.x + 12, hud_rect.y + 48))


def draw_outcome(screen, env, ui):
    """Big banner across the middle: episode result, or a notice."""
    scene_mid = PAD + SCENE_H // 2

    if ui["outcome"] is None:
        notice = ui.get("notice")
        if not notice:
            return
        band = pygame.Surface((WIDTH, 100), pygame.SRCALPHA)
        band.fill((10, 12, 16, 190))
        screen.blit(band, (0, scene_mid - 50))
        big = font(24).render(notice, True, (255, 210, 90))
        if ui.get("notice_sub"):
            big_y = scene_mid - 22
            sub = font(16).render(ui["notice_sub"], True, C_TEXT_DIM)
            screen.blit(sub, sub.get_rect(center=(WIDTH // 2,
                                                   scene_mid + 14)))
        else:
            big_y = scene_mid - 12
        screen.blit(big, big.get_rect(center=(WIDTH // 2, big_y)))
        return

    banner = {
        "pole_fell":  ("POLE FELL!", C_LOSE),
        "cart_out":   ("CART OFF THE TRACK!", C_LOSE),
        "time_limit": (f"SURVIVED ALL {MAX_STEPS}!", C_WIN),
    }
    text, color = banner[ui["outcome"]]

    band = pygame.Surface((WIDTH, 120), pygame.SRCALPHA)
    band.fill((10, 12, 16, 190))
    screen.blit(band, (0, scene_mid - 60))

    big = font(38).render(text, True, color)
    screen.blit(big, big.get_rect(center=(WIDTH // 2, scene_mid - 14)))

    sub_text = ("press R to restart" if ui["mode"] == "HUMAN"
                else "restarting...")
    sub = font(20).render(sub_text, True, C_TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, scene_mid + 24)))


def draw(screen, env, ui):
    """Render one complete frame (scene -> HUD -> banner)."""
    screen.fill(C_BG)
    draw_scene(screen, env, ui)
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
    ui["last_action"] = action      # so the scene can draw the push
    if done:
        ui["outcome"] = info["reason"]   # pole_fell / cart_out / time_limit


def restart(env, ui):
    """Fresh episode: new noise, score zeroed, banner cleared."""
    env.reset()
    ui["reward"] = 0.0
    ui["outcome"] = None
    ui["last_action"] = None


# ---------------------------------------------------------------
# THE MAIN LOOP — events in, frames out
# ---------------------------------------------------------------
# Keys that acknowledge a warning banner before doing their job
# (so the message is read once, then gets out of the way).
DISMISS_KEYS = (
    pygame.K_SPACE, pygame.K_r, pygame.K_TAB,
    pygame.K_LEFT, pygame.K_RIGHT,
    pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
    pygame.K_MINUS, pygame.K_KP_MINUS,
)


def run():
    """Open the window and play until the user quits."""
    pygame.init()
    _FONTS.clear()   # a previous pygame.quit() poisoned the cache
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = CartPole()
    pygame.display.set_caption("CartPole - RL Fundamentals")

    ui = {"mode": "HUMAN", "reward": 0.0, "outcome": None,
          "speed_i": SPEED_DEFAULT_I, "notice": None, "notice_sub": None,
          "live": None, "last_action": None}
    restart(env, ui)
    trained = None   # loaded lazily when TRAINED mode is entered

    # LIVE mode's state (None / unused outside that mode).
    live = None            # the Trainer driving the dashboard
    live_budget = 0.0      # saved-up fraction of an episode
    live_saved_stage = 0   # last stage that was written to disk
    live_finished = False  # has the completed run been saved yet?
    demo = None            # the animated "current best play"
    demo_env = None        # the CartPole the demo pokes state into
    demo_timer = 0.0       # ms until the demo takes its next step

    watch_timer = 0.0   # counts ms until the next auto-move/pause

    def start_live():
        """Begin a FRESH training session (episode 1) for LIVE mode.

        Called when SPACE enters the mode. If nothing was ever
        trained, a warning is dumped on screen — training still
        starts immediately behind it.
        """
        nonlocal live, live_saved_stage, live_finished, live_budget
        nonlocal demo_env, demo, demo_timer
        live = Trainer(quiet=True)
        live_saved_stage = live.stage
        live_finished = False
        live_budget = 0.0
        demo_timer = 0.0
        demo_env = CartPole()
        demo = {"phase": "build"}
        ui["live"] = {"episode": 0, "total": live.episodes,
                      "stage": live.stage,
                      "epsilon": live.agent.epsilon, "win": 0.0,
                      "best": 0.0}
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

    def apply_demo_state(index):
        """Point the demo's world at state #index of the replay."""
        tick, x, theta, action = demo["states"][index]
        demo_env.x, demo_env.theta = x, theta
        demo_env.steps = tick
        ui["reward"] = float(tick)        # reward == steps, always
        ui["last_action"] = action

    running = True
    while running:
        # tick() both caps the frame rate AND tells us how many
        # milliseconds passed since the last frame (for timers).
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
                elif event.key == pygame.K_TAB and ui["mode"] == "LIVE" \
                        and live is not None:
                    # Fast-forward: the rest of this stage runs at
                    # once, then the replay shows a smarter agent.
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

        # ---- 2a. LIVE: train behind the scenes, replay on stage -----
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
                          "win": live.win_rate(),
                          "best": live.best_score[0] if live.best_q
                          else 0.0}

            # The scene replays the agent's CURRENT BEST play — one
            # state every DEMO_STEP_MS, rebuilt (smarter) after each.
            if demo is not None:
                demo_timer += elapsed
                if demo["phase"] == "build":
                    states, reason = build_demo(live.agent)
                    demo.update(phase="play", states=states,
                                reason=reason, i=0)
                    ui["outcome"] = None
                    demo_timer = 0.0
                    apply_demo_state(0)
                elif demo["phase"] == "play" and demo_timer >= DEMO_STEP_MS:
                    demo_timer = 0.0
                    demo["i"] += 1
                    apply_demo_state(demo["i"])
                    if demo["i"] >= len(demo["states"]) - 1:
                        ui["outcome"] = demo["reason"]
                        demo["phase"] = "hold"
                elif demo["phase"] == "hold" and demo_timer >= DEMO_HOLD_MS:
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
                    action = trained.choose_action(env.state,
                                                   explore=False)
                apply_step(env, ui, action)
            elif env.done and watch_timer >= WATCH_PAUSE_MS:
                watch_timer = 0.0
                restart(env, ui)   # auto-start the next attempt

        # ---- 3. PAINT the frame --------------------------------------
        # In LIVE mode the painted world is the demo's replay, not
        # the (messy, in-progress) training environment's.
        screen_env = demo_env if (ui["mode"] == "LIVE"
                                  and demo_env is not None) else env
        draw(screen, screen_env, ui)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
