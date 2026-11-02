"""
play.py — THE WINDOW (step 4 of the lunar lander project)
=========================================================

Until now the lander lived in the terminal. This file gives it the
full cockpit: the world on the right, live telemetry plots on the
left, a controller keypad in the top-right corner, and a HUD along
the bottom. Two modes only — you, or the graduate:

    HUMAN   : you fly it. Hold the keys; the physics runs in real
              time (50 ticks/s), and doing nothing IS a choice —
              IDLE free-falls. This is the fastest way to learn why
              "just point up and burn" does not land anything.
    TRAINED : the DQN graduate flies greedily (no exploration, no
              learning — pure policy). Auto-reloads dqn_model.pt
              every time you enter the mode, so a fresh
              `python train.py` run is picked up live. If no
              diploma exists yet, a warning banner appears and a
              QUICK course (~300 episodes) trains right here.

    (No RANDOM flailing, no LIVE training-in-window: 02/03 taught
    both, and this window's money shots are HUMAN vs TRAINED —
    a lander episode is 20 s long, a random one is 20 s of noise.)

Run it (from this folder):

    python play.py

Controls
--------
    UP .............. main engine burn       (HUMAN)
    LEFT / RIGHT .... rotate the lander      (HUMAN)
    + / - ........... speed up / slow down   (TRAINED)
    R ............... retry this attempt
    SPACE ........... switch HUMAN <-> TRAINED
    V ............... open the charts        (launches visualize.py)
    ESC or Q ........ quit

Layout (one window, four instruments)
-------------------------------------
    LEFT   : six scrolling plots — x, y, vx, vy, tilt, leg contact —
             each showing the last ~10 s of flight (500 ticks).
             Watching vx/vy fight your burns is the whole skill.
    RIGHT  : the Moon: world 100 x 80 m, ground, the 30 m pad with
             two flags, the OOB walls and ceiling, and the lander
             drawn from geometry() — the SAME points the physics
             rules read, so the picture can never disagree with
             the math.
    TOP-R  : the keypad — MAIN / LEFT / RIGHT lamps lit for the
             last action taken, IDLE lamp otherwise. You SEE which
             button the brain (or you) pressed, every tick.
    BOTTOM : the HUD — mode, attempt number, return (live money),
             fuel burned, and the death reason of the last attempt.

The window talks to the world ONLY through the standard calls
(reset / step / state) — environment.py stays the single source of
truth. Drawing maps world meters to pixels ONCE, at the draw
boundary: x right, y flipped down (the one flip; the world keeps
math space).
"""

# Every import earns its comment (repo rule: every file explains itself).
import math        # degrees for the tilt plot, flame/fleg geometry
import os          # paths anchored next to THIS file (not the cwd)
import subprocess  # launch visualize.py in its own process (see V)
import sys         # sys.executable: the same Python runs the visualizer

import pygame  # the window (dummy video driver in tests)

# The world, the brain, and the school — all in THIS folder. train.py
# is imported for the quick-course fallback (TRAINED with no diploma),
# not for a training loop here.
from agent import DQNAgent
from environment import (
    LunarLander, IDLE, MAIN, ROT_LEFT, ROT_RIGHT,
    MAX_STEPS, WORLD_HALF_WIDTH, CEILING_Y, PAD_HALF,
)
from train import Trainer

# Everything lives next to this file, whatever folder you launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# LAYOUT — pure geometry: where every instrument sits, in pixels
# ---------------------------------------------------------------
WIDTH  = 900            # window width
PAD    = 16             # border around everything
GAP    = 14             # space between the panels and the HUD
HUD_H  = 96             # the bottom info bar

# LEFT: the telemetry panel (six stacked scrolling plots).
PLOT_X    = PAD          # 16
PLOT_W    = 300
PLOT_TOP  = PAD + 8
PLOT_PANE_H = 68         # each mini-plot's height
PLOT_GAP  = 6            # vertical gap between mini-plots
PLOT_TICKS = 500         # history depth: 500 ticks * 0.02 s = 10 s

# RIGHT: the flight scene (world -> pixels, ONE flip at this line).
SCENE_X0 = PAD + PLOT_W + GAP          # 330
SCENE_Y0 = PAD                         # 16
SCENE_H  = 460                         # scene band height
SCENE_W  = WIDTH - SCENE_X0 - PAD      # 554

PPX      = 5.0                         # pixels per meter
X_L_PX   = SCENE_X0 + 40               # world x=-50 m lands here
X_R_PX   = X_L_PX + 500                # world x=+50 m lands here
CEIL_PX  = SCENE_Y0 + 40               # world y=80 m (the ceiling) here
GROUND_Y = SCENE_Y0 + 440              # world y=0 m (the ground) here

HEIGHT = PAD + SCENE_H + GAP + HUD_H + PAD   # 602

FPS = 60               # window redraws per second
TICK_MS = 20           # 50 physics ticks/s == environment.py's DT

# HUMAN mode is always real-time; + / - in TRAINED multiplies it.
SPEED_MULTS   = (0.5, 1.0, 2.0, 4.0)
SPEED_DEFAULT_I = 1

WATCH_PAUSE_MS = 1600  # TRAINED: admire the wreck/landing, then again

# The quick course: TRAINED mode's fallback when no diploma exists.
# Writes its OWN files (quick_course_*), never the official run's
# dqn_model.pt/training_log.json — a middle-of-training fallback must
# not stomp the artifacts `python train.py` is still writing.
RETRAIN_EPISODES = 300
QUICK_DIPLOMA    = os.path.join(BASE_DIR, "quick_course.pt")
QUICK_LOG        = os.path.join(BASE_DIR, "quick_course_log.json")

DIPLOMA_PATH = os.path.join(BASE_DIR, "dqn_model.pt")   # the real one

# ---------------------------------------------------------------
# COLORS
# ---------------------------------------------------------------
C_BG       = (24, 26, 32)      # window background
C_SKY      = (16, 18, 24)      # scene backdrop (no air on the Moon)
C_GROUND   = (96, 92, 88)      # the regolith top edge
C_GROUND_F = (44, 42, 40)      # below the ground line
C_PAD      = (200, 205, 215)   # the 30 m pad strip
C_FLAG     = (240, 90, 80)     # the two pad flags
C_WALL     = (150, 52, 52)     # OOB side walls + ceiling dashes
C_LANDER   = (190, 200, 215)   # hull
C_LANDER_R = (240, 246, 255)   # hull outline
C_LEG      = (150, 160, 175)   # legs
C_TOUCH    = (120, 235, 150)   # a leg tip currently in the dirt
C_FLAME    = (255, 150, 60)    # main engine plume
C_PUFF     = (200, 220, 255)   # side-thruster puffs
C_PANEL    = (36, 40, 50)      # plot panels / keypad box
C_PANEL_R  = (70, 76, 92)      # their border
C_HUD_BG   = (40, 44, 54)
C_TEXT     = (235, 235, 235)
C_TEXT_DIM = (170, 175, 185)
C_MODE_HUMAN   = (90, 160, 255)
C_MODE_TRAINED = (110, 225, 150)
C_WIN  = (80, 220, 120)
C_LOSE = (255, 95, 85)

MODES = ("HUMAN", "TRAINED")
MODE_COLORS = {"HUMAN": C_MODE_HUMAN, "TRAINED": C_MODE_TRAINED}


def next_mode(current):
    """The mode SPACE switches to: HUMAN <-> TRAINED (two-way)."""
    return MODES[(MODES.index(current) + 1) % len(MODES)]


# The scrolling instruments. (key, label, value range for the axis,
# line color). legs is a special panel: two 0/1 traces.
PLOT_PANELS = (
    ("x",    "x m",     -WORLD_HALF_WIDTH, WORLD_HALF_WIDTH, (110, 220, 190)),
    ("y",    "y m",      0.0,              CEILING_Y,        (130, 180, 250)),
    ("vx",   "vx m/s",  -6.0,       6.0,  (240, 200, 110)),
    ("vy",   "vy m/s",  -8.0,       8.0,  (240, 140, 110)),
    ("tilt", "tilt deg", -90.0,    90.0,  (190, 140, 250)),
    ("legs", "legs L/R", None,     None,  (120, 230, 140)),
)
PLOT_KEYS = tuple(p[0] for p in PLOT_PANELS)


# ---------------------------------------------------------------
# INPUT — HUMAN mode: held keys are the policy
# ---------------------------------------------------------------
# Checked every frame, in priority order: thrust wins if you hold
# two at once (the up-key is the one you usually need most).
HELD_TO_ACTION = (
    (pygame.K_UP,    MAIN),
    (pygame.K_LEFT,  ROT_LEFT),
    (pygame.K_RIGHT, ROT_RIGHT),
)


def _key_is_down(pressed, key):
    """Is this key currently held? Tolerant lookup: pygame's
    get_pressed() sequence and plain test dicts both work."""
    try:
        return bool(pressed[key])
    except (KeyError, IndexError):
        return False


def action_from_held(pressed):
    """Held-key state -> ONE action (IDLE if nothing is held).

    Args: pressed — pygame.key.get_pressed() (or any mapping keyed
    by pygame.K_* constants with truthy values; dicts keep this
    testable without a window).
    """
    for key, action in HELD_TO_ACTION:
        if _key_is_down(pressed, key):
            return action
    return IDLE


# ---------------------------------------------------------------
# TRAINED MODE'S BRAIN SUPPLY — load, or train the quick course
# ---------------------------------------------------------------
def load_trained_agent(path=None):
    """Load the graduate from disk, or None if there isn't one yet.

    Returning None (instead of raising) lets the window degrade
    gracefully: no dqn_model.pt yet -> friendly banner + quick
    course. Corrupt/half-written checkpoints (the file can be
    mid-write when training is still running) also yield None.
    """
    try:
        return DQNAgent.load(DIPLOMA_PATH if path is None else path)
    except Exception:
        return None        # missing / half-written / wrong shape


def retrain(episodes=RETRAIN_EPISODES, diploma_path=None):
    """The quick course: a small Trainer run, done right here.

    Shrinks every dial proportionally (decay over the WHOLE course,
    exams halfway), writes its diploma/log to the quick_course_*
    files by default, and returns the trained agent — so pressing
    TRAINED with no `python train.py` behind you still costs ~a
    minute, not a crash.
    """
    if diploma_path is None:
        diploma_path = QUICK_DIPLOMA
    trainer = Trainer(
        episodes=episodes,
        epsilon_decay_episodes=episodes,        # decay the whole course
        exam_every=max(2, episodes // 2),
        exam_episodes=4,
        seed=7,                                  # official run uses 42
        log_path=QUICK_LOG,
        diploma_path=diploma_path,
    )
    trainer.run()
    return trainer.agent


def ensure_trained(ui):
    """Return a trained agent, warning first if training was missing."""
    agent = load_trained_agent()
    if agent is not None:
        return agent
    ui["notice"] = "WARNING: no trained brain!"
    ui["notice_sub"] = "quick course (~1 min) launching now"
    return retrain()


def launch_visualizer(runner=None):
    """Open the charts (visualize.py) in its own process.

    A separate process keeps pygame and matplotlib from fighting
    over the windowing system (on Windows they really do).
    """
    if not os.path.exists(os.path.join(BASE_DIR, "visualize.py")):
        return False
    if runner is None:
        runner = subprocess.Popen
    try:
        runner([sys.executable, "visualize.py"], cwd=BASE_DIR)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------
# EPISODE PLUMBING — thin wrappers around the environment API
# ---------------------------------------------------------------
def fresh_history():
    """Empty telemetry: one list per plotted channel."""
    return {key: [] for key in PLOT_KEYS}


def fresh_ui():
    """The mutable window state (everything draw() reads)."""
    return {"mode": "HUMAN", "return": 0.0, "outcome": None,
            "crash_type": None, "attempt": 0, "last_action": None,
            "speed_i": SPEED_DEFAULT_I, "notice": None,
            "notice_sub": None, "history": fresh_history()}


def restart(env, ui):
    """Fresh attempt: new spawn, score zeroed, telemetry cleared."""
    env.reset()
    ui["return"] = 0.0
    ui["outcome"] = None
    ui["crash_type"] = None
    ui["last_action"] = None
    ui["attempt"] += 1
    ui["history"] = fresh_history()


def record_tick(env, ui):
    """Append this tick's numbers to every scrolling channel."""
    h = ui["history"]
    h["x"].append(env.x)
    h["y"].append(env.y)
    h["vx"].append(env.vx)
    h["vy"].append(env.vy)
    h["tilt"].append(math.degrees(env.theta))
    h["legs"].append((env.leg_l, env.leg_r))


def apply_step(env, ui, action):
    """One tick through the standard step() contract + bookkeeping.

    Updates the running return, the last action (the keypad lamps
    read it), the telemetry, and — if the world says done — the
    death reason the HUD banner will show.
    """
    if env.done:
        return                      # finished episode: must reset first
    _, reward, done, info = env.step(action)
    ui["return"] += reward
    ui["last_action"] = action
    record_tick(env, ui)
    if done:
        ui["outcome"] = info["reason"]
        ui["crash_type"] = info.get("crash_type")


def banner_for(outcome, crash_type):
    """(text, color) for the big center banner, per death reason."""
    if outcome == "landed":
        return "TOUCHDOWN!", C_WIN
    if outcome == "out_of_bounds":
        return "OUT OF BOUNDS!", C_LOSE
    if outcome == "timeout":
        return "OUT OF TIME - NO SCORE", C_TEXT_DIM
    kind = {"hull": "BELLY FLOP", "impact": "HARD IMPACT",
            "missed_pad": "MISSED THE PAD"}.get(crash_type, "CRASHED")
    return kind + "!", C_LOSE


def hud_lines(env, ui):
    """The HUD's text lines (kept pure so tests can read them)."""
    n = max(1, len(ui["history"]["x"]))
    last = ui["outcome"] or "flying"
    if ui["outcome"] == "crashed" and ui["crash_type"]:
        last = f"crashed ({ui['crash_type']})"
    return [
        f"ATTEMPT {ui['attempt']}   TICK {env.steps}/{MAX_STEPS}   "
        f"SPEED {SPEED_MULTS[ui['speed_i']]:.1f}x   "
        f"TELEMETRY {n}/{PLOT_TICKS} samples",
        f"RETURN {ui['return']:+9.2f}   FUEL {env.fuel_used:6.2f}   "
        f"LAST: {last}",
    ]


# ---------------------------------------------------------------
# WORLD -> PIXELS (the ONE flip: world y-up becomes screen y-down)
# ---------------------------------------------------------------
def x_to_px(x):
    """World meters (-50..+50) -> scene pixels (the OOB walls).

    -50 m lands ON the left wall, +50 m ON the right wall, 0 in the
    exact middle. 5 px/m (the span is 500 px for 100 m).
    """
    return X_L_PX + ((x + WORLD_HALF_WIDTH) / (2 * WORLD_HALF_WIDTH)
                     * (X_R_PX - X_L_PX))


def y_to_px(y):
    """World meters (y-up, 0..80) -> screen pixels (y-down)."""
    return GROUND_Y - (y / CEILING_Y) * (GROUND_Y - CEIL_PX)


def world_to_px(wx, wy):
    """A world point -> a screen point (the composed flip)."""
    return (x_to_px(wx), y_to_px(wy))


def body_to_px(point):
    """A body-frame fixture (from env.geometry) -> screen point."""
    return world_to_px(*point)


# Fonts are cached: creating a font object is expensive, drawing is
# not. pygame.font must be initialised first (run() does that).
_FONTS = {}


def font(size):
    """A monospace font of the given pixel size (cached)."""
    if size not in _FONTS:
        _FONTS[size] = pygame.font.SysFont("consolas", size)
    return _FONTS[size]


# ---------------------------------------------------------------
# DRAWING — one frame of the cockpit
# ---------------------------------------------------------------
def draw_scene(screen, env, ui):
    """The Moon stage: sky, ground, pad + flags, bounds, the lander."""
    scene = pygame.Rect(SCENE_X0, SCENE_Y0, SCENE_W, SCENE_H)
    pygame.draw.rect(screen, C_SKY, scene, border_top_left_radius=10,
                     border_top_right_radius=10)

    # --- the bounds: side walls (x = +/-50 m) and the ceiling ------
    pygame.draw.rect(screen, C_WALL,
                     pygame.Rect(X_L_PX - 6, CEIL_PX, 6,
                                 GROUND_Y - CEIL_PX))
    pygame.draw.rect(screen, C_WALL,
                     pygame.Rect(X_R_PX, CEIL_PX, 6,
                                 GROUND_Y - CEIL_PX))
    for cx in range(X_L_PX, X_R_PX, 34):            # dashed ceiling
        pygame.draw.line(screen, C_WALL, (cx, CEIL_PX),
                         (cx + 18, CEIL_PX), 3)

    # --- the ground, the pad, and the two flags --------------------
    pygame.draw.rect(screen, C_GROUND_F,
                     pygame.Rect(X_L_PX, GROUND_Y,
                                 X_R_PX - X_L_PX,
                                 scene.bottom - GROUND_Y))
    pygame.draw.line(screen, C_GROUND, (X_L_PX, GROUND_Y),
                     (X_R_PX, GROUND_Y), 3)
    pad_l, pad_r = x_to_px(-PAD_HALF), x_to_px(PAD_HALF)
    pygame.draw.line(screen, C_PAD, (pad_l, GROUND_Y),
                     (pad_r, GROUND_Y), 6)
    for fx in (pad_l, pad_r):                       # the flags
        pygame.draw.line(screen, C_FLAG, (fx, GROUND_Y),
                         (fx, GROUND_Y - 22), 3)
        pygame.draw.polygon(screen, C_FLAG,
                            [(fx, GROUND_Y - 22),
                             (fx + 12, GROUND_Y - 17),
                             (fx, GROUND_Y - 12)])

    # --- the lander: the SAME fixtures the physics rules read ------
    g = env.geometry()
    hull = [body_to_px(p) for p in g["hull"]]
    pygame.draw.polygon(screen, C_LANDER, hull)
    pygame.draw.polygon(screen, C_LANDER_R, hull, 2)
    for joint, tip, touching in (
        (g["leg_joints"][0], g["leg_tips"][0], env.leg_r),
        (g["leg_joints"][1], g["leg_tips"][1], env.leg_l),
    ):
        pygame.draw.line(screen, C_LEG, body_to_px(joint),
                         body_to_px(tip), 4)
        if touching:                                # in the dirt NOW
            pygame.draw.circle(screen, C_TOUCH, body_to_px(tip), 5)

    # --- the engines, speaking for the last action -----------------
    action = ui.get("last_action")
    if action is not None and not env.done:
        theta = env.theta
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Body up-axis is (-sin, cos); the plume points the OPPOSITE
        # way (away = what thrust pushes the ship off of).
        away = (sin_t, -cos_t)
        perp = (cos_t, sin_t)
        cx, cy = env.x, env.y
        if action == MAIN:
            flick = 1.9 + 0.35 * (env.steps % 3)     # cheap flicker
            base = [world_to_px(cx + away[0] * 0.8 + s * perp[0] * 0.5,
                                cy + away[1] * 0.8 + s * perp[1] * 0.5)
                    for s in (-1, 1)]
            tip = world_to_px(cx + away[0] * flick,
                              cy + away[1] * flick)
            pygame.draw.polygon(screen, C_FLAME,
                                [base[0], base[1], tip])
        elif action in (ROT_LEFT, ROT_RIGHT):
            # A puff at the hull edge; the rotation is visible in
            # the scene itself, the lamp tells you WHICH button.
            s = 1.1 if action == ROT_LEFT else -1.1
            pygame.draw.circle(
                screen, C_PUFF,
                world_to_px(cx + perp[0] * s, cy + perp[1] * s), 4)


def draw_plots(screen, env, ui):
    """The left telemetry stack: six scrolling mini-plots."""
    pygame.draw.rect(screen, C_PANEL,
                     pygame.Rect(PLOT_X, SCENE_Y0, PLOT_W, SCENE_H),
                     border_radius=10)
    for i, (key, label, lo, hi, color) in enumerate(PLOT_PANELS):
        top = PLOT_TOP + i * (PLOT_PANE_H + PLOT_GAP)
        pane = pygame.Rect(PLOT_X + 8, top, PLOT_W - 16, PLOT_PANE_H)
        pygame.draw.rect(screen, C_BG, pane, border_radius=6)
        pygame.draw.rect(screen, C_PANEL_R, pane, 1, border_radius=6)
        screen.blit(font(13).render(label, True, C_TEXT_DIM),
                    (pane.x + 6, pane.y + 2))

        data = ui["history"].get(key) or []
        if key == "legs":
            _plot_legs(screen, pane, data)
        elif len(data) >= 2:
            latest = data[-1]
            screen.blit(font(13).render(f"{latest:+.2f}", True, color),
                        (pane.right - 58, pane.y + 2))
            _plot_series(screen, pane, data, lo, hi, color)


def _plot_series(screen, pane, data, lo, hi, color):
    """One polyline: the last PLOT_TICKS samples across the pane."""
    data = data[-PLOT_TICKS:]
    n = len(data)
    span = pane.width - 12
    points = []
    for i, v in enumerate(data):
        px = pane.x + 6 + (i / (n - 1)) * span
        py = _value_to_py(v, lo, hi, pane)
        points.append((px, py))
    pygame.draw.lines(screen, color, False, points, 2)


def _plot_legs(screen, pane, data):
    """Two 0/1 traces: left leg and right leg contact over time."""
    data = data[-PLOT_TICKS:]
    n = len(data)
    if n < 2:
        screen.blit(font(12).render("waiting for a touch...",
                                    True, C_TEXT_DIM),
                    (pane.x + 6, pane.y + pane.h // 2 - 8))
        return
    span = pane.width - 12
    for idx, color in ((0, (150, 230, 160)), (1, (110, 200, 250))):
        points = [(pane.x + 6 + (i / (n - 1)) * span,
                   _value_to_py(float(leg[idx]), 0.0, 1.0, pane))
                  for i, leg in enumerate(data)]
        pygame.draw.lines(screen, color, False, points, 2)


def _value_to_py(v, lo, hi, pane):
    """Clamped value -> y pixel between the pane's pad and floor."""
    top, bottom = pane.y + 18, pane.bottom - 4
    v = min(max(v, lo), hi)
    return bottom - (v - lo) / (hi - lo) * (bottom - top)


def draw_keypad(screen, ui):
    """The controller, top-right: three buttons + the IDLE lamp.

    Mirrors the LAST action taken — yours or the brain's — so the
    window always answers "what did we just press?".
    """
    box = pygame.Rect(X_R_PX - 148, CEIL_PX - 24, 148, 118)
    pygame.draw.rect(screen, C_PANEL, box, border_radius=10)
    pygame.draw.rect(screen, C_PANEL_R, box, 1, border_radius=10)
    screen.blit(font(13).render("CONTROLS", True, C_TEXT_DIM),
                (box.x + 10, box.y + 6))

    action = ui.get("last_action")
    chip_y = box.y + 30
    chips = (
        (ROT_LEFT,  "L", -1),
        (MAIN,      "T", 0),
        (ROT_RIGHT, "R", 1),
    )
    for act, glyph, _dir in chips:
        rect = pygame.Rect(0, 0, 40, 34)
        rect.centerx = (box.centerx + _dir * 47)
        rect.y = chip_y
        lit = action == act
        pygame.draw.rect(screen, C_FLAME if lit else C_BG, rect,
                         border_radius=8)
        pygame.draw.rect(screen, C_PANEL_R, rect, 1, border_radius=8)
        # the glyph: an arrow telling you what the button DOES
        cxc, cyc = rect.center
        if act == MAIN:
            tri = [(cxc, cyc - 10), (cxc + 8, cyc + 7), (cxc - 8, cyc + 7)]
        elif act == ROT_LEFT:
            tri = [(cxc - 10, cyc), (cxc + 7, cyc - 8), (cxc + 7, cyc + 8)]
        else:
            tri = [(cxc + 10, cyc), (cxc - 7, cyc - 8), (cxc - 7, cyc + 8)]
        pygame.draw.polygon(screen, C_TEXT if not lit else C_BG, tri)

    # the IDLE lamp: lit exactly when nothing was pressed
    idle_rect = pygame.Rect(box.x + 10, box.bottom - 34,
                            box.width - 20, 22)
    lit = action == IDLE
    pygame.draw.rect(screen, (70, 130, 95) if lit else C_BG,
                     idle_rect, border_radius=8)
    pygame.draw.rect(screen, C_PANEL_R, idle_rect, 1, border_radius=8)
    screen.blit(font(13).render("IDLE", True,
                                C_TEXT if lit else C_TEXT_DIM),
                (idle_rect.centerx - 18, idle_rect.y + 3))


def draw_hud(screen, env, ui):
    """The bottom bar: mode badge, the flight numbers, the hints."""
    hud = pygame.Rect(PAD, SCENE_Y0 + SCENE_H + GAP,
                      WIDTH - 2 * PAD, HUD_H)
    pygame.draw.rect(screen, C_HUD_BG, hud, border_radius=10)

    mode = ui["mode"]
    badge = font(22).render(mode, True, C_BG)
    badge_bg = pygame.Surface((badge.get_width() + 12,
                               badge.get_height()))
    badge_bg.fill(MODE_COLORS[mode])
    screen.blit(badge_bg, (hud.x + 12, hud.y + 10))
    screen.blit(badge, (hud.x + 18, hud.y + 10))

    x = hud.x + 40 + badge.get_width()
    for i, line in enumerate(hud_lines(env, ui)):
        screen.blit(font(17).render(line, True, C_TEXT),
                    (x, hud.y + 12 + i * 22))

    if mode == "HUMAN":
        hint = "[UP] main  [LEFT/RIGHT] rotate  [SPACE] brain  " \
               "[R] retry  [+/-] speed"
    else:
        hint = "[SPACE] you fly  [R] retry  [+/-] speed  [V] charts"
    screen.blit(font(14).render(hint, True, C_TEXT_DIM),
                (hud.x + 12, hud.bottom - 22))


def draw_outcome(screen, env, ui):
    """Big banner across the scene: the attempt's verdict / a notice."""
    mid = SCENE_Y0 + SCENE_H // 2

    if ui["outcome"] is None:
        notice = ui.get("notice")
        if not notice:
            return
        band = pygame.Surface((SCENE_W, 100), pygame.SRCALPHA)
        band.fill((10, 12, 16, 190))
        screen.blit(band, (SCENE_X0, mid - 50))
        big = font(24).render(notice, True, (255, 210, 90))
        screen.blit(big, big.get_rect(center=(scene_center_x(), mid - 14)))
        if ui.get("notice_sub"):
            sub = font(16).render(ui["notice_sub"], True, C_TEXT_DIM)
            screen.blit(sub, sub.get_rect(
                center=(scene_center_x(), mid + 14)))
        return

    text, color = banner_for(ui["outcome"], ui["crash_type"])
    band = pygame.Surface((SCENE_W, 120), pygame.SRCALPHA)
    band.fill((10, 12, 16, 190))
    screen.blit(band, (SCENE_X0, mid - 60))

    big = font(38).render(text, True, color)
    screen.blit(big, big.get_rect(center=(scene_center_x(), mid - 14)))

    sub_text = ("press R to retry" if ui["mode"] == "HUMAN"
                else "restarting...")
    sub = font(20).render(sub_text, True, C_TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(scene_center_x(), mid + 24)))


def scene_center_x():
    """The scene band's horizontal middle (banner centering)."""
    return SCENE_X0 + SCENE_W // 2


def draw(screen, env, ui):
    """Render one complete frame (scene -> plots -> keypad -> HUD)."""
    screen.fill(C_BG)
    draw_scene(screen, env, ui)
    draw_plots(screen, env, ui)
    draw_keypad(screen, ui)
    draw_hud(screen, env, ui)
    draw_outcome(screen, env, ui)


# ---------------------------------------------------------------
# THE MAIN LOOP — events in, frames out
# ---------------------------------------------------------------
DISMISS_KEYS = (
    pygame.K_SPACE, pygame.K_r, pygame.K_UP,
    pygame.K_LEFT, pygame.K_RIGHT,
    pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS,
    pygame.K_MINUS, pygame.K_KP_MINUS,
)


def run():
    """Open the window and fly until the user quits."""
    pygame.init()
    _FONTS.clear()   # a previous pygame.quit() poisoned the cache
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Lunar Lander DQN - RL Fundamentals")

    env = LunarLander()
    ui = fresh_ui()
    restart(env, ui)
    trained = None       # loaded lazily when TRAINED mode is entered
    tick_budget = 0.0    # saved-up fraction of physics ticks
    pause_timer = 0.0    # counts ms AFTER a finished TRAINED episode

    running = True
    while running:
        elapsed = clock.tick(FPS)

        # ---- 1. EVENTS (keyboard / closing the window) -----------
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
                    restart(env, ui)
                    pause_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    ui["mode"] = next_mode(ui["mode"])
                    pause_timer = 0.0
                    if ui["mode"] == "TRAINED":
                        # (Re)load so a fresh `python train.py` run is
                        # picked up — quick course if nothing exists.
                        trained = ensure_trained(ui)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS,
                                   pygame.K_KP_PLUS):
                    ui["speed_i"] = min(ui["speed_i"] + 1,
                                        len(SPEED_MULTS) - 1)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    ui["speed_i"] = max(ui["speed_i"] - 1, 0)
                elif event.key == pygame.K_v:
                    if not launch_visualizer():
                        ui["notice"] = "No charts to open yet"
                        ui["notice_sub"] = "run: python visualize.py"

        # ---- 2. THE PHYSICS TIMER (real-time, HUMAN and TRAINED) --
        # Speeds first: TRAINED's +/- ladder AND the notice pause
        # (a warning on screen pauses the auto-pilot, not your hands).
        flying = not env.done
        if flying and not (ui["notice"] and ui["mode"] == "TRAINED"):
            mult = 1.0 if ui["mode"] == "HUMAN" \
                else SPEED_MULTS[ui["speed_i"]]
            tick_budget += elapsed * mult / TICK_MS
            n_ticks = min(int(tick_budget), 8)   # cap: never spiral
            tick_budget -= int(tick_budget)
            for _ in range(n_ticks):
                if ui["mode"] == "HUMAN":
                    action = action_from_held(pygame.key.get_pressed())
                else:
                    action = trained.choose_action(env.state,
                                                   explore=False)
                apply_step(env, ui, action)
                if env.done:
                    break
        elif env.done and ui["mode"] == "TRAINED":
            pause_timer += elapsed
            if pause_timer >= WATCH_PAUSE_MS and not ui["notice"]:
                pause_timer = 0.0
                restart(env, ui)         # auto-start the next attempt

        # ---- 3. PAINT the frame -----------------------------------
        draw(screen, env, ui)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
