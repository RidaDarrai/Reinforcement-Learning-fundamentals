"""
play.py — THE WINDOW (part C of our RL project)
================================================

Until now the world lived in the terminal. This file gives it a real
game window: colored tiles, a HUD, and two ways to interact.

Run it (from this folder):

    python play.py

Controls
--------
    Arrow keys ... move the agent        (HUMAN mode)
    + / - ........ speed up / slow down  (WATCH / TRAINED modes)
    R ............ restart the episode
    SPACE ........ cycle to the next mode
    ESC or Q ..... quit

Modes
-----
    HUMAN   : you play the maze yourself — the best way to FEEL what
              the task demands before asking an agent to learn it.
    WATCH   : a mindless random agent stumbles around by itself —
              the baseline to beat.
    TRAINED : the GRADUATE from train.py plays greedily (no random
              moves, no learning — pure performance). Needs
              q_table.json next to this file (run: python train.py);
              otherwise a hint banner explains what to do.

The window talks to the world only through the same two calls
everything else uses (reset / step) — exactly how Gymnasium's own
environments render themselves. Nothing here knows the maze rules;
environment.py remains the single source of truth.
"""

import os
import random

import pygame

from agent import QAgent
from environment import (
    GridWorld, UP, DOWN, LEFT, RIGHT, N_ACTIONS, MAX_STEPS,
    EMPTY, WALL, GOAL, PIT,
)

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
C_MODE_TRAINED = (110, 225, 150)  # "TRAINED" badge color
C_WIN  = (80, 220, 120)        # win banner
C_LOSE = (255, 95, 85)         # loss banner
C_TIME = (255, 210, 90)        # timeout banner

# The mode cycle that SPACE walks through, and each mode's badge color.
MODES = ("HUMAN", "WATCH", "TRAINED")
MODE_COLORS = {
    "HUMAN": C_MODE_HUMAN,
    "WATCH": C_MODE_WATCH,
    "TRAINED": C_MODE_TRAINED,
}


def next_mode(current):
    """The mode SPACE switches to: HUMAN -> WATCH -> TRAINED -> HUMAN."""
    return MODES[(MODES.index(current) + 1) % len(MODES)]


# The trained brain lives next to this file (train.py saves it there),
# so it works no matter which folder you launched python from.
Q_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "q_table.json")


def load_trained_agent(path=Q_TABLE_PATH):
    """Load the graduate from disk, or None if there isn't one yet.

    Returning None (instead of raising) lets the window degrade
    gracefully: no q_table.json -> friendly hint banner, no crash.
    """
    try:
        return QAgent.load(path)
    except (OSError, ValueError, KeyError):
        return None        # missing file / half-written / wrong shape

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

    # --- line 1: mode badge + step counter + running score ----------
    # Badge: text on a colored patch sized to fit (12px total margin).
    # The TRAINED badge is the longest word in the HUD, so every width
    # below is chosen to keep the worst case inside the window.
    mode = ui["mode"]
    badge_text = font(22).render(mode, True, C_BG)
    badge_bg = pygame.Surface((badge_text.get_width() + 12,
                               badge_text.get_height()))
    badge_bg.fill(MODE_COLORS[mode])
    badge_x = hud_rect.x + 12
    screen.blit(badge_bg, (badge_x, hud_rect.y + 10))
    screen.blit(badge_text, (badge_x + 6, hud_rect.y + 10))

    info = font(22).render(
        f"  STEP {env.steps}/{MAX_STEPS}   REWARD {ui['reward']:+.1f}",
        True, C_TEXT,
    )
    info_x = badge_x + badge_bg.get_width() + 6
    screen.blit(info, (info_x, hud_rect.y + 10))

    # Timer modes (WATCH / TRAINED) show the current speed as a
    # multiplier — small font so the TRAINED badge + worst-case
    # numbers (100/100, -10.0) still fit the window.
    if mode in ("WATCH", "TRAINED"):
        speed = SPEED_BASE_MS / SPEED_LEVELS_MS[ui["speed_i"]]
        speed_img = font(17).render(f" {speed:.1f}x", True, MODE_COLORS[mode])
        screen.blit(speed_img,
                    (info_x + info.get_width() + 4, hud_rect.y + 14))

    # --- line 2: control hints (they change with the mode) -----------
    # Kept short on purpose: longer text would clip at the window edge.
    if mode == "HUMAN":
        hint = "[arrows] move  [SPACE] modes  [R] restart  [ESC] quit"
    else:
        hint = "[+/-] speed  [SPACE] modes  [R] restart  [ESC] quit"
    screen.blit(font(16).render(hint, True, C_TEXT_DIM),
                (hud_rect.x + 12, hud_rect.y + 47))


def draw_outcome(screen, env, ui):
    """Big banner across the middle: episode result, or a notice
    (e.g. "you picked TRAINED but there's no trained agent yet")."""
    if ui["outcome"] is None:
        # No episode result -> show the notice, if the mode set one.
        notice = ui.get("notice")
        if not notice:
            return
        band = pygame.Surface((WIDTH, 100), pygame.SRCALPHA)
        band.fill((10, 12, 16, 190))
        screen.blit(band, (0, (PAD + BOARD_H) // 2 - 50))
        big = font(24).render(notice, True, C_TIME)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))
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
def run():
    """Open the window and play until the user quits."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("GridWorld Maze - RL Fundamentals")
    clock = pygame.time.Clock()

    env = GridWorld()
    ui = {"mode": "HUMAN", "reward": 0.0, "outcome": None,
          "speed_i": SPEED_DEFAULT_I, "notice": None}
    restart(env, ui)
    trained = None   # loaded lazily when TRAINED mode is entered

    watch_timer = 0.0   # counts ms until the next auto-move/pause

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
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    restart(env, ui)
                    watch_timer = 0.0
                elif event.key == pygame.K_SPACE:
                    ui["mode"] = next_mode(ui["mode"])
                    ui["notice"] = None
                    if ui["mode"] == "TRAINED":
                        # (Re)load so a fresh `python train.py` run is
                        # picked up without restarting the window.
                        trained = load_trained_agent()
                        if trained is None:
                            ui["notice"] = "No trained agent - run: python train.py"
                    watch_timer = 0.0
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

        # ---- 2. TIMER MODES act on a timer, not on events ------------
        # WATCH and TRAINED share the clock, speed ladder, and
        # auto-restart; they differ only in WHO picks the moves.
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
        draw(screen, env, ui)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run()
