"""
play.py — THE COCKPIT (step 4 of the lunar-lander-gym project)
==============================================================

The window where you watch the graduate fly the REAL world. The
landings are the same problem class as 04's, but the picture now
comes from gym's own renderer — Box2D-drawn frames piped into our
pygame window — and the instrument column lives beside it, same
habit as 04: the state/reward flow in -> panels + text out.

WHAT 04 HAD THAT THIS WINDOW NOW CARRIES TOO (the port that grew
this file): 04's cockpit drew SIX SLIDING telemetry plots beside
the flight (x, y, vx, vy, tilt, leg contact) — watching the speeds
fight the burns is the whole skill. This window ships the same
six channels, and this time they respect two new cockpit rules:

    1. TOGGLE BUTTONS — each graph has a clickable chip (the
       channel's own color when ON, dark when OFF). A mouse click
       toggles; keys 1..6 toggle the same channels from the desk.
    2. SLIDING ONLY — every plot is a ROLLING WINDOW of the last
       PLOT_TICKS ticks (~6 s of gym time). The window travels
       with the replay; nothing here ever draws a whole-episode
       static curve. Charts of the whole run belong to
       visualize.py — the cockpit shows the flight MOVING.

THE MANUAL (all of it on one screen):
    R       restart the current attempt (fresh spawn)
    S       toggle sample mode  GREEDY <-> SAMPLED
            GREEDY  = act_greedy: argmax of the policy — the exam
                      mode, deterministic, what "can it land?" means
            SAMPLED = act(): the dice the TRAINING policy rolls (the
                      reference's own test.py flys sampled) — not the
                      graduate's best self, just its honest shape
    + / -   playback speed 0.5x .. 4x (04's speed ladder; 1.0x is
            real time — gym steps at 50 ticks/s)
    1..6    toggle telemetry graphs (same as clicking the chips)
    mouse   click a chip to toggle its graph
    V       open visualize.py's charts (own process, like 04)
    ESC     close
(no HUMAN mode this time: 04's hands-on cockpit was OUR hand-built
world's toy; here the world is gym's and the brain is the show.)

VISUAL DNA — GYM'S OWN COLORS
-----------------------------
The picture comes from gym, so the instruments dress in gym's
palette (sampled straight out of a rendered LunarLander-v3 frame,
pinned by tests): BLACK sky panels, WHITE terrain/lines, the
lander's PURPLE (128, 102, 230) and its darker outline purple
(77, 77, 128), the pad flags' YELLOW (204, 204, 0), the smoke
plume's fresh RED (255, 127, 127) and aged GREY (129, 129, 129).
Every instrument color in the window is one of gym's paints.

THE BRAIN SUPPLY (the 04 lesson, kept):
    no actor_critic_model.pt? A quick course (a small Trainer run,
    ~a minute) is flown RIGHT HERE into quick_course_* files —
    never the official run's artifacts. Corrupt/half-written
    files (training can be mid-write) degrade to the same path.

EVERYTHING TESTABLE STAYS FRAMELESS:
    load_trained_agent / quick_course / hud lines / run_episode /
    history + sliding / series geometry / toggle buttons /
    banner + lamp layout all work with pygame on the dummy video
    driver (the test suite sets SDL_VIDEODRIVER=dummy) — no test
    ever opens a real window.

Run it:

    python play.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import os          # artifact paths anchored next to THIS file
import subprocess  # visualize.py opens in its OWN process (matplotlib
                   # and pygame fight over the window system)
import sys         # the interpreter path for that subprocess

import pygame      # our surface: gym's frames blit under the column

# The world (rgb_array so every step ALSO yields a pixel frame —
# the drawing comes from gym, not from us) and the brain pieces.
from environment import LunarLanderEnv, OBS_DIM, N_ACTIONS, ACTION_NAMES
from agent import ActorCritic
from train import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))
DIPLOMA_PATH = os.path.join(HERE, "actor_critic_model.pt")   # official
QUICK_EPISODES = 400        # the quick course's length (~1 min) —
                            # scaled down official run, brain only
QUICK_LOG    = os.path.join(HERE, "quick_course_log.json")
QUICK_DIPLOMA = os.path.join(HERE, "quick_course.pt")

# ---------------------------------------------------------------
# THE VIEW — gym's frame stays 1:1; the cockpit column mirrors 04
# ---------------------------------------------------------------
# TOP-LEFT : gym's 600x400 render, blitted verbatim.
# TOP-RIGHT: the instrument column — HUD text, the six toggle
#            chips, the six sliding telemetry plots.
# BOTTOM   : the verdict band (outcome banner) + the four button
#            lamps (which button the brain pressed — 04's keypad).
FRAME_W, FRAME_H = 600, 400
TELE_W     = 280                    # the right column (04's panel habit)
BOTTOM_H   = 240                    # banner + lamps under the video
WINDOW_SIZE = (FRAME_W + TELE_W, FRAME_H + BOTTOM_H)   # (880, 640)
FPS = 30                            # window redraws per second
PLAYBACK_HZ = 50.0                  # gym steps at 50 ticks/s (its FPS):
                                    # speed 1.0x plays REAL-Time.

# Hot keys (named constants — the manual is pinned by tests):
KEY_RESTART = pygame.K_r
KEY_MODE    = pygame.K_s     # greedy <-> sampled
KEY_CHARTS  = pygame.K_v
KEY_QUIT    = pygame.K_ESCAPE
KEY_SPEED_UP   = (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)
KEY_SPEED_DOWN = (pygame.K_MINUS, pygame.K_KP_MINUS)
KEY_PLOT_TOGGLES = (
    pygame.K_1, pygame.K_2, pygame.K_3,
    pygame.K_4, pygame.K_5, pygame.K_6,
)   # key i toggles PLOT_KEYS[i]

SPEED_MULTS     = (0.5, 1.0, 2.0, 4.0)
SPEED_DEFAULT_I = 1

# The HUD's text grid: (rows of (label key, formatter)) — rendered
# one line at a time from the flight record, so tests can check the
# WORDS without a window.
HUD_LINES = (
    ("attempt",  "attempt #{attempt}"),
    ("episode",  "tick {tick}"),
    ("return",   "return {return:+8.1f}"),
    ("mode",     "policy {mode}"),
    ("position", "x {x:+.2f}  y {y:+.2f}"),
    ("speed",    "vx {vx:+.2f}  vy {vy:+.2f}"),
    ("attitude", "tilt {tilt:+7.1f} deg"),
    ("legs",     "legs {legs}"),
    ("banner",   "{banner}"),
)


# ===============================================================
# THE PALETTE — sampled from gymnasium's own LunarLander-v3 render
# ===============================================================
# Every color below was read OUT of a rendered frame (the pins live
# in test_play.py): the night-sky BLACK, the regolith WHITE, the
# lander's purple hull + darker purple outline, the pad flags'
# yellow, and the engine smoke's red->grey fade.
GYM_BLACK    = (0, 0, 0)
GYM_WHITE    = (255, 255, 255)
GYM_PURPLE   = (128, 102, 230)    # LANDER_POLY color1 — the lander
GYM_PURPLE_D = (77, 77, 128)      # lander color2 — hull outline
GYM_YELLOW   = (204, 204, 0)      # the helipad flags
GYM_RED      = (255, 127, 127)    # a FRESH main-engine smoke particle
GYM_GREY     = (129, 129, 129)    # an AGED particle / aa-line grey
GYM_DARKGREY = (65, 65, 65)       # the darkest aa grey

# The terminal verdicts take the LOUDEST paints we have:
BANNER_COLORS = {
    "landed": GYM_YELLOW,         # the flags' yellow — a good landing
    "crashed": GYM_RED,           # fresh smoke red — a bad ending
    "out_of_bounds": GYM_RED,     # same bill as a crash, same paint
    "timeout": GYM_GREY,          # stale smoke grey — a stalled life
}


def banner_for(outcome):
    """(text, color) for the big bottom-band verdict, per ending."""
    text = {"landed": "TOUCHDOWN!", "crashed": "CRASHED!",
            "out_of_bounds": "OUT OF BOUNDS!", "timeout": "TIMED OUT."
            }.get(outcome, "?")
    return text, BANNER_COLORS.get(outcome, GYM_GREY)


# ===============================================================
# TELEMETRY — six sliding plots + toggle buttons (04's port)
# ===============================================================
PLOT_TICKS = 300    # the rolling window: 300 ticks * 0.02 s = ~6 s.
                    # SLIDING ONLY: the window ROLLS with the replay,
                    # nothing ever grows to the whole episode.

# The channels. (key, label, axis lo, axis hi, line color) — ranges
# are gym's OWN observation-space bounds (x/y +-2.5, speeds +-10,
# angle shown in degrees), colors come from the palette above, one
# distinct family per channel so each chip self-matches the line it
# toggles. legs is a special panel: two 0/1 trace lines.
PLOT_PANELS = (
    ("x",    "x",     -1.0, 1.0, GYM_WHITE),
    ("y",    "y",     -2.5, 2.5, GYM_YELLOW),
    ("vx",   "vx",   -10.0, 10.0, GYM_GREY),
    ("vy",   "vy",   -10.0, 10.0, GYM_RED),
    ("tilt", "tilt", -180.0, 180.0, GYM_PURPLE),
    ("legs", "legs",  None, None, None),
)
PLOT_KEYS = tuple(p[0] for p in PLOT_PANELS)

# Column geometry (right of the video): HUD text -> the playback
# speed line -> chips -> plots (each drawer respects the line below).
HUD_TOP    = 8       # HUD text start (9 lines x 16 px = the grid)
HUD_LINE_H = 16
CHIP_Y     = HUD_TOP + (len(HUD_LINES) + 1) * HUD_LINE_H + 6  # 174:
                   # the HUD grid + one extra line for playback speed
CHIP_W, CHIP_H = 84, 26
CHIP_COLS, CHIP_GAP = 3, 8
PLOT_TOP   = 236     # below both chip rows
PANE_H     = 62
PLOT_GAP   = 2
PANE_X     = 8            # margins inside the instrument column
PANE_W     = TELE_W - 2 * PANE_X                      # 264

# The four button lamps (04's keypad): which button the brain
# pressed THIS tick — lit in the lander's purple, dark otherwise.
LAMP_W, LAMP_H, LAMP_GAP = 120, 40, 12


def fresh_enabled():
    """Every telemetry graph starts ON (a fresh cockpit)."""
    return {key: True for key in PLOT_KEYS}


def toggle_plot(enabled, key):
    """Flip one graph's visibility in place; return its new state."""
    if key not in enabled:
        raise ValueError(f"unknown telemetry channel {key!r}: "
                         f"use one of {PLOT_KEYS}")
    enabled[key] = not enabled[key]
    return enabled[key]


def plot_button_rect(i):
    """Chip i's click area, in WINDOW pixels (i = index into
    PLOT_KEYS). Two rows of three, directly above the plots."""
    row, col = divmod(i, CHIP_COLS)
    x = FRAME_W + PANE_X + col * (CHIP_W + CHIP_GAP)
    y = CHIP_Y + row * (CHIP_H + 6)
    return pygame.Rect(x, y, CHIP_W, CHIP_H)


def plot_button_at(pos):
    """Which channel's chip holds this window position (or None)."""
    for i, key in enumerate(PLOT_KEYS):
        if plot_button_rect(i).collidepoint(pos):
            return key
    return None


def lamp_rect(action):
    """One lamp's display area in WINDOW pixels (action 0..3)."""
    x = 42 + action * (LAMP_W + LAMP_GAP)
    return pygame.Rect(x, FRAME_H + 108, LAMP_W, LAMP_H)


# ---------------------------------------------------------------
# THE SLIDING HISTORY — one ring of numbers per channel
# ---------------------------------------------------------------
def fresh_history():
    """Empty telemetry: one list per plotted channel."""
    return {key: [] for key in PLOT_KEYS}


def fill_history(history, row):
    """Append ONE flight record's tick to every channel.

    Args: history — fresh_history()'s dict; row — one of
          run_episode()'s flight records (its numeric channels:
          x, y, vx, vy, tilt in DEGREES, both leg flags).
    """
    history["x"].append(row["x"])
    history["y"].append(row["y"])
    history["vx"].append(row["vx"])
    history["vy"].append(row["vy"])
    history["tilt"].append(row["tilt"])
    history["legs"].append((row["legL"], row["legR"]))


def slid(data):
    """The SLIDING window: only the last PLOT_TICKS samples.

    The one gate every plot walks through — "sliding only" is
    enforced in exactly this place, so no graph can ever grow
    into a static whole-episode curve by accident.
    """
    return data[-PLOT_TICKS:]


def series_points(pane, data, lo, hi):
    """A rolling channel -> px points ready to draw.

    Args: pane — (x, y, w, h) window pixels; data — already-SLID
          values; lo/hi — the axis. Values clamp into [lo, hi]
          (gym's spawns/bounds can kiss the edges).
    Returns: [(px, py)...] — oldest left, newest right, so the
          newly-arrived tick always enters at the same edge and
          the whole line SLIDES left as the replay advances.
    """
    x, y, w, h = pane
    left, right = x + 4, x + w - 4
    top, bottom = y + 16, y + h - 3        # 16 px of label headroom
    n = len(data)
    if n < 2:
        return []
    span = right - left
    points = []
    for i, v in enumerate(data):
        px = left + (i / (n - 1)) * span
        v = min(max(v, lo), hi)
        py = bottom - (v - lo) / (hi - lo) * (bottom - top)
        points.append((px, py))
    return points


# ---------------------------------------------------------------
# THE BRAIN SUPPLY — load, or fly the quick course
# ---------------------------------------------------------------
def load_trained_agent(path=None):
    """Load the graduate from disk, or None if there isn't one.

    Returning None (instead of raising) lets the window degrade
    gracefully — the 04 contract. Corrupt or half-written
    checkpoints (the file can be mid-write while training runs)
    also yield None.
    """
    try:
        return ActorCritic.load(DIPLOMA_PATH if path is None else path)
    except Exception:
        return None


def quick_course(episodes=QUICK_EPISODES, log_path=None,
                 model_path=None):
    """The stand-in diploma: a small Trainer run, done right here.

    Shrinks the school to a ~1-minute course (exams kept, cadence
    scaled) and writes its artifacts to the quick_course_* files
    (redirectable for tests) — the official run's files are NEVER
    touched by play.py.
    """
    trainer = Trainer(
        episodes=episodes,
        exam_every=max(2, episodes // 4),  # fresh exams each quarter
        exam_episodes=4,
        seed=7,                            # official run varies itself
        log_path=QUICK_LOG if log_path is None else log_path,
        model_path=(QUICK_DIPLOMA if model_path is None
                    else model_path),
    )
    trainer.run()
    return trainer.policy


def ensure_trained():
    """The graduate, or the quick course behind it — the 04 promise."""
    agent = load_trained_agent()
    if agent is not None:
        return agent
    print("no trained brain found - flying a quick course (about a minute)")
    return quick_course()


# ---------------------------------------------------------------
# ONE EPISODE, COLLECTED (frameless; the window just plays it)
# ---------------------------------------------------------------
def run_episode(agent, seed=0, sample=False, max_ticks=1000):
    """Fly ONE episode against the world; collect everything the
    window and the tests want.

    Args: agent  — the brain (greedy act_greedy vs act() by `sample`)
          seed   — the world's first-reset seed (this env instance
                   is fresh here, so seed=seed replays exactly)
          sample — False = greedy; True = the TRAINING dice
          max_ticks — a hard cap WELL inside the world's own
                   1,000-tick clock... no: this cap only exists so a
                   stuck nobody-lands loop is impossible in TESTS;
                   the world's own truncation fires first everywhere.

    Returns (frames, hud_rows, outcome, return, ticks):
        frames   — list of gym's numpy frames (one per tick)
        hud_rows — dicts holding what HUD_LINES render from, PLUS
                   the numeric telemetry the sliding plots and the
                   button lamps read (tilt in degrees, legL/legR
                   flags as ints, the action int)
        outcome  — landed / crashed / out_of_bounds / timeout
    """
    mode = "SAMPLED" if sample else "GREEDY"
    world = LunarLanderEnv(seed=seed, render_mode="rgb_array")
    state = world.reset()

    frames, rows = [], []
    ret = 0.0
    banner = ""
    while True:
        action = agent.act(state) if sample else agent.act_greedy(state)
        state, reward, done, info = world.step(action)
        ret += reward
        frame = world.render()      # gym's rgb_array (None only if
                                    # the mode got lost — expected
                                    # rgb_array was pinned above)
        frames.append(frame)
        legs = int(bool(state[6])) + int(bool(state[7]))
        rows.append({
            "attempt": 1, "tick": world.steps, "return": ret,
            "mode": mode, "x": state[0], "y": state[1],
            "vx": state[2], "vy": state[3],
            "tilt": state[4] * 180.0 / 3.141592653589793,
            "legs": f"L{state[6]:.0f}/R{state[7]:.0f}",
            "legL": int(bool(state[6])),   # the sliding plots' two
            "legR": int(bool(state[7])),   # rolling 0/1 contact lines
            "action": int(action),         # the four button lamps
            "banner": banner,
        })
        if done:
            banner = f"{info['outcome']}!"
            rows[-1]["banner"] = banner
            break
    world.close()
    return frames, rows, info["outcome"], ret, world.steps


# ---------------------------------------------------------------
# THE HUD TEXT GRID (frameless render — the window just blits it)
# ---------------------------------------------------------------
def hud_lines(row):
    """The HUD text, built from one flight record row.

    The FORMATTERS live in HUD_LINES, so the window shows exactly
    what the tests assert (the 04 lesson: words pinned in one
    place, drawn from everywhere).
    """
    lines = []
    for _key, template in HUD_LINES:
        try:
            lines.append(template.format(**row))
        except (KeyError, IndexError):
            # A missing field is a coding bug, not a display
            # nicety — scream here rather than show "None" in the
            # cockpit (tests pin a distinct wording)
            lines.append(f"?missing {template}")
    return lines


def load_frames_for_pygame(frames):
    """gym's numpy frames -> pygame surfaces (the blit bridge).

    gymnasium yields (H, W, 3) uint8; pygame's surfarray wants
    the axes transposed — the canonical swapaxes from pygame's
    docs, one line, no copies beyond the surface itself.
    """
    import numpy as np   # only here: numpy is gym's dependency and
    return [pygame.surfarray.make_surface(np.swapaxes(f, 0, 1))
            for f in frames]


# ===============================================================
# DRAWING — the cockpit, dressed in gym's palette
# ===============================================================
# Fonts are cached: creating a font object is expensive, drawing
# is not. pygame.font must be initialised first (main() does it).
_FONTS = {}


def font(size):
    """A monospace font of the given pixel size (cached)."""
    if size not in _FONTS:
        _FONTS[size] = pygame.font.SysFont("consolas", size)
    return _FONTS[size]


def draw_column(screen):
    """The right column: a sky-black panel for the instruments."""
    pygame.draw.rect(screen, GYM_BLACK,
                     pygame.Rect(FRAME_W, 0, TELE_W,
                                 FRAME_H + BOTTOM_H))


def draw_hud(screen, row, attempt, speed_i):
    """The HUD text grid, WHITE-on-black like gym's lines."""
    row = dict(row, attempt=attempt)
    for y_line, text in enumerate(hud_lines(row)):
        # the last grid line is the banner row — grey; all else white
        color = GYM_WHITE if y_line < len(HUD_LINES) - 1 else GYM_GREY
        img = font(13).render(text, True, color)
        screen.blit(img, (FRAME_W + 10,
                          HUD_TOP + y_line * HUD_LINE_H))
    img = font(13).render(f"playback {SPEED_MULTS[speed_i]:.1f}x   "
                          f"[+/-] speed", True, GYM_GREY)
    screen.blit(img, (FRAME_W + 10, HUD_TOP + len(HUD_LINES)
                      * HUD_LINE_H))    # under the 9-line grid


def draw_chips(screen, enabled):
    """The six toggle chips — the graphs' on/off BUTTONS.

    A lit chip wears ITS channel's line color (the button tells
    you the color of the line it controls); a toggled-OFF chip
    goes dark-grey with a dim label. The legs chip pairs two
    trace colors, so IT wears the flag-side yellow on top. Click
    to toggle; keys 1..6 do the same from the desk.
    """
    for i, (key, label, _lo, _hi, color) in enumerate(PLOT_PANELS):
        rect = plot_button_rect(i)
        on = enabled[key]
        chip_color = color or (GYM_WHITE if on else GYM_DARKGREY)
        pygame.draw.rect(screen, GYM_BLACK, rect)
        pygame.draw.rect(screen, chip_color if on else GYM_DARKGREY,
                         rect, 2 if on else 1)
        chip = font(14).render(
            label, True, chip_color if on else GYM_GREY)
        screen.blit(chip, chip.get_rect(center=rect.center))


def draw_plots(screen, history, upto, enabled):
    """The six sliding channels, played UP TO this frame.

    Args: history — the channels fill_history() has been pouring
          ticks into; upto — how many ticks of the episode have
          PLAYED so far; enabled — the toggle-button states.
    ENABLED panes PACK to the top of the plot area (a toggled-off
    channel leaves no dead slot — the stack compacts). Each pane:
    sky-black bg, darkest-grey border (gym's aa grey), a tiny
    label with the window's newest sample, and the rolling line.
    The legs pane draws TWO rolling 0/1 traces.
    """
    slot = 0                               # the packing cursor
    for key, label, lo, hi, color in PLOT_PANELS:
        if not enabled[key]:
            continue                       # OFF: no pane, no gap
        i = PLOT_KEYS.index(key)
        y = PLOT_TOP + slot * (PANE_H + PLOT_GAP)
        slot += 1
        pane = (FRAME_W + PANE_X, y, PANE_W, PANE_H)
        pane_rect = pygame.Rect(*pane)
        pygame.draw.rect(screen, GYM_BLACK, pane_rect)
        pygame.draw.rect(screen, GYM_DARKGREY, pane_rect, 1)

        data = slid(history[key][:upto])
        head = ("  waiting" if not data
                else f"{data[-1]:+7.2f}" if key != "legs"
                else f"L{data[-1][0]:.0f}/R{data[-1][1]:.0f}")
        if key == "legs":
            screen.blit(font(13).render(label, True, GYM_GREY),
                        (pane[0] + 4, pane[1] + 2))
            for idx, color_pair in ((0, GYM_YELLOW), (1, GYM_PURPLE)):
                pts = series_points(pane,
                                    [float(leg[idx]) for leg in data],
                                    0.0, 1.0)
                if len(pts) >= 2:
                    pygame.draw.lines(screen, color_pair, False,
                                      pts, 2)
        else:
            screen.blit(font(13).render(
                f"{label} {head}", True, GYM_GREY),
                (pane[0] + 4, pane[1] + 2))
            pts = series_points(pane, data, lo, hi)
            if len(pts) >= 2:
                pygame.draw.lines(screen, color, False, pts, 2)


def draw_banner(screen, row):
    """The bottom band's verdict text (invisible while flying)."""
    raw = row.get("banner") or ""
    if not raw:
        return
    outcome = raw[:-1] if raw.endswith("!") else raw.rstrip(".")
    text, color = banner_for(outcome)
    big = font(30).render(text, True, color)
    screen.blit(big, big.get_rect(center=(FRAME_W // 2,
                                          FRAME_H + 56)))


def draw_lamps(screen, action):
    """The four button lamps, lit purple when the brain presses.

    04's keypad, ported: the histogram of what was JUST pressed,
    lit in the lander's own purple, dark in gym's outline purple.
    """
    for a in range(N_ACTIONS):
        rect = lamp_rect(a)
        lit = action == a
        pygame.draw.rect(screen, GYM_BLACK, rect)
        pygame.draw.rect(screen, GYM_PURPLE if lit else GYM_DARKGREY,
                         rect, 2 if lit else 1)
        label = font(15).render(
            ACTION_NAMES[a], True,
            GYM_WHITE if lit else GYM_GREY)
        screen.blit(label, label.get_rect(center=rect.center))


def draw(screen, frames, frame_i, rows, attempt, history, enabled,
         speed_i):
    """One full frame: gym's video + the cockpit column + verdict."""
    screen.fill(GYM_BLACK)
    safe = min(frame_i, len(frames) - 1)
    blit = load_frames_for_pygame([frames[safe]])[0]
    screen.blit(blit, (0, 0))
    row = dict(rows[min(frame_i, len(rows) - 1)])

    draw_column(screen)
    draw_hud(screen, row, attempt, speed_i)
    draw_chips(screen, enabled)
    draw_plots(screen, history, safe + 1, enabled)
    draw_banner(screen, row)
    draw_lamps(screen, row["action"])


# ---------------------------------------------------------------
# THE WINDOW (the only pygame-drawing part; tests use the pieces)
# ---------------------------------------------------------------
def main():
    pygame.init()
    pygame.display.set_caption("05 lunar-lander-gym - TRAINED replay")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    agent = ensure_trained()
    mode_sample = False       # the S toggle: False = GREEDY
    enabled = fresh_enabled() # the toggle chips' memory
    speed_i = SPEED_DEFAULT_I

    episode_seed = 0
    frames, rows, outcome, ret, ticks = run_episode(
        agent, seed=episode_seed, sample=mode_sample)
    frame_i = 0
    filled = 0                # ticks already poured into `history`
    history = fresh_history()
    attempt = 1
    budget = 0.0              # saved-up playback fraction (frames)

    running = True
    while running:
        elapsed = clock.tick(FPS)

        # --- a fresh episode when the replay ran out -------------
        if frame_i >= len(frames):
            episode_seed += 1
            frames, rows, outcome, ret, ticks = run_episode(
                agent, seed=episode_seed, sample=mode_sample)
            frame_i = 0
            filled = 0
            history = fresh_history()
            attempt += 1

        # --- the playback clock: real time at 1.0x ----------------
        rate = PLAYBACK_HZ * SPEED_MULTS[speed_i]
        budget += elapsed * rate / 1000.0
        advance = int(budget)
        budget -= advance
        frame_i = min(frame_i + advance, len(frames) - 1)

        # --- SLIDING telemetry: pour the played ticks into the
        #     channels (skipped ticks pour too, at 2x/4x) ---------
        while filled < frame_i and filled < len(rows):
            fill_history(history, rows[filled])
            filled += 1

        # --- paint: gym's world under our instrument column ------
        draw(screen, frames, frame_i, rows, attempt, history,
             enabled, speed_i)
        pygame.display.flip()

        # --- events ----------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == KEY_QUIT:
                    running = False
                elif event.key == KEY_RESTART:
                    frames, rows, outcome, ret, ticks = run_episode(
                        agent, seed=episode_seed, sample=mode_sample)
                    frame_i, filled = 0, 0
                    history = fresh_history()
                elif event.key == KEY_MODE:
                    mode_sample = not mode_sample
                    frames, rows, outcome, ret, ticks = run_episode(
                        agent, seed=episode_seed, sample=mode_sample)
                    frame_i, filled = 0, 0
                    history = fresh_history()
                elif event.key in KEY_SPEED_UP:
                    speed_i = min(speed_i + 1, len(SPEED_MULTS) - 1)
                elif event.key in KEY_SPEED_DOWN:
                    speed_i = max(speed_i - 1, 0)
                elif event.key in KEY_PLOT_TOGGLES:
                    toggle_plot(
                        enabled,
                        PLOT_KEYS[KEY_PLOT_TOGGLES.index(event.key)])
                elif event.key == KEY_CHARTS:
                    subprocess.Popen([sys.executable, "visualize.py"],
                                     cwd=HERE)
            elif event.type == pygame.MOUSEBUTTONDOWN \
                    and event.button == 1:
                channel = plot_button_at(event.pos)
                if channel is not None:
                    toggle_plot(enabled, channel)

    pygame.quit()


if __name__ == "__main__":
    main()
