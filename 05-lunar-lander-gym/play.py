"""
play.py — THE COCKPIT (step 4 of the lunar-lander-gym project)
==============================================================

The window where you watch the graduate fly the REAL world. The
landings are the same problem class as 04's, but the picture now
comes from gym's own renderer — Box2D-drawn frames piped into our
pygame window — and our HUD lives on TOP of it, same habit as 04:

    state, reward flow in -> panels + text out

HOW IT RUNS (two keys are the whole manual):
    R   restart the current attempt (fresh spawn)
    S   toggle sample mode  GREEDY <-> SAMPLED
        GREEDY  = act_greedy: argmax of the policy — the exam
                  mode, deterministic, what "can it land?" means
        SAMPLED = act(): the dice the TRAINING policy rolls (the
                  reference's own test.py flys sampled) — not the
                  graduate's best self, just its honest shape
    V   open visualize.py's charts (own process, like 04)
    ESC close
(no HUMAN mode this time: 04's hands-on cockpit was OUR hand-built
world's toy; here the world is gym's and the brain is the show.)

THE BRAIN SUPPLY (the 04 lesson, kept):
    no actor_critic_model.pt? A quick course (a small Trainer run,
    ~a minute) is flown RIGHT HERE into quick_course_* files —
    never the official run's artifacts. Corrupt/half-written
    files (training can be mid-write) degrade to the same path.

EVERYTHING TESTABLE STAYS FRAMELESS:
    load_trained_agent / quick_course / hud lines / run_episode
    all work with pygame on the dummy video driver (the test
    suite sets SDL_VIDEODRIVER=dummy) — no test ever opens a
    real window.

Run it:

    python play.py
"""

# Every import earns its comment (repo rule: every file explains itself).
import os          # artifact paths anchored next to THIS file
import subprocess  # visualize.py opens in its OWN process (matplotlib
                   # and pygame fight over the window system)
import sys         # the interpreter path for that subprocess

import pygame      # our surface: gym's frames blit under the HUD

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

# The view: gym's frames are 600x400 (its own constants); we show
# them 1:1, with the HUD strip to the right like 04's window.
FRAME_W, FRAME_H = 600, 400
HUD_W = 200
WINDOW_SIZE = (FRAME_W + HUD_W, FRAME_H)
FPS = 30
# Hot keys:
KEY_RESTART = pygame.K_r
KEY_MODE    = pygame.K_s     # greedy <-> sampled
KEY_CHARTS  = pygame.K_v
KEY_QUIT    = pygame.K_ESCAPE

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
        frames   — list of pygame-ready numpy frames (gym's view)
        hud_rows — dicts holding what HUD_LINES render from
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


# ---------------------------------------------------------------
# THE WINDOW (the only pygame-drawing part; tests use the pieces)
# ---------------------------------------------------------------
def main():
    pygame.init()
    pygame.display.set_caption("05 lunar-lander-gym - TRAINED replay")
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 13)

    agent = ensure_trained()
    mode_sample = False       # the S toggle: False = GREEDY

    episode_seed = 0
    frames, rows, outcome, ret, ticks = run_episode(
        agent, seed=episode_seed, sample=mode_sample)
    frame_i = 0
    attempt = 1

    running = True
    while running:
        # --- fresh frames when an episode finished last frame ---
        if frame_i >= len(frames):
            episode_seed += 1
            frames, rows, outcome, ret, ticks = run_episode(
                agent, seed=episode_seed, sample=mode_sample)
            attempt += 1
            frame_i = 0

        # --- draw gym's world, then our HUD over it -------------
        screen.fill((12, 14, 18))
        blit = load_frames_for_pygame([frames[frame_i]])[0]
        screen.blit(blit, (0, 0))
        row = dict(rows[frame_i], attempt=attempt)
        for y_line, text in enumerate(hud_lines(row)):
            img = font.render(text, True, (200, 205, 215))
            screen.blit(img, (FRAME_W + 8, 10 + y_line * 16))
        pygame.display.flip()

        # --- events ---------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == KEY_QUIT:
                    running = False
                elif event.key == KEY_RESTART:
                    frames, rows, outcome, ret, ticks = run_episode(
                        agent, seed=episode_seed, sample=mode_sample)
                    frame_i = 0
                elif event.key == KEY_MODE:
                    mode_sample = not mode_sample
                    frames, rows, outcome, ret, ticks = run_episode(
                        agent, seed=episode_seed, sample=mode_sample)
                    frame_i = 0
                elif event.key == KEY_CHARTS:
                    subprocess.Popen([sys.executable, "visualize.py"],
                                     cwd=HERE)

        frame_i += 1
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
