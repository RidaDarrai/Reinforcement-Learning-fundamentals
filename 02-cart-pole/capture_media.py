"""
capture_media.py — THE CAMERA (makes the README's GIFs)
========================================================

This file exists so every image in the README can be REGENERATED
instead of being a mystery blob in git:

    python capture_media.py

It reuses play.py's own draw() code — the frames you see in the
GIFs are pixel-identical to the real window — but renders them onto
an off-screen surface instead of opening a window, then stitches the
frames into four GIFs:

    screenshots/human.gif    my own run: balance 60 ticks, then
                             a deliberate mistake (it falls)
    screenshots/watch.gif    the random agent's sad 14-tick ending
    screenshots/live.gif     training flying in the HUD while the
                             scene replays the current best play
    screenshots/trained.gif  the graduate's full 500-tick balance

Cart-pole episodes run 10-500 ticks — way too many frames to show
one-by-one — so every recording runs the episode CHEAPLY first
(storing just numbers), then renders every k-th moment. Rendering
only what we keep means a 500-tick episode costs 60 frames of
memory, not 500.

Needs: q_table.json (python train.py) for trained.gif.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # never open a window

import math
import random

import numpy as np
import pygame
from PIL import Image

import play
from environment import CartPole, LEFT, RIGHT, N_ACTIONS
from train import Trainer

# Where the finished GIFs land (committed — the README links to them).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOT_DIR = os.path.join(BASE_DIR, "screenshots")

# Per-GIF pacing (milliseconds per frame) — the "cinematography".
HUMAN_FRAMES = (800, 70, 2600)     # first hold, per tick, end hold
WATCH_FRAMES = (700, 250, 2200)    # slow ticks: every death is visible
TRAINED_FRAMES = (700, 55, 2800)   # fast ticks: 500 in under a minute
LIVE_FRAMES = (700, 65, 2600)

MAX_FRAMES = 60                    # no GIF frame budget beyond this


def new_ui(mode, **overrides):
    """A ui dict exactly like the one play.run() builds."""
    ui = {"mode": mode, "reward": 0.0, "outcome": None,
          "speed_i": play.SPEED_DEFAULT_I, "notice": None,
          "notice_sub": None, "live": None, "last_action": None}
    ui.update(overrides)
    return ui


def render(env, ui):
    """One frame through play.py's real drawing code (off-screen)."""
    surface = pygame.Surface((play.WIDTH, play.HEIGHT))
    play.draw(surface, env, ui)
    return surface


# ---------------------------------------------------------------
# 1. RUN CHEAP, RENDER SELECTIVELY
# ---------------------------------------------------------------
def run_episode(action_fn, seed=None):
    """Play one episode WITHOUT rendering; keep only numbers.

    Returns a list of snapshots:
        (steps, x, theta, action, reward, outcome)

    The opening state is included (action=None, reward=0) so every
    GIF starts with a clean "before anything happens" frame.
    """
    if seed is not None:
        random.seed(seed)
    env = CartPole()
    env.reset()
    ui = new_ui("HUMAN")           # mode only matters when drawing
    snaps = [(0, env.x, env.theta, None, 0.0, None)]
    while not env.done:
        play.apply_step(env, ui, action_fn(env))
        snaps.append((env.steps, env.x, env.theta, ui["last_action"],
                      ui["reward"], ui["outcome"]))
    return snaps


def decimate(snaps, max_frames=MAX_FRAMES):
    """Thin snapshots to at most max_frames, ALWAYS keeping the
    first (the setup) and the last (the ending banner)."""
    if len(snaps) <= max_frames:
        return list(snaps)
    k = math.ceil((len(snaps) - 1) / (max_frames - 1))
    idxs = list(range(0, len(snaps), k))
    if idxs[-1] != len(snaps) - 1:
        idxs.append(len(snaps) - 1)      # the ending is mandatory
    return [snaps[i] for i in idxs]


def render_snaps(mode, snaps):
    """Turn kept snapshots into frames by poking the numbers straight
    into an environment — the same trick LIVE mode's demo uses."""
    env = CartPole()
    ui = new_ui(mode)
    frames = []
    for steps, x, theta, action, reward, outcome in snaps:
        env.x, env.theta, env.steps = x, theta, steps
        ui["reward"] = reward
        ui["last_action"] = action
        ui["outcome"] = outcome
        frames.append(render(env, ui))
    return frames


def build_durations(n, pacing):
    """first_hold, per-tick, end_hold — the end hold lets the
    outcome banner sit on screen long enough to read."""
    first_ms, step_ms, end_ms = pacing
    if n <= 1:
        return [first_ms]
    if n == 2:
        return [first_ms, end_ms]
    return [first_ms] + [step_ms] * (n - 2) + [end_ms]


def save_gif(frames, durations, path):
    """Stitch surfaces into one GIF with a SHARED palette (a palette
    per frame would flicker). disposal=2 = every frame repaints fully,
    so the moving cart never leaves ghosts."""
    rgbs = [Image.fromarray(
        np.transpose(pygame.surfarray.array3d(s), (1, 0, 2)))
        for s in frames]

    # Palette from a contact sheet of first + middle + last frame,
    # so the banner colors are included too.
    w, h = rgbs[0].size
    picks = [rgbs[0], rgbs[len(rgbs) // 2], rgbs[-1]]
    sheet = Image.new("RGB", (w * 3, h))
    for i, img in enumerate(picks):
        sheet.paste(img, (i * w, 0))
    palette = sheet.convert("P", palette=Image.ADAPTIVE, colors=256)

    quantized = [img.quantize(palette=palette) for img in rgbs]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:],
        duration=durations, loop=0, optimize=True, disposal=2,
    )
    kb = os.path.getsize(path) / 1024
    print(f"  saved {os.path.basename(path):<16} "
          f"{len(frames):>3} frames  {kb:7.0f} KB")


def record_simple(mode, action_fn, gif_name, pacing, seed=None):
    """Shared path for HUMAN / WATCH / TRAINED: run, thin, render.
    Returns (ticks, frames_shown) so callers can report honestly."""
    raw = run_episode(action_fn, seed=seed)
    snaps = decimate(raw)
    frames = render_snaps(mode, snaps)
    save_gif(frames, build_durations(len(frames), pacing),
             os.path.join(SHOT_DIR, gif_name))
    return len(raw) - 1, len(frames)


# ---------------------------------------------------------------
# 2. LIVE: training flies while the scene replays the best play
# ---------------------------------------------------------------
def record_live():
    """One short LIVE session, frame by frame — the same choreography
    play.run() performs, driven by the camera instead of a clock:

        every frame  -> advance training, update the HUD numbers
        every frame  -> take ONE step of the demo replay
        demo finished -> hold (outcome banner), then rebuild it
                         with whatever the brain has learned by then
    """
    trainer = Trainer(quiet=True)          # the real 15,000-ep session
    env = CartPole()                       # the world the demo pokes
    ui = new_ui("LIVE", speed_i=5)         # HUD claims ~500 ep/s

    EPS_PER_FRAME = 35          # 45 frames * 35 ≈ mid-session stop
    TOTAL_FRAMES = 45
    HOLD_FRAMES = 6             # linger on each demo's ending

    # Let the brain get past its very first clueless steps before
    # anything is filmed — frame 1 should already show a real attempt.
    trainer.run_episodes(300)
    demo_states, demo_reason = play.build_demo(trainer.agent)
    demo_i, hold_left = 0, 0

    frames = []
    for _ in range(TOTAL_FRAMES):
        trainer.run_episodes(EPS_PER_FRAME)
        ui["live"] = {
            "episode": trainer.episode, "total": trainer.episodes,
            "stage": trainer.stage, "epsilon": trainer.agent.epsilon,
            "win": trainer.win_rate(),
            "best": trainer.best_score[0] if trainer.best_q else 0.0,
        }

        if hold_left > 0:
            # The ending is on screen; when the hold expires, rebuild
            # the demo with whatever the brain knows NOW.
            hold_left -= 1
            if hold_left == 0:
                demo_states, demo_reason = play.build_demo(trainer.agent)
                demo_i = 0
                ui["outcome"] = None
        else:
            tick, x, theta, action = demo_states[demo_i]
            env.x, env.theta, env.steps = x, theta, tick
            ui["reward"] = float(tick)           # reward == steps, always
            ui["last_action"] = action
            demo_i += 1
            if demo_i >= len(demo_states):
                ui["outcome"] = demo_reason      # show WHY it ended
                hold_left = HOLD_FRAMES

        frames.append(render(env, ui))

    # End on something readable: the final frame must show an ending,
    # not a demo cut off mid-flight.
    if ui["outcome"] is None:
        tick, x, theta, action = demo_states[-1]
        env.x, env.theta, env.steps = x, theta, tick
        ui["last_action"] = action
        ui["outcome"] = demo_reason
        frames[-1] = render(env, ui)

    save_gif(frames, build_durations(len(frames), LIVE_FRAMES),
             os.path.join(SHOT_DIR, "live.gif"))


# ---------------------------------------------------------------
# 3. THE FOUR RECORDINGS
# ---------------------------------------------------------------
def human_action(env):
    """My own rule of thumb while it lasts — push the way the pole
    leans (run under the falling tip) — then, on purpose, the wrong
    way, so the GIF ends with an honest fall."""
    if env.steps < 60:
        return RIGHT if env.theta > 0 else LEFT
    return LEFT


def main():
    pygame.init()
    play._FONTS.clear()          # a stale cache would draw nothing
    os.makedirs(SHOT_DIR, exist_ok=True)
    print("recording...")

    # ---- 1. HUMAN: balance by hand, then lose on purpose ----------
    # sign(theta) bang-bang survives all 60 ticks on every seed I
    # tried (the physics really are that forgiving to a simple
    # rule); the constant LEFT pushes afterwards guarantee the fall.
    ticks, shown = record_simple("HUMAN", human_action, "human.gif",
                                 HUMAN_FRAMES, seed=7)
    print(f"    human:    fell at tick {ticks} (shown: {shown} frames)")

    # ---- 2. WATCH: the random agent (seeded = reproducible) ------
    # Seed 30 = a typical random disaster: 14 ticks, then dead.
    ticks, shown = record_simple(
        "WATCH", lambda env: random.randrange(N_ACTIONS),
        "watch.gif", WATCH_FRAMES, seed=30)
    print(f"    watch:    died after {ticks} ticks (shown: {shown})")

    # ---- 3. LIVE: watching the brain fill itself ------------------
    record_live()
    print("    live:     45 frames of training + demo replay")

    # ---- 4. TRAINED: the graduate's full 500-tick balance --------
    trained = play.load_trained_agent()
    if trained is None:
        print("  trained.gif skipped — run: python train.py")
        return
    ticks, shown = record_simple(
        "TRAINED",
        lambda env: trained.choose_action(env.state, explore=False),
        "trained.gif", TRAINED_FRAMES, seed=9)
    print(f"    trained:  survived {ticks} of 500 ticks "
          f"(shown: {shown} frames)")


if __name__ == "__main__":
    main()
