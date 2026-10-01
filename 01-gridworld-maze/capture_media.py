"""
capture_media.py — THE CAMERA (makes the README's GIFs)
========================================================

This file exists so every image in the README can be REGENERATED
instead of being a mystery blob in git:

    python capture_media.py

It reuses play.py's own draw() code — the frames you see in the
GIFs are pixel-identical to the real window — but renders them onto
an off-screen surface instead of opening a window, then stitches the
frames into three GIFs:

    screenshots/human.gif    me playing (one honest wall-bump included)
    screenshots/watch.gif    the random agent's sad ending
    screenshots/trained.gif  the trained agent's clean win

Needs: q_table.json (python train.py) for the trained GIF.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # never open a window

import random

import numpy as np
import pygame
from PIL import Image

import play
from environment import GridWorld, UP, DOWN, LEFT, RIGHT, N_ACTIONS

# Where the finished GIFs land (committed — the README links to them).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOT_DIR = os.path.join(BASE_DIR, "screenshots")


def new_ui(mode):
    """A ui dict exactly like the one play.run() builds."""
    return {"mode": mode, "reward": 0.0, "outcome": None,
            "speed_i": play.SPEED_DEFAULT_I, "notice": None}


def render(env, ui):
    """One frame through play.py's real drawing code (off-screen)."""
    surface = pygame.Surface((play.WIDTH, play.HEIGHT))
    play.draw(surface, env, ui)
    return surface


def record(mode, step_fn, first_hold_ms=800, step_ms=350,
           end_hold_ms=2400, seed=None, agent=None):
    """Play one episode and return (frames, durations).

    step_fn(env) -> action | None  picks each move; returning None
    (or finishing the episode) ends the recording.
    """
    if seed is not None:
        random.seed(seed)
    env = GridWorld()
    env.reset()
    ui = new_ui(mode)

    frames = [render(env, ui)]
    while not env.done:
        action = step_fn(env)
        if action is None:
            break
        play.apply_step(env, ui, action)
        frames.append(render(env, ui))

    durations = [first_hold_ms] + [step_ms] * (len(frames) - 2) \
        + [end_hold_ms] if len(frames) >= 2 else [first_hold_ms]
    return frames, durations


def save_gif(frames, durations, path):
    """Stitch surfaces into one GIF with a SHARED palette (a palette
    per frame would flicker). disposal=2 = every frame repaints fully,
    so the moving agent never leaves ghosts."""
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
    print(f"  saved {os.path.basename(path):<12} "
          f"{len(frames):>3} frames  {kb:7.0f} KB")


def main():
    pygame.init()
    os.makedirs(SHOT_DIR, exist_ok=True)

    # ---- 1. HUMAN: my own run, mistakes included -------------------
    # One deliberate bump into the wall at (1,1): bumps still cost a
    # step (-0.1) — the HUD's dipping reward shows it.
    path = [RIGHT, DOWN, RIGHT, DOWN, RIGHT, RIGHT, DOWN, DOWN,
            RIGHT, RIGHT, RIGHT, DOWN, DOWN]

    def human_steps(env, remaining=iter(path)):
        return next(remaining, None)

    print("recording...")
    frames, durs = record("HUMAN", human_steps, step_ms=420)
    save_gif(frames, durs, os.path.join(SHOT_DIR, "human.gif"))

    # ---- 2. WATCH: the random agent (seeded so it's reproducible) --
    # Seed 60 = a typical random disaster: 69 wandering steps and a
    # fall into the pit. Seeded, so the README GIF never changes.
    frames, durs = record(
        "WATCH", lambda env: random.randrange(N_ACTIONS),
        first_hold_ms=700, step_ms=70, end_hold_ms=2200, seed=60)
    save_gif(frames, durs, os.path.join(SHOT_DIR, "watch.gif"))

    # ---- 3. TRAINED: the graduate, greedy and unbeatable -----------
    trained = play.load_trained_agent()
    if trained is None:
        print("  trained.gif skipped — run: python train.py")
        return

    frames, durs = record(
        "TRAINED",
        lambda env: trained.choose_action(env.agent_pos, explore=False),
        first_hold_ms=700, step_ms=250, end_hold_ms=2600)
    save_gif(frames, durs, os.path.join(SHOT_DIR, "trained.gif"))


if __name__ == "__main__":
    main()
