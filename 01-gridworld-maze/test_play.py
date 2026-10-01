"""
test_play.py — headless checks for the window (step 4)
=======================================================

Runs with SDL's "dummy" video driver — no real window pops up, but
every drawing code path still executes on an off-screen surface.

Run with:  python test_play.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # MUST precede pygame import

import tempfile

import numpy as np
import pygame

import play
from agent import QAgent
from environment import GridWorld


def _ui(mode="HUMAN", **overrides):
    """A ui dict shaped exactly like the one run() builds."""
    ui = {"mode": mode, "reward": 0.0, "outcome": None,
          "speed_i": play.SPEED_DEFAULT_I, "notice": None}
    ui.update(overrides)
    return ui


# ---------------------------------------------------------------
# MODE CYCLING
# ---------------------------------------------------------------
def test_mode_cycle():
    """SPACE walks HUMAN -> WATCH -> TRAINED -> back to HUMAN."""
    assert play.next_mode("HUMAN") == "WATCH"
    assert play.next_mode("WATCH") == "TRAINED"
    assert play.next_mode("TRAINED") == "HUMAN"
    print("  HUMAN -> WATCH -> TRAINED -> HUMAN")


# ---------------------------------------------------------------
# LOADING THE GRADUATE
# ---------------------------------------------------------------
def test_load_missing_file_is_none():
    """No q_table.json yet -> None (window shows a hint, no crash)."""
    missing = os.path.join(tempfile.gettempdir(), "definitely_not_here.json")
    assert play.load_trained_agent(missing) is None
    print("  missing file -> None")


def test_load_corrupt_file_is_none():
    """A half-written file -> None instead of an ugly traceback."""
    path = os.path.join(tempfile.gettempdir(), "corrupt_q.json")
    with open(path, "w") as f:
        f.write("{not valid json")
    try:
        assert play.load_trained_agent(path) is None
    finally:
        os.remove(path)
    print("  corrupt file -> None")


def test_load_round_trip():
    """A saved brain comes back identical through the play.py loader."""
    agent = QAgent(n_actions=4)
    agent.get_q((2, 3))[1] = 6.5
    path = os.path.join(tempfile.gettempdir(), "q_roundtrip.json")
    try:
        agent.save(path)
        loaded = play.load_trained_agent(path)
        assert loaded is not None
        assert loaded.q_table == agent.q_table
    finally:
        os.remove(path)
    print("  saved brain loads back with identical Q-table")


# ---------------------------------------------------------------
# HUD NEVER CLIPS (worst-case numbers in every mode)
# ---------------------------------------------------------------
def test_hud_fits_in_every_mode():
    """Draw onto a surface 100px wider than the window. Anything
    painted past the window's right edge would be invisible to the
    user — i.e. clipped text — so the extra strip must stay empty."""
    wide = pygame.Surface((play.WIDTH + 100, play.HEIGHT))
    bg = np.array(play.C_BG)

    cases = []
    for mode in play.MODES:
        # worst case: max step counter, most negative reward, fastest
        # speed (longest multiplier text), banner up, notice up...
        cases.append((mode, _ui(mode, reward=-10.0)))
    cases[1] = ("WATCH", _ui("WATCH", reward=-10.0, speed_i=5))
    cases[2] = ("TRAINED", _ui("TRAINED", reward=-10.0, speed_i=5))
    # ...and every banner state
    cases += [
        ("HUMAN", _ui("HUMAN", outcome="goal")),
        ("WATCH", _ui("WATCH", outcome="pit")),
        ("TRAINED", _ui("TRAINED", outcome="timeout")),
        ("TRAINED", _ui("TRAINED", notice="No trained agent - run: python train.py")),
    ]

    for mode, ui in cases:
        env = GridWorld()
        env.reset()
        env.steps = 100                      # worst-case counter text
        play.draw(wide, env, ui)
        strip = pygame.surfarray.array3d(wide)[play.WIDTH:, :, :]
        assert np.all(strip == bg), \
            f"HUD/banner clipped past window edge in {ui}"

    print(f"  {len(cases)} worst-case frames: nothing clipped")


# ---------------------------------------------------------------
# THE GRADUATE PLAYS THROUGH THE WINDOW'S PLUMBING
# ---------------------------------------------------------------
def test_trained_agent_wins_via_apply_step():
    """Train briefly, load through play.load_trained_agent, then
    drive a full greedy episode through apply_step() — the exact
    code path TRAINED mode uses in the real window."""
    import random
    from train import train

    random.seed(123)
    agent, _ = train(episodes=600, quiet=True)
    path = os.path.join(tempfile.gettempdir(), "q_play_test.json")
    try:
        agent.save(path)
        loaded = play.load_trained_agent(path)
        assert loaded is not None

        env = GridWorld()
        ui = _ui("TRAINED")
        play.restart(env, ui)
        while not env.done:
            action = loaded.choose_action(env.agent_pos, explore=False)
            play.apply_step(env, ui, action)

        assert ui["outcome"] == "goal", f"lost: {ui['outcome']}"
        assert ui["reward"] >= 8.5, f"clumsy path: {ui['reward']:+.2f}"
        print(f"  graduate won through apply_step: {ui['reward']:+.2f} "
              f"in {env.steps} steps")
    finally:
        os.remove(path)


# ---------------------------------------------------------------
# THE WINDOW OPENS AND CLOSES
# ---------------------------------------------------------------
def test_run_opens_and_exits():
    """run() must survive a full open->paint->quit cycle."""
    pygame.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    play.run()
    print("  run() opened and exited cleanly")


if __name__ == "__main__":
    pygame.font.init()          # fonts need this before any test draws
    tests = [
        test_mode_cycle,
        test_load_missing_file_is_none,
        test_load_corrupt_file_is_none,
        test_load_round_trip,
        test_hud_fits_in_every_mode,
        test_trained_agent_wins_via_apply_step,
        test_run_opens_and_exits,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} window checks verified.")
