"""
test_play.py — THE COCKPIT'S CONTRACT, pinned by tests
=======================================================

Everything play.py can do without a real window gets pinned here
(pygame runs on the DUMMY video driver):
    - brain supply: missing/corrupt -> None, real -> ActorCritic
    - the quick course: small dials, OWN artifact files
    - one collected frameless episode: frames == ticks, rows
      contract, GREEDY/SAMPLED wording, banner names the ending
    - the HUD text grid: exact formats for a known row, and the
      '?missing' scream if a key ever slips
    - gym's frames -> pygame surfaces (the blit bridge)

    python test_play.py
"""
import os
import sys
import tempfile

# The whole suite runs FRAMELESS - pygame never sees a real window.
os.environ["SDL_VIDEODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from play import (
    load_trained_agent, quick_course, run_episode, hud_lines,
    load_frames_for_pygame, HUD_LINES,
    KEY_RESTART, KEY_MODE, KEY_CHARTS, KEY_QUIT,
    QUICK_EPISODES,
)
from agent import ActorCritic


def test_keys_are_named_constants():
    """The manual: R restart, S greedy/sampled, V charts, ESC out —
    as named module constants, not magic numbers in the loop."""
    assert KEY_RESTART is pygame.K_r
    assert KEY_MODE is pygame.K_s
    assert KEY_CHARTS is pygame.K_v
    assert KEY_QUIT is pygame.K_ESCAPE
    assert QUICK_EPISODES < 600, "the quick course stays quick"
    print("  keys: R/S/V/ESC are consts; quick course stays quick")


def test_load_trained_agent_paths():
    """Missing file -> None; corrupt file -> None; real file ->
    an ActorCritic (play.py degrades gracefully, 04's contract)."""
    assert load_trained_agent("nope_nowhere.pt") is None
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "corrupt.pt")
        with open(bad, "wb") as f:
            f.write(b"this is not a torch checkpoint")
        assert load_trained_agent(bad) is None
        brain = ActorCritic(seed=9)
        real = os.path.join(tmp, "real.pt")
        brain.save(real)
        loaded = load_trained_agent(real)
        assert isinstance(loaded, ActorCritic)
        assert loaded.hidden == 128
    print("  brain supply: missing/corrupt -> None, real -> brain")


def test_quick_course_small_and_separate():
    """The quick course: small dial set, exams scaled, and BOTH
    artifacts landing in the rediristed tmp files (never the
    official run's paths)."""
    with tempfile.TemporaryDirectory() as tmp:
        log = os.path.join(tmp, "qc_log.json")
        model = os.path.join(tmp, "qc_model.pt")
        policy = quick_course(episodes=6, log_path=log,
                              model_path=model)
        assert isinstance(policy, ActorCritic)
        assert os.path.exists(log) and os.path.exists(model)
        import json
        with open(log, encoding="utf-8") as f:
            diary = json.load(f)
        assert diary["config"]["episodes"] == 6
        assert diary["config"]["exam_episodes"] == 4
    print("  quick course: 6 dials shrunk, artifacts in tmp only")
    return True


def test_run_episode_collects_everything():
    """One GREEDY episode: frame count == tick count, the rows'
    contract (tick corridor, mode wording, legs string), banner
    names the ending, and the world ends."""
    brain = ActorCritic(seed=2)
    frames, rows, outcome, ret, ticks = run_episode(
        brain, seed=1, sample=False)
    assert len(frames) == ticks == len(rows)
    assert outcome in ("landed", "crashed", "out_of_bounds", "timeout")
    assert rows[0]["tick"] == 1
    assert rows[-1]["tick"] == ticks
    assert all(r["mode"] == "GREEDY" for r in rows)
    assert rows[-1]["banner"].endswith("!"), \
        f"banner must name the ending: {rows[-1]['banner']}"
    assert rows[-1]["banner"] == f"{outcome}!"
    # a sampled twin (same seed) differs at least SOMEWHERE:
    _, rows2, outcome2, _, _ = run_episode(brain, seed=1, sample=True)
    assert all(r["mode"] == "SAMPLED" for r in rows2)
    print(f"  frameless episode: {ticks} frames == ticks, "
          f"outcome {outcome}, banners + modes honest")


def test_hud_lines_wording():
    """The HUD's exact formats for one known row; a missing key
    screams '?missing'. """
    row = {"attempt": 3, "tick": 17, "return": -42.25,
           "mode": "GREEDY", "x": -0.2, "y": 0.5, "vx": 0.1,
           "vy": -0.05, "tilt": 12.3456, "legs": "L1/R0",
           "banner": ""}
    lines = hud_lines(row)
    assert lines[0] == "attempt #3"
    assert lines[1] == "tick 17"
    assert lines[2].startswith("return "), lines[2]
    assert "-42.3" in lines[2] or "-42.2" in lines[2], lines[2]
    assert "GREEDY" in lines[3]
    assert lines[6] == "tilt   +12.3 deg", lines[6]
    assert lines[7] == "legs L1/R0"
    # a row missing keys still renders, and screams loudly:
    lines_broken = hud_lines({"tick": 1})
    assert any("?missing" in line for line in lines_broken), \
        lines_broken
    print("  hud lines: formats pinned; missing keys scream")


def test_frames_to_pygame_bridge():
    """gym's numpy frame -> a pygame surface without errors."""
    brain = ActorCritic(seed=2)
    frames, *_ = run_episode(brain, seed=4, sample=False)
    surface = load_frames_for_pygame(frames[:2])[0]
    assert surface.get_size() == (600, 400), surface.get_size()
    print("  frames->pygame: 600x400 surfaces, two blits, no crash")


if __name__ == "__main__":
    test_keys_are_named_constants()
    test_load_trained_agent_paths()
    test_quick_course_small_and_separate()
    test_run_episode_collects_everything()
    test_hud_lines_wording()
    test_frames_to_pygame_bridge()
    print()
    print("All 6 cockpit checks verified.")
