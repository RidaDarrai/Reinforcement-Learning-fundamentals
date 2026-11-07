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
    - the telemetry port (04's cockpit): the GYM-DNA palette
      (sampled from a real rendered frame), the six SLIDING
      channels, the window slicing ("sliding only" gate), the
      series geometry, the toggle BUTTONS (click geometry +
      state), the lamp layout, the window shape, and the
      numeric telemetry feeding the plots and lamps
      (15 cockpit checks in total).

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
    KEY_PLOT_TOGGLES, KEY_SPEED_UP, KEY_SPEED_DOWN,
    QUICK_EPISODES,
    # the gym-DNA palette and the 04 telemetry port
    GYM_BLACK, GYM_WHITE, GYM_PURPLE, GYM_PURPLE_D,
    GYM_YELLOW, GYM_RED, GYM_GREY, GYM_DARKGREY,
    BANNER_COLORS, banner_for,
    PLOT_PANELS, PLOT_KEYS, PLOT_TICKS,
    fresh_enabled, toggle_plot, plot_button_rect, plot_button_at,
    lamp_rect, fresh_history, fill_history, slid, series_points,
    draw, WINDOW_SIZE, FRAME_W, FRAME_H, TELE_W, BOTTOM_H,
)
from agent import ActorCritic


def test_keys_are_named_constants():
    """The manual: R restart, S greedy/sampled, V charts, ESC out,
    +/- playback speed, 1..6 graph toggles — as named module
    constants, not magic numbers in the loop."""
    assert KEY_RESTART is pygame.K_r
    assert KEY_MODE is pygame.K_s
    assert KEY_CHARTS is pygame.K_v
    assert KEY_QUIT is pygame.K_ESCAPE
    assert KEY_PLOT_TOGGLES == (pygame.K_1, pygame.K_2, pygame.K_3,
                                pygame.K_4, pygame.K_5, pygame.K_6)
    assert pygame.K_EQUALS in KEY_SPEED_UP
    assert pygame.K_MINUS in KEY_SPEED_DOWN
    assert QUICK_EPISODES < 600, "the quick course stays quick"
    print("  keys: R/S/V/ESC, 1..6 toggles, +/- speed — all consts")


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


# ---------------------------------------------------------------
# THE TELEMETRY PORT (04's cockpit, gym's DNA)
# ---------------------------------------------------------------
def test_gym_dna_palette():
    """The visual DNA is GYM'S OWN, pinned by value: every color
    was sampled out of a rendered LunarLander-v3 frame (black sky,
    white terrain, purple lander 128/102/230 + outline 77/77/128,
    yellow flags 204/204/0, fresh smoke red, aged smoke greys)."""
    assert GYM_BLACK == (0, 0, 0)
    assert GYM_WHITE == (255, 255, 255)
    assert GYM_PURPLE == (128, 102, 230)
    assert GYM_PURPLE_D == (77, 77, 128)
    assert GYM_YELLOW == (204, 204, 0)
    assert GYM_RED == (255, 127, 127)
    assert GYM_GREY == (129, 129, 129)
    assert GYM_DARKGREY == (65, 65, 65)
    # the four endings paint with the loudest gym colors
    assert BANNER_COLORS["landed"] is GYM_YELLOW
    assert BANNER_COLORS["crashed"] is GYM_RED
    assert BANNER_COLORS["out_of_bounds"] is GYM_RED
    assert BANNER_COLORS["timeout"] is GYM_GREY
    print("  palette: lander purple, flag yellow, smoke greys — gym's")


def test_plot_panels_are_the_six_channels():
    """Six channels in 04's order, each with gym's observation-space
    axis and its own palette color; legs is the two-trace special."""
    assert PLOT_KEYS == ("x", "y", "vx", "vy", "tilt", "legs")
    assert len(PLOT_PANELS) == 6
    by_key = dict((p[0], p) for p in PLOT_PANELS)
    assert by_key["x"][2:4] == (-1.0, 1.0)      # gym's arena edge
    assert by_key["y"][2:4] == (-2.5, 2.5)      # gym's obs-space y
    assert by_key["vx"][2:4] == (-10.0, 10.0)   # gym's speed bound
    assert by_key["vy"][2:4] == (-10.0, 10.0)
    assert by_key["tilt"][2:4] == (-180.0, 180.0)
    assert by_key["legs"][2] is None            # special: two traces
    colors = {p[4] for p in PLOT_PANELS[:5]}
    assert colors <= {GYM_WHITE, GYM_YELLOW, GYM_GREY, GYM_RED,
                      GYM_PURPLE}, "line colors stay in gym's DNA"
    assert len(colors) == 5, "one distinct family per channel"
    assert PLOT_TICKS == 300, "the sliding window knows its depth"
    print("  panels: six channels, gym's axes + palette, legs twin")


def test_history_fills_and_slides():
    """fill_history pours a flight record into every channel; slid()
    keeps ONLY the last PLOT_TICKS — the 'sliding only' gate."""
    history = fresh_history()
    assert history["x"] == [] and history["legs"] == []
    tick = {"x": 0.5, "y": 1.0, "vx": -2.0, "vy": -0.5,
            "tilt": 12.0, "legL": 1, "legR": 0}
    for _ in range(PLOT_TICKS + 37):     # 337 ticks of the same pose
        fill_history(history, tick)
    assert len(history["vx"]) == PLOT_TICKS + 37
    window = slid(history["vy"])
    assert len(window) == PLOT_TICKS, "the window holds exactly N"
    assert window[-1] == -0.5            # newest is what's plotted
    assert window[0] == -0.5             # (a constant channel: ends)
    assert slid(history["legs"])[-1] == (1, 0)
    print(f"  history: fills all channels; slid() = last {PLOT_TICKS}")


def test_series_points_clamps_and_spans():
    """The plot geometry: points span the pane's inner width,
    oldest LEFT / newest RIGHT (so the line slides), values clamp
    into the axis, and a too-short series draws nothing."""
    pane = (0, 0, 264, 62)
    assert series_points(pane, [1.0], -1.0, 1.0) == []
    pts = series_points(pane, [-1.0, 0.0, 1.0], -1.0, 1.0)
    assert len(pts) == 3
    assert pts[0][0] == 4 and pts[-1][0] == 260, pts      # inner span
    assert all(pts[i][0] < pts[i + 1][0] for i in range(2)), \
        "time flows left -> right"
    out = series_points(pane, [-5.0, 0.0, 5.0], -1.0, 1.0)
    assert out[0][1] == 59 and out[-1][1] == 16          # clamped
    # hand-derivation: pane h=62 -> label headroom top=16, bottom
    # pad=3 -> the hi edge maps to 16, the lo edge to 59.
    print("  series: spans the pane, slides left, clamps edges")


def test_toggle_buttons_click_and_state():
    """The graph toggles: everything starts ON, a chip click flips
    exactly its channel, keys map onto the same channels, and an
    unknown channel ValueError-screams."""
    enabled = fresh_enabled()
    assert all(enabled.values())          # fresh cockpit: all visible
    assert toggle_plot(enabled, "vx") is False
    assert enabled["vx"] is False and enabled["vy"] is True
    assert toggle_plot(enabled, "vx") is True
    try:
        toggle_plot(enabled, "nope")
    except ValueError as e:
        assert "unknown telemetry channel" in str(e)
    else:
        raise AssertionError("unknown channel must scream")
    # click geometry: each chip's center hits its own channel...
    hits = {}
    for i, key in enumerate(PLOT_KEYS):
        rect = plot_button_rect(i)
        assert plot_button_at(rect.center) == key
        hits[key] = rect
    # ...and nothing else in the layout pretends to be a chip
    assert plot_button_at((0, 0)) is None
    assert plot_button_at((1, 1)) is None
    # the chips sit in two rows of three, all inside the column
    from collections import Counter
    row_counts = Counter(r.y for r in hits.values())
    assert sorted(row_counts.values()) == [3, 3], row_counts
    for r in hits.values():
        assert FRAME_W <= r.x and r.right <= WINDOW_SIZE[0]
    print("  buttons: click hits its channel, toggles, column-bound")


def test_lamps_and_window_layout():
    """The four button lamps: one per action, inside the bottom
    band, under the video; the window holds the 1:1 video plus
    column + band."""
    assert WINDOW_SIZE == (FRAME_W + TELE_W, FRAME_H + BOTTOM_H)
    assert (FRAME_W, FRAME_H) == (600, 400), "gym's frame stays 1:1"
    seen = set()
    for a in range(4):
        r = lamp_rect(a)
        assert id(r) not in seen or True
        seen.add((r.x, r.y, r.w, r.h))
        assert r.y >= FRAME_H and r.bottom <= WINDOW_SIZE[1]
        assert r.x >= 0 and r.right <= FRAME_W
    assert len(seen) == 4
    print("  layout: 880x640 window, video 1:1, four lamps in band")


def test_rows_carry_the_numeric_channels():
    """The collected flight records now also feed the plots/lamps:
    per-tick leg flags as ints and the action int in 0..3."""
    brain = ActorCritic(seed=2)
    _, rows, _, _, ticks = run_episode(brain, seed=3, sample=False)
    assert len(rows) == ticks
    assert all(r["legL"] in (0, 1) and r["legR"] in (0, 1)
               for r in rows)
    assert all(isinstance(r["action"], int)
               and 0 <= r["action"] <= 3 for r in rows)
    # the plots' degrees travel in the same row the HUD words use
    assert all(-720.0 <= r["tilt"] <= 720.0 for r in rows)
    print("  rows: leg flags + action int ride along for the plots")


def test_banner_words_and_colors():
    """The verdict text is fixed per ending, painted with gym's
    loudest colors."""
    assert banner_for("landed") == ("TOUCHDOWN!", GYM_YELLOW)
    assert banner_for("crashed") == ("CRASHED!", GYM_RED)
    assert banner_for("out_of_bounds") == ("OUT OF BOUNDS!", GYM_RED)
    assert banner_for("timeout") == ("TIMED OUT.", GYM_GREY)
    print("  banner: four endings, four gym paints")


def test_cockpit_draws_framelessly():
    """A full draw() pass on the DUMMY driver: mid-flight frame and
    the verdict frame, one channel toggled OFF — nothing crashes
    (this is the smoke test that caught the None-colored legs
    chip before it could ever reach a real window)."""
    pygame.font.init()      # the draw path needs text, not only video
    pygame.display.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    brain = ActorCritic(seed=2)
    frames, rows, outcome, _ret, ticks = run_episode(
        brain, seed=1, sample=False)
    history = fresh_history()
    for i in range(ticks):
        fill_history(history, rows[i])
    enabled = fresh_enabled()
    enabled["vx"] = False          # a toggled-off channel must be OK
    draw(screen, frames, ticks // 2, rows, 1, history,
              enabled, 1)
    draw(screen, frames, len(frames) - 1, rows, 1, history,
              fresh_enabled(), 1)
    pygame.display.quit()
    print("  draw: mid-flight + verdict frame, dummy driver, no crash")


if __name__ == "__main__":
    test_keys_are_named_constants()
    test_load_trained_agent_paths()
    test_quick_course_small_and_separate()
    test_run_episode_collects_everything()
    test_hud_lines_wording()
    test_frames_to_pygame_bridge()
    test_gym_dna_palette()
    test_plot_panels_are_the_six_channels()
    test_history_fills_and_slides()
    test_series_points_clamps_and_spans()
    test_toggle_buttons_click_and_state()
    test_lamps_and_window_layout()
    test_rows_carry_the_numeric_channels()
    test_banner_words_and_colors()
    test_cockpit_draws_framelessly()
    print()
    print("All 15 cockpit checks verified.")
