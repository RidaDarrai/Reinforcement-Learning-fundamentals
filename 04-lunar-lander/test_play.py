"""
test_play.py — is the window honest, even headless?
===================================================

The window has ONE rule it must prove without opening anything:
everything the user sees must come from the world's own calls and
numbers, mapped through testable pure functions. So each test:

- runs pygame with the DUMMY video driver (no window pops up);
- calls the SAME draw/plumbing functions run() uses;
- only touches throwaway temp dirs — never the official run's
  dqn_model.pt / training_log.json (a training run may literally
  be writing them while these tests execute).

Pinned here:
- the mode contract: TWO modes only, SPACE toggles between them;
- the key map: held UP/LEFT/RIGHT -> MAIN/ROT_LEFT/ROT_RIGHT,
  nothing held -> IDLE;
- the world->pixel map: ground at the ground line, ceiling at the
  ceiling line, center at center (the ONE flip, no surprises);
- the telemetry recorder: every plotted channel grows per tick;
- episode plumbing: return accumulates, death reasons recorded,
  apply_step is a no-op guard after done, restart clears;
- HUD text and banners name the death reasons the player must see;
- a FULL FRAME renders headless (flying, finished, and noticed);
- the quick-course fallback trains a tiny diploma into a temp dir.
"""

# Imports, each earning its line (repo rule: comment every file well).
import math        # isclose on the return-accumulation hand check
import os          # tempdir paths + the dummy-driver env var
import tempfile    # throwaway dirs for the quick-course artifacts

# The dummy driver MUST be set before pygame opens any surface.
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame

from agent import DQNAgent
from environment import (
    LunarLander, IDLE, MAIN, ROT_LEFT, ROT_RIGHT,
    WORLD_HALF_WIDTH, CEILING_Y,
)
from play import (
    MODES, next_mode, action_from_held, load_trained_agent, retrain,
    fresh_ui, restart, apply_step, banner_for, hud_lines, draw,
    record_tick, x_to_px, y_to_px, world_to_px,
    X_L_PX, X_R_PX, CEIL_PX, GROUND_Y, WIDTH, HEIGHT,
)


# ---------------------------------------------------------------
# The mode contract
# ---------------------------------------------------------------
def test_modes_are_human_and_trained_only():
    """The window has exactly TWO modes and SPACE both ways between
    them — the flailing WATCH and the in-window LIVE school from
    02/03 do not exist here on purpose."""
    assert MODES == ("HUMAN", "TRAINED")
    assert next_mode("HUMAN") == "TRAINED"
    assert next_mode("TRAINED") == "HUMAN"     # two-way toggle
    print("  modes: HUMAN <-> TRAINED, nothing else")


def test_held_keys_map_to_actions():
    """The HUMAN policy: held UP = MAIN, LEFT/RIGHT = rotation,
    multiple held -> thrust wins (priority order), none -> IDLE."""
    up, left, right = pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT
    assert action_from_held({up: True}) == MAIN
    assert action_from_held({left: True}) == ROT_LEFT
    assert action_from_held({right: True}) == ROT_RIGHT
    assert action_from_held({up: True, left: True}) == MAIN
    assert action_from_held({up: False, left: False,
                             right: False}) == IDLE
    assert action_from_held({}) == IDLE
    print("  key map: UP/LEFT/RIGHT -> buttons, none -> IDLE")


# ---------------------------------------------------------------
# The brain supply — load, corrupt, missing, quick course
# ---------------------------------------------------------------
def test_load_trained_agent_paths():
    """load_trained_agent returns None for missing AND corrupt
    checkpoints (never raises at the window), and a real brain for
    a real checkpoint — whichever path it is handed."""
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "never.pt")
        assert load_trained_agent(path=missing) is None

        garbage = os.path.join(tmp, "garbage.pt")
        with open(garbage, "wb") as f:
            f.write(b"not a torch checkpoint at all")
        assert load_trained_agent(path=garbage) is None

        real = os.path.join(tmp, "real.pt")
        DQNAgent(seed=0).save(real)
        agent = load_trained_agent(path=real)
        assert agent is not None
        assert len(agent.get_q((0.0, 70.0, 0, 0, 0, 0, 0, 0))) == 4
    print("  brain supply: missing/corrupt -> None, real -> agent")


def test_quick_course_smoke():
    """Pressing TRAINED with no diploma trains a small course right
    there, WRITES ONLY ITS OWN FILES (never the official run's
    artifacts), and returns a working 4-button brain."""
    with tempfile.TemporaryDirectory() as tmp:
        diploma = os.path.join(tmp, "quick.pt")
        agent = retrain(episodes=3, diploma_path=diploma)
        assert isinstance(agent, DQNAgent)
        assert os.path.exists(diploma), "quick course must save"
        q = agent.get_q((0.0, 70.0, 0, 0, 0, 0, 0, 0))
        assert len(q) == 4 and all(-1e6 < v < 1e6 for v in q)
    print("  quick course: 3 episodes -> diploma saved into tmp")


# ---------------------------------------------------------------
# The world->pixel map — the ONE flip
# ---------------------------------------------------------------
def test_screen_mapping_endpoints():
    """x=-50 m is the left wall pixel, +50 the right, 0 the middle;
    y=0 sits ON the ground line, y=80 ON the ceiling line, and up
    in the world is UP on the screen (smaller px)."""
    assert x_to_px(-WORLD_HALF_WIDTH) == X_L_PX
    assert x_to_px(WORLD_HALF_WIDTH) == X_R_PX
    assert abs(x_to_px(0.0) - (X_L_PX + X_R_PX) / 2) < 1e-9

    assert y_to_px(0.0) == GROUND_Y
    assert y_to_px(CEILING_Y) == CEIL_PX
    assert y_to_px(40.0) == (GROUND_Y + CEIL_PX) / 2
    assert y_to_px(70.0) < y_to_px(10.0), "y-up must survive"
    print("  mapping: walls/ground/ceiling land on their lines")


def test_world_to_px_composes_the_flip():
    """The composed map: world (0, 0) -> ground center; world up
    moves the pixel up (screen y DOWN, the one documented flip)."""
    x, y = world_to_px(0.0, 0.0)
    assert y == GROUND_Y
    high = world_to_px(0.0, 40.0)[1]
    assert high < y
    assert abs(x - width_mid()) < 1e-9
    print("  composition: ground center + up-is-up")


def width_mid():
    return (x_to_px(-WORLD_HALF_WIDTH) + x_to_px(WORLD_HALF_WIDTH)) / 2


# ---------------------------------------------------------------
# The telemetry recorder
# ---------------------------------------------------------------
def test_record_tick_grows_all_channels():
    """Every plotted channel (x, y, vx, vy, tilt, legs) gets ONE
    sample per tick — the plots can never show a channel the
    plumbing failed to feed."""
    env = LunarLander(seed=3)
    env.reset()
    ui = fresh_ui()
    env.x, env.y, env.vx, env.vy = 5.0, 66.0, 1.0, -2.0
    env.leg_l, env.leg_r = 1, 0
    record_tick(env, ui)
    h = ui["history"]
    assert set(h) == {"x", "y", "vx", "vy", "tilt", "legs"}
    assert h["x"] == [5.0]
    assert h["y"] == [66.0]
    assert h["vx"] == [1.0] and h["vy"] == [-2.0]
    assert abs(h["tilt"][0] - math.degrees(env.theta)) < 1e-9
    assert h["legs"] == [(1, 0)]
    print("  telemetry: six channels, one tick, honest values")


# ---------------------------------------------------------------
# Episode plumbing
# ---------------------------------------------------------------
def test_apply_step_accumulates_and_records():
    """apply_step walks the standard contract: return grows by the
    tick reward, the action is mirrored for the keypad, telemetry
    grows — and after done it REFUSES to step a finished world."""
    env = LunarLander(seed=1)
    env.reset()
    ui = fresh_ui()
    r0 = ui["return"]
    apply_step(env, ui, MAIN)
    assert ui["last_action"] == MAIN
    assert len(ui["history"]["x"]) == 1
    # Twin worlds, same seed: the tick reward is recomputable, so the
    # accumulator must hold it EXACTLY (money accounting, not vibes).
    twin = LunarLander(seed=1)
    twin.reset()
    _, twin_reward, _, _ = twin.step(MAIN)
    assert math.isclose(ui["return"] - r0, twin_reward, abs_tol=1e-9), \
        f"accumulated {ui['return'] - r0} vs world's {twin_reward}"

    # Fly to ANY end (burn burns money; idle falls) — then try again.
    actions = (MAIN, IDLE, ROT_LEFT, ROT_RIGHT)
    i = 0
    while not env.done:
        apply_step(env, ui, actions[i % 4])
        i += 1
    assert ui["outcome"] in ("landed", "crashed", "out_of_bounds",
                             "timeout")
    ticks = len(ui["history"]["x"])
    assert ticks == env.steps

    before = ui["return"]
    apply_step(env, ui, IDLE)          # the no-op guard
    assert ui["return"] == before
    assert len(ui["history"]["x"]) == ticks
    print(f"  plumbing: {ticks} ticks, outcome={ui['outcome']}, "
          f"done-guard holds")


def test_restart_resets_the_attempt():
    """restart() gives a NEW spawn, zeroed return, cleared banner
    and telemetry, counts the attempt — but keeps the mode and
    speed dial untouched."""
    env = LunarLander(seed=5)
    env.reset()
    ui = fresh_ui()
    ui["mode"] = "TRAINED"
    ui["speed_i"] = 2
    apply_step(env, ui, MAIN)
    state_before = env.state
    restart(env, ui)
    assert ui["attempt"] == 1          # fresh_ui started at 0
    assert ui["return"] == 0.0
    assert ui["outcome"] is None
    assert ui["last_action"] is None
    assert all(v == [] for v in ui["history"].values())
    assert env.steps == 0
    assert ui["mode"] == "TRAINED"     # untouched
    assert ui["speed_i"] == 2          # untouched
    assert env.state != state_before   # a fresh scatter (x differs)
    print("  restart: new scatter, clean slate, dials kept")


# ---------------------------------------------------------------
# WHAT THE PLAYER READS — HUD lines and the verdict banner
# ---------------------------------------------------------------
def test_hud_lines_name_the_flight():
    """The HUD text carries the five facts: attempt, tick, speed,
    return, fuel — and the last death reason with its crash type."""
    env = LunarLander(seed=2)
    env.reset()
    ui = fresh_ui()
    lines = hud_lines(env, ui)
    joined = "\n".join(lines)
    assert "ATTEMPT 0" in joined
    assert "TICK 0/" in joined
    assert "RETURN" in joined and "FUEL" in joined
    assert "flying" in joined          # nothing has died yet

    ui["outcome"] = "crashed"
    ui["crash_type"] = "hull"
    joined = "\n".join(hud_lines(env, ui))
    assert "crashed (hull)" in joined
    print("  HUD: attempt/tick/return/fuel + crash flavor")


def test_banner_names_every_death():
    """All four reasons map to sayable, colored verdicts — landed
    green-ish, every crash readable, timeout honest ('no score')."""
    text, color = banner_for("landed", None)
    assert "TOUCHDOWN" in text and color != (255, 95, 85)

    for crash_type, word in (("hull", "BELLY"),
                             ("impact", "IMPACT"),
                             ("missed_pad", "MISSED")):
        text, _ = banner_for("crashed", crash_type)
        assert word in text, f"{crash_type} must be named: {text}"

    text, _ = banner_for("crashed", None)
    assert "CRASH" in text
    text, _ = banner_for("out_of_bounds", None)
    assert "BOUND" in text
    text, _ = banner_for("timeout", None)
    assert "NO SCORE" in text
    print("  banner: landed/crash flavors/OOB/timeout all named")


# ---------------------------------------------------------------
# THE BIG ONE — a full frame renders, headless
# ---------------------------------------------------------------
def test_full_frame_draws_headless():
    """draw() paints a complete cockpit frame on a dummy surface:
    flying, finished (banner up), with a notice, and mid-flame —
    none of it needs a real screen."""
    pygame.init()
    from play import _FONTS
    _FONTS.clear()          # a previous pygame.quit() poisons the cache
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    env = LunarLander(seed=9)
    env.reset()
    ui = fresh_ui()
    for _ in range(40):                # some flight + telemetry
        apply_step(env, ui, (MAIN, ROT_LEFT, IDLE, ROT_RIGHT)[_ % 4])

    draw(screen, env, ui)              # flying frame, no error
    assert screen.get_size() == (WIDTH, HEIGHT)

    # A flame frame: is the plume visible somewhere under the ship?
    env.done = False
    ui["outcome"] = None
    ui["last_action"] = MAIN
    draw(screen, env, ui)

    # A finished frame (banner text must not blow up fonts):
    ui["outcome"] = "crashed"
    ui["crash_type"] = "missed_pad"
    draw(screen, env, ui)

    # A noticed frame (the quick-course warning):
    ui["outcome"] = None
    ui["notice"] = "WARNING: no trained brain!"
    ui["notice_sub"] = "quick course (~1 min) launching now"
    draw(screen, env, ui)

    ui["notice"] = None
    draw(screen, env, ui)              # clean frame again
    pygame.quit()
    print("  full frames: flying / flame / crashed / noticed, headless")


if __name__ == "__main__":
    tests = [
        test_modes_are_human_and_trained_only,
        test_held_keys_map_to_actions,
        test_load_trained_agent_paths,
        test_quick_course_smoke,
        test_screen_mapping_endpoints,
        test_world_to_px_composes_the_flip,
        test_record_tick_grows_all_channels,
        test_apply_step_accumulates_and_records,
        test_restart_resets_the_attempt,
        test_hud_lines_name_the_flight,
        test_banner_names_every_death,
        test_full_frame_draws_headless,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} window checks verified.")
