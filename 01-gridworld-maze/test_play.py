"""
test_play.py — headless checks for the window (step 4+6)
=========================================================

Runs with SDL's "dummy" video driver — no real window pops up, but
every drawing code path still executes on an off-screen surface.

Step 6 added: four modes (incl. LIVE), the N/V/TAB keys, the
auto-training safety net, and the random-maze persistence.

Run with:  python test_play.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # MUST precede pygame import

import json
import sys
import tempfile

import numpy as np
import pygame

import environment
import play
import train
from agent import QAgent
from environment import GridWorld, path_exists


def _ui(mode="HUMAN", **overrides):
    """A ui dict shaped exactly like the one run() builds."""
    ui = {"mode": mode, "reward": 0.0, "outcome": None,
          "speed_i": play.SPEED_DEFAULT_I, "notice": None,
          "notice_sub": None, "maze_name": "classic", "live": None}
    ui.update(overrides)
    return ui


def _live_ui(**overrides):
    """A worst-case LIVE dashboard ui (biggest possible numbers)."""
    return _ui("LIVE",
               live={"episode": 2000, "total": 2000, "stage": 2,
                     "epsilon": 1.0, "win": 100.0},
               speed_i=len(play.LIVE_EPS_PER_SEC) - 1,
               **overrides)


# ---------------------------------------------------------------
# MODE CYCLING
# ---------------------------------------------------------------
def test_mode_cycle():
    """SPACE walks HUMAN -> WATCH -> LIVE -> TRAINED -> back to HUMAN."""
    assert play.next_mode("HUMAN") == "WATCH"
    assert play.next_mode("WATCH") == "LIVE"
    assert play.next_mode("LIVE") == "TRAINED"
    assert play.next_mode("TRAINED") == "HUMAN"
    print("  HUMAN -> WATCH -> LIVE -> TRAINED -> HUMAN")


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
# THE SAFETY NET — forgot to launch training?
# ---------------------------------------------------------------
def test_ensure_trained_warns_and_trains():
    """No brain on disk -> warning banner + training launched for real."""
    tmp = tempfile.mkdtemp(prefix="ensure_")
    old_q, old_sq, old_sl = (play.Q_TABLE_PATH, train.SAVE_Q, train.SAVE_LOG)
    play.Q_TABLE_PATH = os.path.join(tmp, "q_table.json")
    train.SAVE_Q = play.Q_TABLE_PATH
    train.SAVE_LOG = os.path.join(tmp, "training_log.json")
    try:
        ui = _ui("TRAINED")
        agent = play.ensure_trained(ui)
        assert agent is not None, "safety net must return an agent"
        assert "WARNING" in ui["notice"], ui["notice"]
        assert ui["notice_sub"], "warning needs a sub-line"
        assert os.path.exists(play.Q_TABLE_PATH), "brain not saved"
        # Second call: brain exists now -> instant, no banner.
        ui2 = _ui("TRAINED")
        assert play.ensure_trained(ui2) is not None
        assert ui2["notice"] is None, "should not warn twice"
    finally:
        play.Q_TABLE_PATH, train.SAVE_Q, train.SAVE_LOG = old_q, old_sq, old_sl
    print("  missing brain -> warning banner + auto-training + saved")


# ---------------------------------------------------------------
# N KEY — a new random maze becomes THE maze
# ---------------------------------------------------------------
def test_new_random_maze_persists():
    """play.new_random_maze() writes maze_random.json and builds on it."""
    tmp = os.path.join(tempfile.mkdtemp(prefix="maze_"), "maze_random.json")
    old_file = environment.ACTIVE_MAZE_FILE
    environment.ACTIVE_MAZE_FILE = tmp
    try:
        env = play.new_random_maze()
        assert os.path.exists(tmp), "maze not persisted"
        assert env.maze_name == "random"
        assert env.rows == 7 and env.cols == 7
        # The start we render is the maze's own 'S', wherever it is.
        r, c = env.start_pos
        assert env.maze_rows[r][c] == "S", "start_pos must point at S"
        assert path_exists(env.maze_rows), "generated maze must be winnable"
        # Same file back -> same maze (permanent across runs).
        assert environment.load_active_maze(tmp)[0] == "random"
        assert environment.load_random_maze(tmp) == env.maze_rows
    finally:
        environment.ACTIVE_MAZE_FILE = old_file
    print("  generated, persisted, and rebuilt from disk")


# ---------------------------------------------------------------
# V KEY — the visualizer runs as a separate process
# ---------------------------------------------------------------
def test_launch_visualizer():
    calls = []

    def fake_runner(cmd, cwd=None):
        calls.append((cmd, cwd))

    assert play.launch_visualizer(fake_runner) is True
    cmd, cwd = calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == "visualize.py"
    assert cwd == play.BASE_DIR

    def broken_runner(cmd, cwd=None):
        raise OSError("no python")

    assert play.launch_visualizer(broken_runner) is False
    print("  launches [python, visualize.py] in BASE_DIR; OSError -> False")


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
    cases[3] = ("TRAINED", _ui("TRAINED", reward=-10.0, speed_i=5))
    # ...the LIVE dashboard at its most crowded...
    cases.append(("LIVE", _live_ui()))
    cases.append(("LIVE", _live_ui(notice="WARNING: no trained agent!",
                                   notice_sub="training was not launched "
                                              "- starting now")))
    # ...and every banner state
    cases += [
        ("HUMAN", _ui("HUMAN", outcome="goal")),
        ("WATCH", _ui("WATCH", outcome="pit")),
        ("TRAINED", _ui("TRAINED", outcome="timeout")),
        ("TRAINED", _ui("TRAINED",
                        notice="WARNING: no trained agent!",
                        notice_sub="training was not launched - "
                                   "starting it now")),
        ("HUMAN", _ui("HUMAN", notice="NEW MAZE GENERATED!",
                      notice_sub="retraining the agent for it...")),
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
    # maze_name pinned to "classic": the GridWorld below is classic,
    # and a leftover maze_random.json must not change what we train.
    agent, _ = train(episodes=600, quiet=True, maze_name="classic")
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
# THE WHOLE WINDOW, DRIVEN BY A SCRIPT OF KEY PRESSES
# ---------------------------------------------------------------
def test_run_event_script():
    """Drive run() headlessly through SPACE/SPACE/TAB/N/V/SPACE/ESC
    and check the side effects: a persisted random maze, a brain +
    log trained for THAT maze, and the visualizer being launched."""
    tmp = tempfile.mkdtemp(prefix="script_")
    saved = (environment.ACTIVE_MAZE_FILE, play.Q_TABLE_PATH,
             train.SAVE_Q, train.SAVE_LOG, play.Trainer,
             play.launch_visualizer)
    environment.ACTIVE_MAZE_FILE = os.path.join(tmp, "maze_random.json")
    train.SAVE_Q = os.path.join(tmp, "q_table.json")
    train.SAVE_LOG = os.path.join(tmp, "training_log.json")
    play.Q_TABLE_PATH = train.SAVE_Q

    launched = []
    play.launch_visualizer = lambda runner=None: launched.append("V") or True

    RealTrainer = play.Trainer

    class Tiny(RealTrainer):
        """60 episodes instead of 2000 — same code, less waiting."""
        def __init__(self, **kw):
            kw.setdefault("quiet", True)
            super().__init__(episodes=60, **kw)

    play.Trainer = Tiny

    maze_path = os.path.join(tmp, "maze_random.json")
    q_path = os.path.join(tmp, "q_table.json")
    log_path = os.path.join(tmp, "training_log.json")

    script = {
        5: pygame.K_SPACE,     # HUMAN -> WATCH
        10: pygame.K_SPACE,    # WATCH -> LIVE (no brain -> warning)
        40: pygame.K_TAB,      # skip to the next stage
        70: pygame.K_n,        # new maze + retrain + fresh LIVE
        95: pygame.K_v,        # visualizer (brain exists by now)
        100: pygame.K_SPACE,   # LIVE -> TRAINED
        140: pygame.K_ESCAPE,  # quit
    }
    real_get = pygame.event.get
    frame = [0]

    def fake_get(*args, **kwargs):
        events = list(real_get(*args, **kwargs))
        frame[0] += 1
        if frame[0] in script:
            events.append(pygame.event.Event(pygame.KEYDOWN,
                                             key=script[frame[0]]))
        return events

    pygame.event.get = fake_get
    try:
        pygame.init()
        play.run()
    finally:
        pygame.event.get = real_get
        (environment.ACTIVE_MAZE_FILE, play.Q_TABLE_PATH,
         train.SAVE_Q, train.SAVE_LOG, play.Trainer,
         play.launch_visualizer) = saved

    assert os.path.exists(maze_path), "N did not persist a maze"
    rows = environment.load_random_maze(maze_path)
    assert len(rows) == 7 and path_exists(rows)

    assert os.path.exists(q_path), "no brain was saved"
    assert os.path.exists(log_path), "no log was saved"
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    assert log["maze"] == "random", "log trained for the wrong maze"
    assert log["maze_rows"] == rows, "log stores the wrong maze"
    assert launched == ["V"], "visualizer was not launched"
    print(f"  {frame[0]} scripted frames: maze persisted, "
          f"brain+log trained for it, visualizer launched")


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
        test_ensure_trained_warns_and_trains,
        test_new_random_maze_persists,
        test_launch_visualizer,
        test_hud_fits_in_every_mode,
        test_trained_agent_wins_via_apply_step,
        test_run_event_script,
        test_run_opens_and_exits,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} window checks verified.")
