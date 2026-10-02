"""
test_play.py — headless checks for the cart-pole window (step 4)
================================================================

Runs with SDL's "dummy" video driver — no real window pops up, but
every drawing code path still executes on an off-screen surface.

Covers: the mode cycle, loading the graduate, the auto-training
safety net, the LIVE demo replay, the HUD never clipping in any
mode, the graduate surviving through apply_step(), the whole
window driven by a script of key presses, and a clean open/quit.

Run with:  python test_play.py
"""

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"   # MUST precede pygame import

import json
import sys
import tempfile

import numpy as np
import pygame

import play
import train
from agent import QAgent
from environment import CartPole, LEFT, RIGHT


def _ui(mode="HUMAN", **overrides):
    """A ui dict shaped exactly like the one run() builds."""
    ui = {"mode": mode, "reward": 0.0, "outcome": None,
          "speed_i": play.SPEED_DEFAULT_I, "notice": None,
          "notice_sub": None, "live": None, "last_action": None}
    ui.update(overrides)
    return ui


def _live_ui(**overrides):
    """A worst-case LIVE dashboard ui (biggest possible numbers)."""
    return _ui("LIVE",
               live={"episode": 15000, "total": 15000, "stage": 2,
                     "epsilon": 1.0, "win": 100.0, "best": 100.0},
               speed_i=len(play.LIVE_EPS_PER_SEC) - 1,
               **overrides)


# ---------------------------------------------------------------
# MODE CYCLING + KEY MAP
# ---------------------------------------------------------------
def test_mode_cycle():
    """SPACE walks HUMAN -> WATCH -> LIVE -> TRAINED -> back to HUMAN."""
    assert play.next_mode("HUMAN") == "WATCH"
    assert play.next_mode("WATCH") == "LIVE"
    assert play.next_mode("LIVE") == "TRAINED"
    assert play.next_mode("TRAINED") == "HUMAN"
    # The arrows must reach the environment's own action numbers.
    assert play.KEY_TO_ACTION[pygame.K_LEFT] == LEFT
    assert play.KEY_TO_ACTION[pygame.K_RIGHT] == RIGHT
    print("  HUMAN -> WATCH -> LIVE -> TRAINED -> HUMAN; arrows mapped")


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
    agent = QAgent(n_actions=2)
    agent.get_q((0.1, 0.0, -0.02, 0.0))[1] = 6.5
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
    old_eps = play.RETRAIN_EPISODES
    play.Q_TABLE_PATH = os.path.join(tmp, "q_table.json")
    train.SAVE_Q = play.Q_TABLE_PATH
    train.SAVE_LOG = os.path.join(tmp, "training_log.json")
    play.RETRAIN_EPISODES = 80      # same code, less waiting
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
        play.RETRAIN_EPISODES = old_eps
    print("  missing brain -> warning banner + auto-training + saved")


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
# LIVE'S DEMO REPLAY — subsampled so it never runs 10 minutes
# ---------------------------------------------------------------
def test_build_demo_subsamples():
    """build_demo() runs one greedy episode and returns every other
    state — plus the final one, so the ending is always visible."""
    agent = QAgent(n_actions=2)     # untrained: constant pushes, dies fast
    states, reason = play.build_demo(agent)
    assert reason in ("pole_fell", "cart_out", "time_limit"), reason
    assert len(states) >= 2, "episode produced no states"
    ticks = [s[0] for s in states]
    assert ticks[0] == 0 and ticks == sorted(ticks), "ticks must count up"
    # Subsampled at DEMO_SUBSAMPLE: at most half (rounded up) remain.
    assert len(states) <= ticks[-1] // play.DEMO_SUBSAMPLE + 2
    assert states[-1][0] == ticks[-1], "the final state must be kept"
    print(f"  {ticks[-1] + 1} ticks -> {len(states)} states, "
          f"ends with {reason}")


# ---------------------------------------------------------------
# APPLY_STEP GUARDS + RESTART
# ---------------------------------------------------------------
def test_apply_step_guard_and_restart():
    """apply_step on a finished episode is a no-op; restart clears
    everything the banner/scoreboard depends on."""
    env = CartPole()
    ui = _ui("HUMAN", reward=12.0, outcome="pole_fell",
             last_action=RIGHT)
    env.done = True                      # pretend the episode ended
    play.apply_step(env, ui, LEFT)
    assert ui["reward"] == 12.0, "reward moved on a dead episode"
    assert ui["outcome"] == "pole_fell", "outcome was overwritten"

    play.restart(env, ui)
    assert env.done is False and env.steps == 0
    assert ui["reward"] == 0.0 and ui["outcome"] is None
    assert ui["last_action"] is None
    print("  dead episode untouched; restart resets score + banner")


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
        # worst case: 500/500 steps, biggest reward (+500.0), banner
        # up, notice up...
        cases.append((mode, _ui(mode, reward=500.0, last_action=RIGHT)))
    # ...the longest speed text in the timer modes...
    cases[1] = ("WATCH", _ui("WATCH", reward=500.0, speed_i=5))
    cases[3] = ("TRAINED", _ui("TRAINED", reward=500.0, speed_i=5))
    # ...the LIVE dashboard at its most crowded...
    cases.append(("LIVE", _live_ui()))
    cases.append(("LIVE", _live_ui(notice="WARNING: no trained agent!",
                                   notice_sub="training was not launched "
                                              "- starting now")))
    # ...and every banner state
    cases += [
        ("HUMAN", _ui("HUMAN", outcome="pole_fell")),
        ("WATCH", _ui("WATCH", outcome="cart_out")),
        ("TRAINED", _ui("TRAINED", outcome="time_limit")),
        ("TRAINED", _ui("TRAINED",
                        notice="WARNING: no trained agent!",
                        notice_sub="training was not launched - "
                                   "starting it now")),
        ("HUMAN", _ui("HUMAN", notice="Could not open the visualizer",
                      notice_sub="run: python visualize.py")),
    ]

    for mode, ui in cases:
        env = CartPole()
        env.reset()
        env.steps = 500                        # worst-case counter text
        env.x, env.theta = 2.4, 0.2            # extreme drawing positions
        play.draw(wide, env, ui)
        strip = pygame.surfarray.array3d(wide)[play.WIDTH:, :, :]
        assert np.all(strip == bg), \
            f"HUD/banner clipped past window edge in {ui}"

    print(f"  {len(cases)} worst-case frames: nothing clipped")


# ---------------------------------------------------------------
# THE GRADUATE PLAYS THROUGH THE WINDOW'S PLUMBING
# ---------------------------------------------------------------
def test_trained_agent_survives_via_apply_step():
    """Train briefly, then drive a full greedy episode through
    apply_step() — the exact code path TRAINED mode uses in the
    real window. reward must equal steps, every tick, always."""
    import random

    random.seed(7)
    trainer = train.Trainer(episodes=1500, quiet=True)
    trainer.run_all()
    agent = trainer.agent

    env = CartPole()
    ui = _ui("TRAINED")
    play.restart(env, ui)
    while not env.done:
        action = agent.choose_action(env.state, explore=False)
        play.apply_step(env, ui, action)

    assert ui["reward"] == env.steps, \
        f"scoreboard drift: reward {ui['reward']} != steps {env.steps}"
    assert ui["outcome"] in ("pole_fell", "cart_out", "time_limit")
    assert env.steps >= 40, \
        f"brain barely better than random: {env.steps} steps"
    print(f"  graduate survived {env.steps} ticks "
          f"({ui['outcome']}, reward {ui['reward']:+.1f})")


# ---------------------------------------------------------------
# THE WHOLE WINDOW, DRIVEN BY A SCRIPT OF KEY PRESSES
# ---------------------------------------------------------------
def test_run_event_script():
    """Drive run() headlessly: push the cart, walk through every
    mode, skip a LIVE stage, open the charts, and quit. Side
    effects: a brain + training log written to the patched paths
    and the visualizer launched exactly once."""
    tmp = tempfile.mkdtemp(prefix="script_")
    saved = (play.Q_TABLE_PATH, train.SAVE_Q, train.SAVE_LOG,
             play.Trainer, play.launch_visualizer)
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

    q_path = os.path.join(tmp, "q_table.json")
    log_path = os.path.join(tmp, "training_log.json")

    script = {
        5: pygame.K_LEFT,      # HUMAN: one push through apply_step
        12: pygame.K_SPACE,    # HUMAN -> WATCH
        30: pygame.K_SPACE,    # WATCH -> LIVE (no brain -> warning)
        55: pygame.K_TAB,      # skip the stage -> saves progress
        80: pygame.K_v,        # visualizer (brain exists by now)
        95: pygame.K_SPACE,    # LIVE -> TRAINED
        130: pygame.K_ESCAPE,  # quit
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
        (play.Q_TABLE_PATH, train.SAVE_Q, train.SAVE_LOG,
         play.Trainer, play.launch_visualizer) = saved

    assert os.path.exists(q_path), "no brain was saved"
    assert os.path.exists(log_path), "no log was saved"
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    assert len(log["episode"]) == 60, "log has the wrong episode count"
    assert "maze" not in log, "cart-pole log must not carry maze data"
    assert launched == ["V"], "visualizer was not launched"
    print(f"  {frame[0]} scripted frames: brain+log trained, "
          f"visualizer launched")


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
        test_launch_visualizer,
        test_build_demo_subsamples,
        test_apply_step_guard_and_restart,
        test_hud_fits_in_every_mode,
        test_trained_agent_survives_via_apply_step,
        test_run_event_script,
        test_run_opens_and_exits,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} window checks verified.")
