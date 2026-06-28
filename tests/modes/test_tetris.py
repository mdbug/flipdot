import time

import pytest

import app.modes.tetris as tetris_mod
from app.core.mode_manager import ModeManager
from app.modes.tetris import Tetris


@pytest.fixture
def tetris(tmp_path, monkeypatch):
    # Point the high score at a throwaway file so tests never touch real state.
    monkeypatch.setenv(Tetris._BEST_FILE_ENV, str(tmp_path / "highscore"))
    # Pose helpers are exercised by _handle_gestures; stub them so a fake
    # "person present" object doesn't need real MediaPipe landmarks.
    monkeypatch.setattr(tetris_mod.human_pose, "is_left_hand_raised", lambda pr: False)
    monkeypatch.setattr(
        tetris_mod.human_pose, "get_right_index_finger_position", lambda pr: (None, None)
    )
    return Tetris(28, 28, ModeManager())


def test_ai_lock_taints_and_blocks_highscore(tetris):
    now = time.time()
    tetris._player["lines"] = 5
    tetris._best = 0
    assert tetris._ai_played_this_game is False
    assert tetris._player["piece"] is not None

    # AI is in control at the moment a piece locks.
    tetris._ai_in_control = True
    tetris._player_hard_drop(now)  # routes through _player_lock

    assert tetris._ai_played_this_game is True
    tetris._trigger_game_over(now)
    assert tetris._best == 0  # AI-assisted game must not update the record


def test_brief_takeover_without_lock_does_not_taint(tetris):
    now = time.time()
    tetris._player["lines"] = 7
    tetris._best = 0

    # AI briefly steers, but the human regains control before any piece locks.
    tetris._ai_in_control = True
    tetris._ai_in_control = False
    tetris._player_hard_drop(now)

    assert tetris._ai_played_this_game is False
    tetris._trigger_game_over(now)
    assert tetris._best == 7  # clean game updates the record


def test_handle_gestures_sets_ai_control_flag(tetris):
    now = time.time()

    # Absent past the takeover delay with no controller input -> AI in control.
    tetris._last_person_time = now - tetris.AI_TAKEOVER_DELAY - 1
    tetris._last_controller_input_time = None
    tetris._handle_gestures(None, now)
    assert tetris._ai_in_control is True

    # A detected person clears it again on the next frame.
    class _Pose:
        pose_landmarks = object()

    tetris._handle_gestures(_Pose(), now)
    assert tetris._ai_in_control is False
