from typing import Any

from app.core.mode_manager import ModeManager
from app.modes.autodrum import AutoDrum
from app.modes.beatmirror import BeatMirror
from app.modes.board import Board
from app.modes.caricature import Caricature
from app.modes.clock import Clock
from app.modes.font_preview import FontPreview
from app.modes.menu import Menu
from app.modes.paint import Paint
from app.modes.percussion import Percussion
from app.modes.pong import Pong
from app.modes.script_mode import ScriptMode
from app.modes.tank import Tank
from app.modes.tetris import Tetris
from app.modes.worldcup import WorldCup


def create_mode_instances(width: int, height: int, mode_manager: ModeManager) -> dict[str, Any]:
    """Create one instance of every mode, keyed by its ``ModeManager`` mode id."""
    return {
        "clock": Clock(width, height),
        "menu": Menu(width, height, mode_manager),
        "paint": Paint(width, height, mode_manager),
        "caricature": Caricature(width, height, mode_manager),
        "percussion": Percussion(width, height, mode_manager),
        "autodrum": AutoDrum(width, height, mode_manager),
        "beatmirror": BeatMirror(width, height, mode_manager),
        "tetris": Tetris(width, height, mode_manager),
        "pong": Pong(width, height, mode_manager),
        "tank": Tank(width, height, mode_manager),
        "worldcup": WorldCup(width, height, mode_manager),
        "board": Board(width, height, mode_manager),
        "font_preview": FontPreview(width, height, mode_manager),
        "script": ScriptMode(width, height),
    }
