from app.modes.clock import Clock
from app.modes.menu import Menu
from app.modes.paint import Paint
from app.modes.caricature import Caricature
from app.modes.percussion import Percussion
from app.modes.autodrum import AutoDrum
from app.modes.beatmirror import BeatMirror
from app.modes.tetris import Tetris
from app.modes.pong import Pong
from app.modes.worldcup import WorldCup
from app.modes.board import Board


def create_mode_instances(width, height, mode_manager):
    """Create all mode instances in one place."""

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
        "worldcup": WorldCup(width, height, mode_manager),
        "board": Board(width, height, mode_manager),
    }
