import importlib
import sys
import types


def _load_factory_module(monkeypatch):
    class _StubMode:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    stubs = {
        "app.modes.clock": "Clock",
        "app.modes.menu": "Menu",
        "app.modes.paint": "Paint",
        "app.modes.caricature": "Caricature",
        "app.modes.percussion": "Percussion",
        "app.modes.autodrum": "AutoDrum",
        "app.modes.beatmirror": "BeatMirror",
        "app.modes.tetris": "Tetris",
        "app.modes.pong": "Pong",
        "app.modes.worldcup": "WorldCup",
        "app.modes.board": "Board",
        "app.modes.font_preview": "FontPreview",
        "app.modes.script_mode": "ScriptMode",
    }

    for module_name, symbol in stubs.items():
        monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(**{symbol: _StubMode}))

    sys.modules.pop("app.modes.factory", None)
    return importlib.import_module("app.modes.factory")


def test_create_mode_instances_returns_expected_keys(monkeypatch):
    factory = _load_factory_module(monkeypatch)
    mode_manager = object()

    instances = factory.create_mode_instances(28, 28, mode_manager)

    assert set(instances.keys()) == {
        "clock",
        "menu",
        "paint",
        "caricature",
        "percussion",
        "autodrum",
        "beatmirror",
        "tetris",
        "pong",
        "worldcup",
        "board",
        "font_preview",
        "script",
    }


def test_create_mode_instances_wires_constructor_args(monkeypatch):
    factory = _load_factory_module(monkeypatch)
    mode_manager = object()

    instances = factory.create_mode_instances(28, 28, mode_manager)

    assert instances["clock"].args == (28, 28)
    assert instances["menu"].args == (28, 28, mode_manager)
    assert instances["paint"].args == (28, 28, mode_manager)
