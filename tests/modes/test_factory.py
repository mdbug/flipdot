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
        "app.modes.tank": "Tank",
        "app.modes.worldcup": "WorldCup",
        "app.modes.board": "Board",
        "app.modes.font_preview": "FontPreview",
        "app.modes.script_mode": "ScriptMode",
        "app.modes.life": "LifeMirror",
        "app.modes.sandfall": "Sandfall",
    }

    for module_name, symbol in stubs.items():
        monkeypatch.setitem(sys.modules, module_name, types.SimpleNamespace(**{symbol: _StubMode}))

    # factory binds these via `import app.services.x as x`, i.e. attribute
    # access on the package, so stub both sys.modules and the package attrs.
    services_pkg = importlib.import_module("app.services")
    hair_stub = types.SimpleNamespace(get_hair_mask=lambda frame: None)
    human_pose_stub = types.SimpleNamespace(
        draw_face_features=lambda dots, results, width, height, **kwargs: dots,
        face_feature_anchor=lambda x_norm, y_norm, width, height: (x_norm * width, y_norm * height),
    )
    monkeypatch.setitem(sys.modules, "app.services.hair_segmentation", hair_stub)
    monkeypatch.setitem(sys.modules, "app.services.human_pose", human_pose_stub)
    monkeypatch.setattr(services_pkg, "hair_segmentation", hair_stub, raising=False)
    monkeypatch.setattr(services_pkg, "human_pose", human_pose_stub, raising=False)
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
        "tank",
        "worldcup",
        "board",
        "font_preview",
        "script",
        "life",
        "sandfall",
    }


def test_create_mode_instances_wires_constructor_args(monkeypatch):
    factory = _load_factory_module(monkeypatch)
    mode_manager = object()

    instances = factory.create_mode_instances(28, 28, mode_manager)

    assert instances["clock"].args == (28, 28)
    assert instances["menu"].args == (28, 28, mode_manager)
    assert instances["paint"].args == (28, 28, mode_manager)
    assert callable(instances["caricature"].kwargs["hair_mask_provider"])
    assert callable(instances["caricature"].kwargs["real_face_anchor"])
    assert instances["life"].args == (28, 28)
    assert instances["sandfall"].args == (28, 28)
    assert callable(instances["sandfall"].kwargs["face_renderer"])
