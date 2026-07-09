"""Tests for miscellaneous web server routes."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.infrastructure.web_server import WebServer  # noqa: E402


class DummyInputHub:
    def submit_pointer(self, source, x, y):
        pass

    def submit_click(self, source, x, y):
        pass

    def submit_action(self, source, action):
        pass

    def set_button_down(self, source, is_down):
        pass


def test_favicon_is_bodyless_204(tmp_path):
    server = WebServer(
        input_hub=DummyInputHub(),
        host="127.0.0.1",
        port=9400,
        settings_path=tmp_path / "settings.json",
    )
    client = TestClient(server._app)

    response = client.get("/favicon.ico")

    # A 204 must not carry a body (the old JSON body was malformed HTTP).
    assert response.status_code == 204
    assert response.content == b""
