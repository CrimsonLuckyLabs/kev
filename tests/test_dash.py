from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kev.dash import token_ok
from kev.server import create_app

SECRET = "test-dash-secret"


@pytest.fixture
def dash_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("KEV_DASH_PASSWORD", SECRET)
    return SECRET


def test_dash_disabled_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEV_DASH_PASSWORD", raising=False)
    client = TestClient(create_app(model="mock"))
    login = client.post("/dash/login", json={"password": "anything"})
    assert login.status_code == 503
    assert client.get("/dash/stats").status_code == 401


def test_dash_requires_password(dash_env: str) -> None:
    client = TestClient(create_app(model="mock"))
    page = client.get("/dash")
    assert page.status_code == 200
    assert "password" in page.text
    assert "ZERO POINT ONE" in page.text
    assert dash_env not in page.text
    stats = client.get("/dash/stats")
    assert stats.status_code == 401


def test_dash_rejects_wrong_password(dash_env: str) -> None:
    client = TestClient(create_app(model="mock"))
    response = client.post("/dash/login", json={"password": "nope"})
    assert response.status_code == 403


def test_dash_login_and_stats(dash_env: str) -> None:
    client = TestClient(create_app(model="mock"))
    client.get("/")
    client.post(
        "/v1/systemone",
        json={
            "state": {"ticket": "hello"},
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        },
    )
    login = client.post("/dash/login", json={"password": dash_env})
    assert login.status_code == 200
    assert login.json() == {"ok": True}
    token = client.cookies.get("kev_dash")
    assert token_ok(dash_env, token)
    dash = client.get("/dash")
    assert dash.status_code == 200
    assert "ZERO POINT ONE" in dash.text
    stats = client.get("/dash/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["totals"]["users"] >= 1
    assert body["totals"]["pageviews"] >= 1
    assert body["totals"]["judges"] >= 1
    assert body["users"][0]["visits"] >= 1


def test_public_console_does_not_leak_dash_secret(dash_env: str) -> None:
    client = TestClient(create_app(model="mock"))
    home = client.get("/").text
    assert dash_env not in home
    assert "/dash/login" not in home
    assert "KEV_DASH_PASSWORD" not in home
