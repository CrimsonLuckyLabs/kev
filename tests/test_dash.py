from __future__ import annotations

from fastapi.testclient import TestClient

from kev.dash import DEFAULT_PASSWORD, token_ok
from kev.server import create_app


def test_dash_requires_password() -> None:
    client = TestClient(create_app(model="mock"))
    page = client.get("/dash")
    assert page.status_code == 200
    assert "password" in page.text
    assert "ZERO POINT ONE" in page.text
    assert "OPERATOR DASH" not in page.text
    assert DEFAULT_PASSWORD not in page.text
    stats = client.get("/dash/stats")
    assert stats.status_code == 401


def test_dash_rejects_wrong_password() -> None:
    client = TestClient(create_app(model="mock"))
    response = client.post("/dash/login", json={"password": "nope"})
    assert response.status_code == 403


def test_dash_login_and_stats() -> None:
    client = TestClient(create_app(model="mock"))
    client.get("/")
    client.post(
        "/v1/systemone",
        json={
            "state": {"ticket": "hello"},
            "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
        },
    )
    login = client.post("/dash/login", json={"password": DEFAULT_PASSWORD})
    assert login.status_code == 200
    assert login.json() == {"ok": True}
    token = client.cookies.get("kev_dash")
    assert token_ok(DEFAULT_PASSWORD, token)
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


def test_public_console_does_not_leak_dash_password() -> None:
    client = TestClient(create_app(model="mock"))
    home = client.get("/").text
    assert DEFAULT_PASSWORD not in home
    assert "/dash/login" not in home
