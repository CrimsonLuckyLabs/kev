from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from kev import KevClient, Noul
from kev.access import GATE_COOKIE
from kev.server import create_app
from kev.types import SystemOneResponse


def _judge_body() -> dict[str, object]:
    return {
        "state": {"ticket": "hello"},
        "questions": {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    }


def test_v1_index_lists_judge_and_limits() -> None:
    client = TestClient(create_app(model="mock"))
    response = client.get("/v1")
    assert response.status_code == 200
    body = response.json()
    assert body["judge"]["path"] == "/v1/systemone"
    assert "/v1/meta" in body["endpoints"].values()
    assert body["auth"] == "open"
    assert "judge_per_min" in body["limits"]
    assert "max_questions" in body["limits"]


def test_meta_includes_limits() -> None:
    client = TestClient(create_app(model="mock"))
    meta = client.get("/v1/meta").json()
    assert meta["model"] == "mock"
    assert set(meta["limits"]) >= {
        "judge_per_min",
        "max_questions",
        "max_options",
        "max_state_chars",
        "max_inflight",
    }


def test_judge_open_without_api_key() -> None:
    client = TestClient(create_app(model="mock"))
    assert client.post("/v1/systemone", json=_judge_body()).status_code == 200


def test_judge_requires_bearer_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEV_API_KEY", "test-api-key")
    client = TestClient(create_app(model="mock"))
    denied = client.post("/v1/systemone", json=_judge_body())
    assert denied.status_code == 401
    wrong = client.post(
        "/v1/systemone",
        json=_judge_body(),
        headers={"Authorization": "Bearer nope"},
    )
    assert wrong.status_code == 401
    ok = client.post(
        "/v1/systemone",
        json=_judge_body(),
        headers={"Authorization": "Bearer test-api-key"},
    )
    assert ok.status_code == 200


def test_console_cookie_can_judge_without_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEV_API_KEY", "test-api-key")
    client = TestClient(create_app(model="mock"))
    home = client.get("/")
    assert home.status_code == 200
    assert client.cookies.get(GATE_COOKIE)
    assert "test-api-key" not in home.text
    assert "KEV_API_KEY" not in home.text
    judged = client.post("/v1/systemone", json=_judge_body())
    assert judged.status_code == 200


def test_cors_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEV_CORS_ORIGINS", "https://trade.example")
    client = TestClient(create_app(model="mock"))
    preflight = client.options(
        "/v1/systemone",
        headers={
            "Origin": "https://trade.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.headers.get("access-control-allow-origin") == "https://trade.example"
    blocked = client.options(
        "/v1/systemone",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert blocked.headers.get("access-control-allow-origin") != "https://evil.example"


def test_client_retries_429(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, json={"detail": "rate limit"}, headers={"Retry-After": "0"})
        body = {
            "id": "kev_retry",
            "model": "mock",
            "answers": {"billing": {"type": "noul", "noul": 0.5}},
            "usage": {"input_tokens": 1, "output_tokens": 0, "latency_ms": 1.0},
        }
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url="http://kev.test")
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with KevClient(model="mock", http_client=http, base_url="http://kev.test") as remote:
        res = remote.system_one(
            {"ticket": "hi"},
            {"billing": Noul(instructions="Is this about billing?")},
        )
    assert isinstance(res, SystemOneResponse)
    assert res.id == "kev_retry"
    assert hits["n"] == 2


def test_client_sends_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization") or ""
        return httpx.Response(
            200,
            json={
                "id": "kev_key",
                "model": "mock",
                "answers": {"billing": {"type": "noul", "noul": 0.4}},
                "usage": {"input_tokens": 1, "output_tokens": 0, "latency_ms": 1.0},
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://kev.test")
    with KevClient(
        model="mock",
        http_client=http,
        base_url="http://kev.test",
        api_key="from-arg",
    ) as remote:
        remote.system_one(
            {"ticket": "hi"},
            {"billing": Noul(instructions="Is this about billing?")},
        )
    assert seen["authorization"] == "Bearer from-arg"
