from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kev.limits import WindowCounter, payload_error
from kev.server import create_app

pytestmark = pytest.mark.limits

QUESTION = {"billing": {"type": "noul", "instructions": "Is this about billing?"}}


@pytest.fixture
def tight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEV_JUDGE_PER_MIN", "2")
    monkeypatch.setenv("KEV_LOGIN_PER_MIN", "2")
    monkeypatch.setenv("KEV_MAX_INFLIGHT", "4")
    monkeypatch.setenv("KEV_MAX_QUESTIONS", "3")
    monkeypatch.setenv("KEV_MAX_OPTIONS", "4")
    monkeypatch.setenv("KEV_MAX_STATE_CHARS", "40")
    monkeypatch.setenv("KEV_DASH_PASSWORD", "test-dash-secret")


def test_window_counter_blocks_after_limit() -> None:
    counter = WindowCounter()
    assert counter.allow("a", 2, window_s=60, now=100.0)[0] is True
    assert counter.allow("a", 2, window_s=60, now=101.0)[0] is True
    allowed, retry = counter.allow("a", 2, window_s=60, now=102.0)
    assert allowed is False
    assert retry >= 1
    assert counter.allow("b", 2, window_s=60, now=102.0)[0] is True
    assert counter.allow("a", 2, window_s=60, now=161.0)[0] is True


def test_window_zero_is_unlimited() -> None:
    counter = WindowCounter()
    assert all(counter.allow("a", 0, now=float(i))[0] for i in range(50))


def test_payload_error_caps_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEV_MAX_QUESTIONS", "2")
    monkeypatch.setenv("KEV_MAX_OPTIONS", "8")
    too_many = {
        "a": {"type": "noul", "instructions": "A?"},
        "b": {"type": "noul", "instructions": "B?"},
        "c": {"type": "noul", "instructions": "C?"},
    }
    assert payload_error(too_many) is not None
    assert payload_error({"a": {"type": "noul", "instructions": "A?"}}) is None


def test_judge_rate_limit(tight: None) -> None:
    client = TestClient(create_app(model="mock"))
    body = {"state": {"ticket": "hi"}, "questions": QUESTION}
    assert client.post("/v1/systemone", json=body).status_code == 200
    assert client.post("/v1/systemone", json=body).status_code == 200
    blocked = client.post("/v1/systemone", json=body)
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "rate limit"
    assert int(blocked.headers["Retry-After"]) >= 1


def test_login_rate_limit(tight: None) -> None:
    client = TestClient(create_app(model="mock"))
    assert client.post("/dash/login", json={"password": "nope"}).status_code == 403
    assert client.post("/dash/login", json={"password": "nope"}).status_code == 403
    blocked = client.post("/dash/login", json={"password": "nope"})
    assert blocked.status_code == 429


def test_too_many_questions(tight: None) -> None:
    client = TestClient(create_app(model="mock"))
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"Q{i}?"} for i in range(4)
    }
    response = client.post(
        "/v1/systemone", json={"state": {"ticket": "hi"}, "questions": questions}
    )
    assert response.status_code == 400
    assert "questions" in response.json()["detail"]


def test_too_many_options(tight: None) -> None:
    client = TestClient(create_app(model="mock"))
    criteria = {f"opt{i}": None for i in range(5)}
    response = client.post(
        "/v1/systemone",
        json={
            "state": {"ticket": "hi"},
            "questions": {
                "pick": {"type": "choice", "instructions": "Which?", "criteria": criteria}
            },
        },
    )
    assert response.status_code == 400
    assert "options" in response.json()["detail"]


def test_http_state_char_cap(tight: None) -> None:
    client = TestClient(create_app(model="mock"))
    response = client.post(
        "/v1/systemone",
        json={"state": {"ticket": "x" * 80}, "questions": QUESTION},
    )
    assert response.status_code == 400
    assert "max_state_chars" in response.json()["detail"]
