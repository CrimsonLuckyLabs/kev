from __future__ import annotations

from fastapi.testclient import TestClient

from kev.decks import list_decks
from kev.server import STATIC_DIR, create_app, load_console_html

client = TestClient(create_app(model="mock"))


def test_console_index_is_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "ZERO POINT ONE" in body
    assert "Decide" in body
    assert "Presets" not in body
    assert "your software" not in body
    assert "example app" not in body
    assert "Page billing now" in body
    assert "Act now" in body
    assert "Paste anything." in body
    assert "Paste a support ticket." not in body
    assert 'id="advanced"' in body
    assert ">Advanced</summary>" in body
    assert "Add Noul" in body
    assert "Add Choice" in body
    assert "Add Score" in body
    assert 'id="mode-json"' in body
    assert "Whether to act, wait, or hand to a person." in body
    assert "/v1/systemone" in body
    assert "/v1/decks" in body
    assert "/v1/meta" in body
    assert "<details" in body
    assert ">request</summary>" in body
    assert ">answers</summary>" in body
    assert "127.0.0.1" not in body
    assert "localhost" not in body
    assert "http://" not in body
    assert (STATIC_DIR / "index.html").is_file()


def test_index_html_alias() -> None:
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "Decide" in response.text


def test_healthz_and_meta_ok() -> None:
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    meta = client.get("/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["model"] == "mock"


def test_decks_are_question_packs() -> None:
    decks = client.get("/v1/decks")
    assert decks.status_code == 200
    body = decks.json()
    ids = {item["id"] for item in body["decks"]}
    assert ids == {"general", "triage", "trade"}
    assert body["decks"][0]["id"] == "general"
    actionable = body["decks"][0]["questions"]["actionable"]
    assert actionable["criteria"]["true"]
    assert actionable["criteria"]["false"]
    assert "presets" not in body
    assert {item["id"] for item in list_decks()} == ids


def test_console_html_loader_not_empty() -> None:
    html = load_console_html()
    assert html.strip()
    assert 'fetch("/v1/systemone"' in html


def test_console_judge_omits_model_uses_bound() -> None:
    payload = {
        "state": {"ticket": "I was charged twice. Refund now, ASAP."},
        "questions": {
            "billing": {"type": "noul", "instructions": "Is this about billing?"},
            "tone": {
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": {"calm": None, "frustrated": None, "angry": None},
            },
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert body["usage"]["output_tokens"] == 0
    assert set(body) == {"id", "model", "answers", "usage"}
    assert body["answers"]["billing"]["type"] == "noul"
    assert body["answers"]["tone"]["choice"] in {"calm", "frustrated", "angry"}
