from __future__ import annotations

from fastapi.testclient import TestClient

from kev import Choice, KevClient, Noul, Score
from kev.server import create_app

app = create_app(model="mock")
client = TestClient(app)

CANONICAL_PAYLOAD = {
    "model": "mock",
    "state": {"ticket": "charged twice ASAP"},
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "Tone?",
            "criteria": {"calm": None, "frustrated": None, "angry": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "Urgency?",
            "criteria": ["can wait", "this week", "today"],
        },
    },
}


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models() -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    ids = {item["id"] for item in body["models"]}
    assert "mock" in ids
    assert "kev-latest" in ids
    mock = next(item for item in body["models"] if item["id"] == "mock")
    assert mock["available"] is True


def test_systemone_wire_shape() -> None:
    response = client.post("/v1/systemone", json=CANONICAL_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "model", "answers", "usage"}
    assert body["id"].startswith("kev_")
    assert body["model"] == "mock"
    assert set(body["usage"]) == {"input_tokens", "output_tokens", "latency_ms"}
    assert body["usage"]["output_tokens"] == 0
    billing = body["answers"]["billing"]
    assert billing["type"] == "noul"
    assert 0.05 < billing["noul"] < 0.95
    tone = body["answers"]["tone"]
    assert tone["type"] == "choice"
    assert tone["choice"] in {"calm", "frustrated", "angry"}
    assert "confidence" in tone
    urgency = body["answers"]["urgency"]
    assert urgency["type"] == "score"
    assert 0.0 <= urgency["score"] <= 2.0
    assert urgency["level"] in {"can wait", "this week", "today"}
    assert "confidence" in urgency
    assert "nouls" not in body


def test_systemone_accepts_null_choice_criteria() -> None:
    payload = {
        "model": "mock",
        "state": {"ticket": "charged twice ASAP"},
        "questions": {
            "tone": {
                "type": "choice",
                "instructions": "Tone?",
                "criteria": {"calm": None, "frustrated": None, "angry": None},
            }
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200


def test_systemone_rejects_unknown_type() -> None:
    payload = {
        "model": "mock",
        "state": "hello",
        "questions": {
            "bad": {"type": "chat", "instructions": "Say something nice"},
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert "chat" in response.text.lower() or "type" in response.text.lower()


def test_systemone_rejects_empty_instructions() -> None:
    payload = {
        "model": "mock",
        "state": "hello",
        "questions": {"billing": {"type": "noul", "instructions": "   "}},
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422


def test_systemone_rejects_choice_with_one_option() -> None:
    payload = {
        "model": "mock",
        "state": "hello",
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "Which?",
                "criteria": {"only": None},
            }
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422


def test_systemone_rejects_score_with_one_level() -> None:
    payload = {
        "model": "mock",
        "state": "hello",
        "questions": {
            "urgency": {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["today"],
            }
        },
    }
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422


def test_create_app_factory_is_isolated() -> None:
    other = TestClient(create_app(model="mock", seed=7))
    first = client.post("/v1/systemone", json=CANONICAL_PAYLOAD).json()
    second = other.post("/v1/systemone", json=CANONICAL_PAYLOAD).json()
    assert first["answers"]["billing"]["noul"] != second["answers"]["billing"]["noul"]
    assert first != second


def test_kev_latest_on_mock_server_is_rejected() -> None:
    payload = dict(CANONICAL_PAYLOAD)
    payload["model"] = "kev-latest"
    response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 400


def test_remote_client_asgi_transport() -> None:
    with TestClient(app) as http:
        with KevClient(model="mock", http_client=http, base_url="http://kev.test") as remote:
            assert remote.backend is None
            res = remote.system_one(
                {"ticket": "charged twice ASAP"},
                {
                    "billing": Noul(instructions="Is this about billing?"),
                    "tone": Choice(
                        instructions="Tone?",
                        criteria={"calm": None, "frustrated": None, "angry": None},
                    ),
                    "urgency": Score(
                        instructions="Urgency?",
                        criteria=["can wait", "this week", "today"],
                    ),
                },
            )
    assert res.id.startswith("kev_")
    assert res.model == "mock"
    assert res.usage.output_tokens == 0
    assert "billing" in res.nouls
    assert res.scores["urgency"].level in {"can wait", "this week", "today"}
