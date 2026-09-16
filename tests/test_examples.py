from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from kev import ChoiceAnswer, KevClient, NoulAnswer, ScoreAnswer, SystemOneResponse
from kev.types import Usage

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(filename: str) -> ModuleType:
    path = EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triage_closed_ticket_skips_kev() -> None:
    triage = _load_example("triage_ticket.py")

    class Boom(KevClient):
        def system_one(self, *args: object, **kwargs: object) -> SystemOneResponse:  # type: ignore[override]
            raise AssertionError("closed tickets must not call Kev")

    route, response = triage.evaluate(Boom(model="mock"), {"status": "closed", "ticket": "old"})
    assert route == "no_action"
    assert response is None


def test_triage_compose_routes() -> None:
    triage = _load_example("triage_ticket.py")
    billing_prio = SystemOneResponse(
        id="kev_test",
        model="mock",
        answers={
            "billing": NoulAnswer(noul=0.91),
            "tone": ChoiceAnswer(
                choice="calm",
                probabilities={"calm": 1.0, "frustrated": 0.0, "angry": 0.0},
                confidence=1.0,
            ),
            "urgency": ScoreAnswer(
                score=1.8,
                levels=["can wait", "this week", "today"],
                probabilities={"can wait": 0.1, "this week": 0.0, "today": 0.9},
                confidence=0.7,
            ),
        },
        usage=Usage(input_tokens=1, output_tokens=0, latency_ms=1.0),
    )
    assert triage.compose_route({"status": "open"}, billing_prio) == "billing_prio"

    angry = SystemOneResponse(
        id="kev_test",
        model="mock",
        answers={
            "billing": NoulAnswer(noul=0.2),
            "tone": ChoiceAnswer(
                choice="angry",
                probabilities={"calm": 0.1, "frustrated": 0.2, "angry": 0.7},
                confidence=0.61,
            ),
            "urgency": ScoreAnswer(
                score=0.4,
                levels=["can wait", "this week", "today"],
                probabilities={"can wait": 0.7, "this week": 0.2, "today": 0.1},
                confidence=0.4,
            ),
        },
        usage=Usage(input_tokens=1, output_tokens=0, latency_ms=1.0),
    )
    assert triage.compose_route({"status": "open"}, angry) == "human"
    assert triage.compose_route({"status": "open"}, None) == "default"
