from __future__ import annotations

import inspect
import json
from pathlib import Path

import httpx
import pytest

from kev.train.dataset import DecisionDataset
from kev.train.distill import (
    chat_teacher_json,
    distill_rows,
    extract_json_object,
    fake_teacher_payload,
    load_question_pack,
    load_tickets,
    run_distill,
    teacher_answer_to_row,
    teacher_user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "examples" / "triage_questions.yaml"
RAW = ROOT / "data" / "raw" / "tickets.jsonl"


def test_question_pack_is_triage() -> None:
    specs = load_question_pack(PACK)
    assert [item.id for item in specs] == ["billing", "tone", "urgency"]
    assert [item.type for item in specs] == ["noul", "choice", "score"]


def test_teacher_rationale_is_discarded() -> None:
    billing = load_question_pack(PACK)[0]
    payload = {
        "type": "noul",
        "noul": 0.86,
        "choice": None,
        "probabilities": None,
        "score_level": None,
        "rationale_internal": "because money moved twice",
    }
    row = teacher_answer_to_row(
        "t1",
        {"ticket": "charged twice"},
        billing,
        payload,
        source="teacher:test",
    )
    dumped = row.model_dump()
    assert "rationale_internal" not in dumped
    assert "rationale" not in dumped
    assert row.teacher_p == 0.86
    assert row.label.noul == 1.0
    assert "because money moved twice" not in json.dumps(dumped)


def test_teacher_choice_and_score_typed_fields() -> None:
    questions = {item.id: item for item in load_question_pack(PACK)}
    tone = teacher_answer_to_row(
        "t1",
        {"ticket": "I am furious"},
        questions["tone"],
        {
            "type": "choice",
            "noul": None,
            "choice": "angry",
            "probabilities": {"calm": 0.05, "frustrated": 0.25, "angry": 0.70},
            "score_level": None,
            "rationale_internal": "drop me",
        },
        source="teacher:test",
    )
    assert tone.label.choice == "angry"
    assert tone.teacher_probs is not None
    assert abs(sum(tone.teacher_probs.values()) - 1.0) < 1e-9
    urgency = teacher_answer_to_row(
        "t1",
        {"ticket": "ASAP"},
        questions["urgency"],
        {
            "type": "score",
            "noul": None,
            "choice": None,
            "probabilities": {"can wait": 0.05, "this week": 0.25, "today": 0.70},
            "score_level": "today",
            "rationale_internal": "drop me",
        },
        source="teacher:test",
    )
    assert urgency.label.level == "today"
    assert urgency.label.index == 2


def test_extract_json_object_strips_fence() -> None:
    payload = extract_json_object('```json\n{"type": "noul", "noul": 0.5}\n```')
    assert payload["noul"] == 0.5


def test_dry_run_writes_three_rows(tmp_path: Path) -> None:
    out = tmp_path / "teacher.jsonl"
    rows = run_distill(
        questions_path=PACK,
        out_path=out,
        input_path=RAW,
        max_rows=100,
        model="gpt-4o-mini",
        api_key=None,
    )
    assert len(rows) == 3
    dataset = DecisionDataset.from_jsonl(out)
    assert len(dataset) == 3
    assert {row.question.type for row in dataset} == {"noul", "choice", "score"}
    for row in dataset:
        dumped = row.model_dump()
        assert "rationale_internal" not in dumped
        assert row.source.endswith("dry-run")


def test_distill_rows_respects_max_rows() -> None:
    questions = load_question_pack(PACK)
    tickets = load_tickets(RAW)
    rows = distill_rows(
        tickets,
        questions,
        max_rows=2,
        source="teacher:dry-run",
        teacher=None,
    )
    assert len(rows) == 2


def test_fake_payload_includes_then_drops_rationale() -> None:
    questions = load_question_pack(PACK)
    payload = fake_teacher_payload({"ticket": "I was charged twice ASAP"}, questions[0])
    assert "rationale_internal" in payload
    row = teacher_answer_to_row(
        "x",
        {"ticket": "I was charged twice ASAP"},
        questions[0],
        payload,
        source="teacher:dry-run",
    )
    assert "rationale" not in row.model_dump()


def test_teacher_prompt_forces_json() -> None:
    questions = load_question_pack(PACK)
    prompt = teacher_user_prompt({"ticket": "hello"}, questions[1])
    assert "angry" in prompt
    assert "rationale_internal" in prompt


def test_infer_has_no_openai_dependency() -> None:
    import kev.infer as infer

    source = inspect.getsource(infer)
    assert "openai" not in source.lower()
    assert "chat.completions" not in source
    assert "OPENAI_API_KEY" not in source


def test_chat_teacher_json_uses_httpx_not_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, object]:
            payload = {
                "type": "noul",
                "noul": 0.7,
                "choice": None,
                "probabilities": None,
                "score_level": None,
                "rationale_internal": "ignore",
            }
            return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def post(
            self, url: str, headers: dict[str, str], json: dict[str, object]
        ) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    result = chat_teacher_json(
        base_url="http://teacher.test/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "label"}],
    )
    assert result["noul"] == 0.7
    assert captured["url"] == "http://teacher.test/v1/chat/completions"
    body = captured["json"]
    assert isinstance(body, dict)
    assert body["response_format"]["type"] == "json_schema"
