"""Teacher distillation. Teachers may emit JSON; Kev still never generates text.

Uses httpx against an OpenAI-compatible Chat Completions endpoint.
This module is not imported by kev.infer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kev.encode import state_to_text
from kev.primitives import Choice, Noul
from kev.train.dataset import DecisionDataset, DecisionRow, LabelSpec, QuestionSpec
from kev.types import State

logger = logging.getLogger("kev.distill")

TeacherKind = Literal["noul", "choice", "score"]

TEACHER_SYSTEM = """You label Kev System One questions.
Reply with a single JSON object. No markdown. No prose outside JSON.
The student model will never see rationale_internal and will never generate JSON.
Schema:
{
  "type": "noul" | "choice" | "score",
  "noul": 0-1 or null,
  "choice": string or null,
  "probabilities": object or null,
  "score_level": string or null,
  "rationale_internal": "short, discarded, never trained on"
}
Rules:
- type must match the question type.
- noul is P(true) in [0, 1].
- choice must be one of the supplied options.
- score_level must be one of the supplied levels.
- probabilities, when present, must cover only the supplied options or levels.
- rationale_internal is discarded and is never written to the training file.
"""

TEACHER_JSON_SCHEMA: dict[str, Any] = {
    "name": "kev_teacher_label",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["noul", "choice", "score"]},
            "noul": {"type": ["number", "null"]},
            "choice": {"type": ["string", "null"]},
            "probabilities": {
                "anyOf": [
                    {"type": "object", "additionalProperties": {"type": "number"}},
                    {"type": "null"},
                ]
            },
            "score_level": {"type": ["string", "null"]},
            "rationale_internal": {"type": "string"},
        },
        "required": [
            "type",
            "noul",
            "choice",
            "probabilities",
            "score_level",
            "rationale_internal",
        ],
    },
}


class TeacherAnswer(BaseModel):
    """Teacher JSON. rationale_internal is accepted then dropped."""

    model_config = ConfigDict(extra="ignore")

    type: TeacherKind
    noul: float | None = Field(default=None)
    choice: str | None = None
    probabilities: dict[str, float] | None = None
    score_level: str | None = None
    rationale_internal: str | None = None


class TicketRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    ticket: str | None = None
    state: State | None = None

    def as_state(self) -> State:
        if self.state is not None:
            return self.state
        if self.ticket is not None:
            extra = {
                key: value
                for key, value in self.model_dump().items()
                if key not in {"id", "ticket", "state"} and value is not None
            }
            payload: dict[str, Any] = {"ticket": self.ticket, **extra}
            return payload
        raise ValueError(f"ticket {self.id} needs ticket or state")


def load_question_pack(path: str | Path) -> list[QuestionSpec]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load a question pack") from exc
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("question pack must be a mapping")
    block = raw.get("questions", raw)
    specs: list[QuestionSpec] = []
    if isinstance(block, list):
        for item in block:
            if not isinstance(item, dict):
                raise ValueError("question list items must be mappings")
            specs.append(QuestionSpec.model_validate(item))
        return specs
    if isinstance(block, dict):
        for question_id, body in block.items():
            if not isinstance(body, dict):
                raise ValueError(f"question {question_id!r} must be a mapping")
            payload = {"id": str(question_id), **body}
            specs.append(QuestionSpec.model_validate(payload))
        return specs
    raise ValueError("questions must be a mapping or a list")


def load_tickets(path: str | Path, limit: int | None = None) -> list[TicketRecord]:
    tickets: list[TicketRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                tickets.append(TicketRecord.model_validate(json.loads(text)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if limit is not None and len(tickets) >= limit:
                break
    if not tickets:
        raise ValueError(f"no tickets in {path}")
    return tickets


def default_dry_run_tickets() -> list[TicketRecord]:
    return [
        TicketRecord(id="dry-001", ticket="I was charged twice. Please refund ASAP."),
        TicketRecord(id="dry-002", ticket="The app crashed once yesterday. No rush."),
        TicketRecord(
            id="dry-003",
            state={"ticket": "Need a quote for more seats next quarter."},
        ),
    ]


def completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def teacher_user_prompt(state: State, question: QuestionSpec) -> str:
    primitive = question.as_primitive()
    lines = [
        "STATE:",
        state_to_text(state),
        "",
        f"QUESTION id={question.id} type={question.type}",
        question.instructions,
    ]
    if isinstance(primitive, Noul):
        lines.append("Return noul = P(true) in [0, 1].")
    elif isinstance(primitive, Choice):
        lines.append("Options (choice must be one of these):")
        for option in primitive.options:
            desc = primitive.criteria.get(option)
            if desc:
                lines.append(f"- {option}: {desc}")
            else:
                lines.append(f"- {option}")
        lines.append("Fill probabilities over those option keys.")
    else:
        lines.append("Levels in order (score_level must be one of these):")
        for index, level in enumerate(primitive.levels):
            lines.append(f"- {index}: {level}")
        lines.append("Fill probabilities over those level keys.")
    lines.append('Include rationale_internal, which will be discarded.')
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("teacher response was not JSON")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("teacher JSON must be an object")
    return payload


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _align_probs(
    names: Sequence[str],
    raw: Mapping[str, float] | None,
    winner: str,
) -> dict[str, float]:
    scores = [max(0.0, float((raw or {}).get(name, 0.0))) for name in names]
    total = float(sum(scores))
    if raw is None or total <= 0.0:
        return {name: (1.0 if name == winner else 0.0) for name in names}
    return {name: score / total for name, score in zip(names, scores, strict=True)}


def teacher_answer_to_row(
    ticket_id: str,
    state: State,
    question: QuestionSpec,
    payload: Mapping[str, Any],
    *,
    source: str,
) -> DecisionRow:
    answer = TeacherAnswer.model_validate(payload)
    if answer.type != question.type:
        raise ValueError(
            f"teacher type {answer.type!r} does not match question {question.type!r}"
        )
    primitive = question.as_primitive()
    if isinstance(primitive, Noul):
        if answer.noul is None:
            raise ValueError("noul teacher JSON missing noul")
        p_yes = _clip01(answer.noul)
        row = DecisionRow(
            id=f"{ticket_id}-{question.id}",
            state=state,
            question=question,
            label=LabelSpec(noul=1.0 if p_yes >= 0.5 else 0.0),
            teacher_p=p_yes,
            source=source,
        )
    elif isinstance(primitive, Choice):
        options = primitive.options
        choice = answer.choice
        if choice not in options:
            probs = answer.probabilities or {}
            ranked = [name for name in options if name in probs]
            choice = max(ranked, key=lambda name: float(probs[name]), default="")
        if choice not in options:
            raise ValueError("teacher choice is not a supplied option")
        row = DecisionRow(
            id=f"{ticket_id}-{question.id}",
            state=state,
            question=question,
            label=LabelSpec(choice=choice),
            teacher_probs=_align_probs(options, answer.probabilities, choice),
            source=source,
        )
    else:
        levels = primitive.levels
        level = answer.score_level
        if level not in levels:
            probs = answer.probabilities or {}
            ranked = [name for name in levels if name in probs]
            level = max(ranked, key=lambda name: float(probs[name]), default="")
        if level not in levels:
            raise ValueError("teacher score_level is not a supplied level")
        row = DecisionRow(
            id=f"{ticket_id}-{question.id}",
            state=state,
            question=question,
            label=LabelSpec(level=level, index=levels.index(level)),
            teacher_probs=_align_probs(levels, answer.probabilities, level),
            source=source,
        )
    dumped = json.loads(row.model_dump_json())
    if "rationale_internal" in dumped or "rationale" in dumped:
        raise RuntimeError("rationale leaked into a training row")
    return row


def fake_teacher_payload(state: State, question: QuestionSpec) -> dict[str, Any]:
    """Keyword labels for dry-run. Not a model call."""
    text = state_to_text(state).lower()
    primitive = question.as_primitive()
    if isinstance(primitive, Noul):
        hit = any(token in text for token in ("charged", "refund", "invoice", "billing"))
        p_yes = 0.91 if hit else 0.12
        return {
            "type": "noul",
            "noul": p_yes,
            "choice": None,
            "probabilities": None,
            "score_level": None,
            "rationale_internal": "dry-run fake; discard",
        }
    if isinstance(primitive, Choice):
        if any(token in text for token in ("furious", "idiot", "twice")):
            winner = "angry" if "angry" in primitive.options else primitive.options[-1]
        elif any(token in text for token in ("annoying", "frustrated")):
            winner = (
                "frustrated" if "frustrated" in primitive.options else primitive.options[0]
            )
        else:
            winner = primitive.options[0]
        rest = 0.08
        peak = 1.0 - rest * (len(primitive.options) - 1)
        probs = {
            name: (peak if name == winner else rest) for name in primitive.options
        }
        return {
            "type": "choice",
            "noul": None,
            "choice": winner,
            "probabilities": probs,
            "score_level": None,
            "rationale_internal": "dry-run fake; discard",
        }
    if any(token in text for token in ("asap", "urgent", "down")):
        winner = "today" if "today" in primitive.levels else primitive.levels[-1]
    elif "week" in text and "this week" in primitive.levels:
        winner = "this week"
    else:
        winner = primitive.levels[0]
    rest = 0.08
    peak = 1.0 - rest * (len(primitive.levels) - 1)
    probs = {name: (peak if name == winner else rest) for name in primitive.levels}
    return {
        "type": "score",
        "noul": None,
        "choice": None,
        "probabilities": probs,
        "score_level": winner,
        "rationale_internal": "dry-run fake; discard",
    }


def chat_teacher_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST /v1/chat/completions. Prefer json_schema, fall back to json_object."""
    import httpx

    url = completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    schema_body = {
        "model": model,
        "temperature": 0,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": TEACHER_JSON_SCHEMA,
        },
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=schema_body)
        if response.status_code >= 400:
            fallback = dict(schema_body)
            fallback["response_format"] = {"type": "json_object"}
            response = client.post(url, headers=headers, json=fallback)
        if response.status_code >= 400:
            raise RuntimeError(f"teacher HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("teacher response missing message content") from exc
    if not isinstance(content, str):
        raise RuntimeError("teacher content must be a string")
    return extract_json_object(content)


def distill_rows(
    tickets: Sequence[TicketRecord],
    questions: Sequence[QuestionSpec],
    *,
    max_rows: int,
    source: str,
    teacher: Any | None = None,
) -> list[DecisionRow]:
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    if not questions:
        raise ValueError("need at least one question")
    rows: list[DecisionRow] = []
    for ticket in tickets:
        state = ticket.as_state()
        for question in questions:
            if len(rows) >= max_rows:
                return rows
            if teacher is None:
                payload = fake_teacher_payload(state, question)
            else:
                messages = [
                    {"role": "system", "content": TEACHER_SYSTEM},
                    {"role": "user", "content": teacher_user_prompt(state, question)},
                ]
                payload = teacher(messages=messages)
            row = teacher_answer_to_row(
                ticket.id,
                state,
                question,
                payload,
                source=source,
            )
            rows.append(row)
    return rows


def run_distill(
    *,
    questions_path: str | Path,
    out_path: str | Path,
    input_path: str | Path | None = None,
    max_rows: int = 100,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
) -> list[DecisionRow]:
    """Write teacher JSONL. Empty api_key → 3 fake rows, no HTTP."""
    questions = load_question_pack(questions_path)
    source = f"teacher:{model}"
    key = (api_key or "").strip()
    if not key:
        tickets = default_dry_run_tickets()
        if input_path is not None and Path(input_path).is_file():
            tickets = load_tickets(input_path)
        rows = distill_rows(
            tickets,
            questions,
            max_rows=min(3, max_rows),
            source=f"{source}:dry-run",
            teacher=None,
        )
        DecisionDataset(rows).to_jsonl(out_path)
        logger.info("dry-run rows=%s out=%s", len(rows), out_path)
        return rows

    if input_path is None:
        raise ValueError("--in is required when an API key is set")
    tickets = load_tickets(input_path)

    def teacher(*, messages: list[dict[str, str]]) -> dict[str, Any]:
        return chat_teacher_json(
            base_url=base_url,
            api_key=key,
            model=model,
            messages=messages,
        )

    rows = distill_rows(
        tickets,
        questions,
        max_rows=max_rows,
        source=source,
        teacher=teacher,
    )
    DecisionDataset(rows).to_jsonl(out_path)
    logger.info("wrote %s rows=%s", out_path, len(rows))
    return rows

