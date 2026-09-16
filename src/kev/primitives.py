"""Question primitives. Only these three exist. Invalid labels cannot be invented later."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

MAX_CHOICE_OPTIONS = 255


def validation_to_value_error(exc: ValidationError) -> ValueError:
    messages: list[str] = []
    for error in exc.errors():
        msg = str(error.get("msg", "invalid"))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        messages.append(msg)
    return ValueError("; ".join(messages) if messages else str(exc))


class _ValueErrorModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as exc:
            raise validation_to_value_error(exc) from None


class Noul(_ValueErrorModel):
    """Yes/no question. The answer is P(true) in [0, 1]."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, Any] | None = None

    @field_validator("instructions")
    @classmethod
    def _nonempty_instructions(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty instructions")
        return value


class Choice(_ValueErrorModel):
    """Closed option set. Criteria keys are the only legal labels."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str | None]
    max_options: int = Field(default=MAX_CHOICE_OPTIONS, ge=2, le=MAX_CHOICE_OPTIONS)

    @field_validator("instructions")
    @classmethod
    def _nonempty_instructions(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty instructions")
        return value

    @field_validator("criteria", mode="before")
    @classmethod
    def _coerce_criteria(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            coerced: dict[str, None] = {}
            for item in value:
                key = str(item)
                if key in coerced:
                    raise ValueError("Choice criteria labels must be unique")
                coerced[key] = None
            return coerced
        return value

    @model_validator(mode="after")
    def _option_count(self) -> Choice:
        if any(not str(key).strip() for key in self.criteria):
            raise ValueError("Choice option labels must be non-empty")
        count = len(self.criteria)
        if count < 2:
            raise ValueError("Choice requires at least 2 options")
        if count > self.max_options:
            raise ValueError(f"Choice allows at most {self.max_options} options")
        return self

    @property
    def options(self) -> list[str]:
        return list(self.criteria.keys())


class Score(_ValueErrorModel):
    """Ordered rubric. Level 0 is first; score is E[index] in [0, n-1]."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]

    @field_validator("instructions")
    @classmethod
    def _nonempty_instructions(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty instructions")
        return value

    @field_validator("criteria")
    @classmethod
    def _valid_levels(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("Score requires at least 2 levels")
        if len(value) != len(set(value)):
            raise ValueError("Score criteria levels must be unique")
        if any(not level or not str(level).strip() for level in value):
            raise ValueError("Score criteria levels must be non-empty strings")
        return value

    @property
    def levels(self) -> list[str]:
        return list(self.criteria)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]
Questions = Mapping[str, Question]

_QUESTION_ADAPTER: TypeAdapter[Question] = TypeAdapter(Question)


def parse_question(value: Question | Mapping[str, Any]) -> Noul | Choice | Score:
    if isinstance(value, (Noul, Choice, Score)):
        return value
    if isinstance(value, Mapping):
        qtype = value.get("type")
        if qtype not in {"noul", "choice", "score"}:
            raise ValueError(
                f"unknown question type: {qtype!r}; only noul, choice, and score exist"
            )
    try:
        return _QUESTION_ADAPTER.validate_python(value)
    except ValidationError as exc:
        raise validation_to_value_error(exc) from None


def _sequence_items(questions: Sequence[Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for item in questions:
        if isinstance(item, Mapping) and "id" in item:
            body = {key: val for key, val in item.items() if key != "id"}
            items.append((str(item["id"]), body))
        elif isinstance(item, tuple) and len(item) == 2:
            items.append((str(item[0]), item[1]))
        else:
            raise ValueError(
                "question sequence items must be (id, question) or include an id field"
            )
    return items


def parse_questions(
    questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
) -> dict[str, Noul | Choice | Score]:
    if isinstance(questions, Mapping):
        items = list(questions.items())
    elif isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)):
        items = _sequence_items(questions)
    else:
        raise ValueError("questions must be a mapping of id → question")

    if not items:
        raise ValueError("system_one requires at least one question")

    parsed: dict[str, Noul | Choice | Score] = {}
    for name, question in items:
        if not name or not str(name).strip():
            raise ValueError("Question IDs must be non-empty")
        if name in parsed:
            raise ValueError(f"duplicate question id: {name}")
        parsed[name] = parse_question(question)
    return parsed


def question_fingerprint(question: Noul | Choice | Score) -> str:
    """Content hash input for non-mock backends. IDs are for application code."""
    if isinstance(question, Noul):
        return f"noul|{question.instructions}|{question.criteria}"
    if isinstance(question, Choice):
        options = "|".join(f"{key}:{question.criteria[key]}" for key in question.criteria)
        return f"choice|{question.instructions}|{options}"
    levels = "|".join(question.criteria)
    return f"score|{question.instructions}|{levels}"


def dump_questions(
    questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
) -> dict[str, dict[str, Any]]:
    """JSON-ready question map for POST /v1/systemone."""
    parsed = parse_questions(questions)
    dumped: dict[str, dict[str, Any]] = {}
    for name, question in parsed.items():
        payload = question.model_dump(mode="json", exclude_none=True)
        payload.pop("max_options", None)
        dumped[name] = payload
    return dumped
