"""Typed System One answers. Python assembles these from head outputs."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kev.primitives import Choice, Noul, Score

State = str | dict[str, Any] | list[Any]
QuestionValue = Annotated[Noul | Choice | Score, Field(discriminator="type")]

PROB_SUM_TOL = 1e-6


def shannon_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy H(p) in nats. Zero-probability bins contribute 0."""
    entropy = 0.0
    for probability in probs:
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def entropy_confidence(probs: Sequence[float]) -> float:
    """1 - H(p) / log(K), clamped to [0, 1]. K is the number of options/levels."""
    count = len(probs)
    if count <= 1:
        return 1.0
    total = float(sum(probs))
    if total <= 0.0:
        return 0.0
    normalized = [float(probability) / total for probability in probs]
    confidence = 1.0 - shannon_entropy(normalized) / math.log(count)
    return min(1.0, max(0.0, confidence))


class Usage(BaseModel):
    """Token and latency accounting. System One output tokens are always 0."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)


def nearest_level(score: float, levels: Sequence[str]) -> str:
    index = int(round(score))
    index = max(0, min(len(levels) - 1, index))
    return str(levels[index])


def new_request_id() -> str:
    return f"kev_{uuid.uuid4().hex}"


class NoulAnswer(BaseModel):
    """P(statement is true). The noul value is the confidence signal; there is no extra field."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    """Closed-set pick plus the full distribution. `choice` is always a supplied option."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("Choice probabilities must include at least one option")
        if any(probability < 0.0 for probability in value.values()):
            raise ValueError("Choice probabilities must be non-negative")
        total = float(sum(value.values()))
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise ValueError("Choice probabilities must sum to 1")
        return value

    @model_validator(mode="after")
    def _choice_is_argmax(self) -> ChoiceAnswer:
        if self.choice not in self.probabilities:
            raise ValueError("Choice label is not in the supplied option set")
        expected = max(self.probabilities, key=self.probabilities.__getitem__)
        if self.choice != expected:
            raise ValueError("Choice must be the argmax of probabilities")
        return self


class ScoreAnswer(BaseModel):
    """Expected rubric index in [0, n-1], plus the level distribution."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    score: float
    levels: list[str] = Field(min_length=1)
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    level: str | None = None

    @field_validator("probabilities")
    @classmethod
    def _validate_probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("Score probabilities must include at least one level")
        if any(probability < 0.0 for probability in value.values()):
            raise ValueError("Score probabilities must be non-negative")
        total = float(sum(value.values()))
        if abs(total - 1.0) > PROB_SUM_TOL:
            raise ValueError("Score probabilities must sum to 1")
        return value

    @model_validator(mode="after")
    def _score_matches_levels(self) -> ScoreAnswer:
        if len(self.levels) != len(set(self.levels)):
            raise ValueError("Score levels must be unique")
        if set(self.probabilities) != set(self.levels):
            raise ValueError("Score probability keys must match levels")
        n_levels = len(self.levels)
        if not 0.0 <= self.score <= float(n_levels - 1):
            raise ValueError("Score must lie in [0, n-1]")
        expected = sum(
            index * self.probabilities[level] for index, level in enumerate(self.levels)
        )
        if abs(self.score - expected) > 1e-6:
            raise ValueError("Score must be the expected value of the level index")
        nearest = nearest_level(self.score, self.levels)
        if self.level is None:
            self.level = nearest
        return self


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer,
    Field(discriminator="type"),
]


class SystemOneResponse(BaseModel):
    """Typed answers for one state. Convenience views split by primitive."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model: str
    answers: dict[str, Answer]
    usage: Usage

    @property
    def request_id(self) -> str:
        return self.id

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {
            name: answer for name, answer in self.answers.items() if isinstance(answer, NoulAnswer)
        }

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {
            name: answer
            for name, answer in self.answers.items()
            if isinstance(answer, ChoiceAnswer)
        }

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {
            name: answer for name, answer in self.answers.items() if isinstance(answer, ScoreAnswer)
        }


class SystemOneRequest(BaseModel):
    """Inbound System One call. `questions` is a developer-supplied closed world."""

    model_config = ConfigDict(extra="forbid")

    state: State
    questions: dict[str, QuestionValue] = Field(min_length=1)
    model: str = "kev-latest"
