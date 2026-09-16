from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from kev.types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    entropy_confidence,
    shannon_entropy,
)


def test_noul_is_probability_of_true() -> None:
    answer = NoulAnswer(noul=0.73)
    assert answer.type == "noul"
    assert answer.noul == 0.73
    assert not hasattr(answer, "confidence") or "confidence" not in answer.model_fields


def test_noul_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        NoulAnswer(noul=1.2)
    with pytest.raises(ValidationError):
        NoulAnswer(noul=-0.01)


def test_choice_rejects_label_outside_option_set() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="nope",
            probabilities={"billing": 1.0},
            confidence=1.0,
        )


def test_choice_must_be_argmax() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="a",
            probabilities={"a": 0.1, "b": 0.9},
            confidence=0.5,
        )


def test_choice_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="a",
            probabilities={"a": 0.5, "b": 0.6},
            confidence=0.1,
        )


def test_entropy_confidence_uniform_is_zero() -> None:
    assert entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert entropy_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)


def test_entropy_confidence_one_hot_is_one() -> None:
    assert entropy_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_entropy_confidence_matches_formula() -> None:
    probs = [0.8, 0.15, 0.05]
    expected = 1.0 - shannon_entropy(probs) / math.log(3)
    assert entropy_confidence(probs) == pytest.approx(min(1.0, max(0.0, expected)))


def test_entropy_confidence_singleton_is_one() -> None:
    assert entropy_confidence([1.0]) == 1.0


def test_score_is_expected_level_index() -> None:
    answer = ScoreAnswer(
        score=0.7,
        levels=["low", "high"],
        probabilities={"low": 0.3, "high": 0.7},
        confidence=entropy_confidence([0.3, 0.7]),
    )
    assert answer.score == pytest.approx(0 * 0.3 + 1 * 0.7)
    assert 0.0 <= answer.score <= 1.0
    assert answer.level == "high"


def test_score_rejects_mismatched_level_keys() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=0.0,
            levels=["low", "high"],
            probabilities={"low": 1.0},
            confidence=1.0,
        )


def test_score_rejects_wrong_expected_value() -> None:
    with pytest.raises(ValidationError):
        ScoreAnswer(
            score=0.9,
            levels=["a", "b"],
            probabilities={"a": 1.0, "b": 0.0},
            confidence=1.0,
        )


def test_system_one_response_convenience_views() -> None:
    response = SystemOneResponse(
        answers={
            "refund": NoulAnswer(noul=0.9),
            "topic": ChoiceAnswer(
                choice="billing",
                probabilities={"billing": 1.0, "other": 0.0},
                confidence=1.0,
            ),
            "urgency": ScoreAnswer(
                score=2.0,
                levels=["can wait", "this week", "today"],
                probabilities={"can wait": 0.0, "this week": 0.0, "today": 1.0},
                confidence=1.0,
            ),
        },
        usage=Usage(input_tokens=12, output_tokens=0, latency_ms=1.5),
        id="kev_req1",
        model="kev-latest",
    )
    assert set(response.nouls) == {"refund"}
    assert set(response.choices) == {"topic"}
    assert set(response.scores) == {"urgency"}
    assert response.usage.output_tokens == 0
    assert response.model == "kev-latest"
    assert response.request_id == "kev_req1"
    dumped = response.model_dump()
    assert dumped["id"] == "kev_req1"
    assert "nouls" not in dumped
    assert "model" not in dumped["usage"]
