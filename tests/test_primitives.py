from __future__ import annotations

import pytest

from kev.primitives import (
    Choice,
    Noul,
    Score,
    dump_questions,
    parse_question,
    parse_questions,
    question_fingerprint,
)


def test_noul_defaults_type() -> None:
    question = Noul(instructions="Is this about billing?")
    assert question.type == "noul"
    assert question.criteria is None


def test_noul_accepts_criteria_dict() -> None:
    question = Noul(
        instructions="Is this about billing?",
        criteria={"true": "money movement", "false": "anything else"},
    )
    assert question.criteria == {"true": "money movement", "false": "anything else"}


def test_noul_rejects_empty_instructions() -> None:
    with pytest.raises(ValueError, match="empty instructions"):
        Noul(instructions="")


def test_choice_options_are_criteria_keys() -> None:
    question = Choice(
        instructions="Department?",
        criteria={"billing": "money", "technical": None},
    )
    assert question.type == "choice"
    assert question.options == ["billing", "technical"]
    assert question.max_options == 255


def test_choice_accepts_list_criteria() -> None:
    question = Choice(instructions="Pick one", criteria=["a", "b", "c"])
    assert question.options == ["a", "b", "c"]
    assert question.criteria == {"a": None, "b": None, "c": None}


def test_choice_rejects_empty_criteria() -> None:
    with pytest.raises(ValueError, match="at least 2 options"):
        Choice(instructions="Pick one", criteria={})


def test_score_levels_preserve_order() -> None:
    question = Score(instructions="Urgency?", criteria=["low", "mid", "high"])
    assert question.type == "score"
    assert question.levels == ["low", "mid", "high"]


def test_score_rejects_duplicate_levels() -> None:
    with pytest.raises(ValueError, match="unique"):
        Score(instructions="Urgency?", criteria=["low", "low"])


def test_parse_question_from_dict() -> None:
    noul = parse_question({"type": "noul", "instructions": "Yes?"})
    choice = parse_question(
        {"type": "choice", "instructions": "Which?", "criteria": {"x": None, "y": None}}
    )
    score = parse_question(
        {"type": "score", "instructions": "How much?", "criteria": ["a", "b"]}
    )
    assert isinstance(noul, Noul)
    assert isinstance(choice, Choice)
    assert isinstance(score, Score)


def test_parse_questions_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        parse_questions({})


def test_unknown_type_is_invalid() -> None:
    with pytest.raises(ValueError, match="unknown question type: 'essay'"):
        parse_question({"type": "essay", "instructions": "Write a paragraph"})


def test_missing_type_is_invalid() -> None:
    with pytest.raises(ValueError, match="unknown question type"):
        parse_question({"instructions": "Yes?"})


def test_choice_rejects_empty_option_label() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Choice(instructions="Pick one", criteria={"": None, "ok": None})


def test_fingerprint_ignores_python_identity() -> None:
    a = Noul(instructions="Same question")
    b = Noul(instructions="Same question")
    assert question_fingerprint(a) == question_fingerprint(b)


def test_dump_questions_is_wire_json() -> None:
    dumped = dump_questions(
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
        }
    )
    assert dumped["billing"] == {
        "type": "noul",
        "instructions": "Is this about billing?",
    }
    assert dumped["tone"]["criteria"] == {"calm": None, "frustrated": None, "angry": None}
    assert "max_options" not in dumped["tone"]
    assert dumped["urgency"]["criteria"] == ["can wait", "this week", "today"]
