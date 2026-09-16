from __future__ import annotations

import json

import pytest

from kev.encode import state_to_text
from kev.primitives import Choice, Noul, Score
from kev.prompt import Qwen25ChatFormat, noul_pole, serialize_state


def test_string_state_passes_through() -> None:
    assert serialize_state("hello") == "hello"


def test_dict_state_is_pretty_json() -> None:
    state = {"ticket": "charged twice", "id": 104}
    rendered = serialize_state(state)
    assert rendered == json.dumps(state, indent=2, ensure_ascii=False)
    assert "\n" in rendered


def test_list_state_is_pretty_json() -> None:
    state = ["a", {"b": 1}]
    assert serialize_state(state) == json.dumps(state, indent=2, ensure_ascii=False)


def test_state_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        serialize_state(42)  # type: ignore[arg-type]


def test_state_to_text_matches_serialize_state() -> None:
    state = {"ticket": "charged twice", "id": 104}
    assert state_to_text(state) == serialize_state(state)


def test_adding_questions_does_not_change_state_text() -> None:
    state = {"doc": "hello"}
    first = serialize_state(state)
    second = serialize_state(state)
    assert first == second


def test_suffix_does_not_include_question_id() -> None:
    fmt = Qwen25ChatFormat()
    noul = Noul(instructions="Is this about billing?")
    suffix = fmt.suffix_for(noul)
    assert "QUESTION (billing)" not in suffix
    assert "Is this about billing?" in suffix
    assert "YES or NO" in suffix
    other = Noul(instructions="Is this about billing?")
    assert fmt.suffix_for(noul) == fmt.suffix_for(other)


def test_noul_suffix_uses_true_false_criteria() -> None:
    fmt = Qwen25ChatFormat()
    question = Noul(
        instructions="Does this convey urgency?",
        criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    )
    suffix = fmt.suffix_for(question)
    assert "YES: Explicitly time-sensitive" in suffix
    assert "NO: No urgency expressed" in suffix
    assert noul_pole(question.criteria, "yes") == "Explicitly time-sensitive"
    assert noul_pole(question.criteria, "no") == "No urgency expressed"


def test_choice_and_score_suffix_omit_map_keys() -> None:
    fmt = Qwen25ChatFormat()
    choice = Choice(
        instructions="What is the tone?",
        criteria={"calm": "neutral", "angry": None},
    )
    text = fmt.suffix_for(choice)
    assert "QUESTION (tone)" not in text
    assert "- calm: neutral" in text
    assert "- angry" in text
    score = Score(instructions="How soon?", criteria=["later", "now"])
    scored = fmt.suffix_for(score)
    assert "QUESTION (urgency)" not in scored
    assert "- later" in scored
    assert "- now" in scored
