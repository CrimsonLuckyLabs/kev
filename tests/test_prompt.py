from __future__ import annotations

import json

import pytest

from kev.encode import state_to_text
from kev.prompt import serialize_state


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
