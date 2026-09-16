"""State text and token estimates. The model never sees raw Python objects."""

from __future__ import annotations

import json
from collections.abc import Mapping

from kev.primitives import Choice, Noul, Score
from kev.types import State


def state_to_text(state: State) -> str:
    """Normalize state to the string that every backend prefills once."""
    if isinstance(state, str):
        return state
    if isinstance(state, (dict, list)):
        return json.dumps(state, indent=2, ensure_ascii=False)
    raise TypeError(f"state must be str | dict | list, got {type(state).__name__}")


def estimate_tokens(text: str) -> int:
    """Character/4 estimate used when a real tokenizer is not loaded."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_request_tokens(
    state_text: str,
    questions: Mapping[str, Noul | Choice | Score],
) -> int:
    from kev.prompt import render_question_text

    total = estimate_tokens(state_text)
    for question in questions.values():
        total += estimate_tokens(render_question_text(question))
    return total
