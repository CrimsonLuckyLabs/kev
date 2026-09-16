"""ChatML prompt templates. Swap formats for Qwen2.5 vs Qwen3. Never a chat completion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kev.encode import state_to_text
from kev.primitives import Choice, Noul, Score
from kev.types import State

serialize_state = state_to_text

SYSTEM_TEXT = "You are Kev, a decision model. You do not write. You only score options."
STATE_USER_TAIL = "You will be asked one atomic question about this state."

_YES_KEYS = ("yes", "true", "YES", "True", "1")
_NO_KEYS = ("no", "false", "NO", "False", "0")


def noul_pole(criteria: Mapping[str, Any] | None, pole: str) -> str | None:
    """Optional YES/NO rubric from Noul.criteria. Map keys are not used."""
    if not criteria:
        return None
    names = _YES_KEYS if pole == "yes" else _NO_KEYS
    for key in names:
        raw = criteria.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


class ChatFormat:
    """Qwen-style ChatML. Subclasses may change markers; they must not add thinking text."""

    name = "qwen2.5"
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    def _turn(self, role: str, content: str) -> str:
        return f"{self.im_start}{role}\n{content}\n{self.im_end}\n"

    def assistant_open(self) -> str:
        return f"{self.im_start}assistant\n"

    def prefix(self, state_text: str) -> str:
        user = f"STATE:\n{state_text}\n\n{STATE_USER_TAIL}"
        return self._turn("system", SYSTEM_TEXT) + self._turn("user", user)

    def noul_suffix(self, question: Noul) -> str:
        lines = [f"QUESTION:\n{question.instructions}"]
        yes = noul_pole(question.criteria, "yes")
        no = noul_pole(question.criteria, "no")
        if yes:
            lines.append(f"YES: {yes}")
        if no:
            lines.append(f"NO: {no}")
        lines.append("Answer with a single token: YES or NO.")
        return self._turn("user", "\n".join(lines)) + self.assistant_open()

    def choice_suffix(self, question: Choice) -> str:
        lines = [f"QUESTION:\n{question.instructions}", "Options:"]
        for option, description in question.criteria.items():
            if description:
                lines.append(f"- {option}: {description}")
            else:
                lines.append(f"- {option}")
        lines.append("Answer with the option key as a single token / short identifier.")
        return self._turn("user", "\n".join(lines)) + self.assistant_open()

    def score_suffix(self, question: Score) -> str:
        lines = [f"QUESTION:\n{question.instructions}", "Options:"]
        for level in question.levels:
            lines.append(f"- {level}")
        lines.append("Answer with the option key as a single token / short identifier.")
        return self._turn("user", "\n".join(lines)) + self.assistant_open()

    def suffix_for(self, question: Noul | Choice | Score) -> str:
        if isinstance(question, Noul):
            return self.noul_suffix(question)
        if isinstance(question, Choice):
            return self.choice_suffix(question)
        return self.score_suffix(question)


class Qwen25ChatFormat(ChatFormat):
    name = "qwen2.5"


class Qwen3ChatFormat(ChatFormat):
    """Qwen3 Instruct ChatML. No <think> block — the next token must be the decision."""

    name = "qwen3"


def get_chat_format(name: str | None = None) -> ChatFormat:
    key = (name or "qwen2.5").lower().replace("_", "-")
    if "qwen3" in key:
        return Qwen3ChatFormat()
    return Qwen25ChatFormat()


def render_prefill(state_text: str, chat_format: ChatFormat | None = None) -> str:
    """Shared state prefix. Questions attach as suffixes; they do not start generate()."""
    fmt = chat_format or Qwen25ChatFormat()
    return fmt.prefix(state_text)


def render_question_text(question: Noul | Choice | Score) -> str:
    """Developer-facing question text for token estimates, not a chat prompt."""
    if isinstance(question, Noul):
        return f"noul: {question.instructions}"
    if isinstance(question, Choice):
        options = ", ".join(question.options)
        return f"choice: {question.instructions} [{options}]"
    levels = ", ".join(question.levels)
    return f"score: {question.instructions} [{levels}]"


def render_state(state: State) -> str:
    return state_to_text(state)
