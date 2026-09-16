"""Backend protocol. Implementations return assembled answers, never generated text."""

from __future__ import annotations

from typing import Protocol

from kev.primitives import Choice, Noul, Score
from kev.types import Answer


class Backend(Protocol):
    name: str

    def infer(
        self,
        state_text: str,
        questions: dict[str, Noul | Choice | Score],
    ) -> dict[str, Answer]:
        """Judge every question against one already-serialized state string."""
        ...

    def close(self) -> None:
        ...
