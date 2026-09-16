"""Kev: a local System One decision engine. It does not chat."""

from kev.backends.mock import MockBackend
from kev.client import AsyncKevClient, KevClient
from kev.primitives import Choice, Noul, Question, Score
from kev.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    State,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    entropy_confidence,
)

__version__ = "0.1.0"
__all__ = [
    "Answer",
    "AsyncKevClient",
    "Choice",
    "ChoiceAnswer",
    "KevClient",
    "MockBackend",
    "Noul",
    "NoulAnswer",
    "Question",
    "Score",
    "ScoreAnswer",
    "State",
    "SystemOneRequest",
    "SystemOneResponse",
    "Usage",
    "entropy_confidence",
    "__version__",
]
