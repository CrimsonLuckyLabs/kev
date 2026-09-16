"""In-process HTTP caps. No extra services. Env can loosen or disable (0)."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_JUDGE_PER_MIN = 20
DEFAULT_LOGIN_PER_MIN = 8
DEFAULT_WINDOW_S = 60.0
DEFAULT_MAX_INFLIGHT = 1
DEFAULT_MAX_QUESTIONS = 16
DEFAULT_MAX_OPTIONS = 32
DEFAULT_HTTP_STATE_CHARS = 24_000
MAX_TRACKED_KEYS = 20_000


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def http_state_chars() -> int:
    return env_int("KEV_MAX_STATE_CHARS", DEFAULT_HTTP_STATE_CHARS)


def http_limits() -> dict[str, int]:
    return {
        "judge_per_min": env_int("KEV_JUDGE_PER_MIN", DEFAULT_JUDGE_PER_MIN),
        "login_per_min": env_int("KEV_LOGIN_PER_MIN", DEFAULT_LOGIN_PER_MIN),
        "max_inflight": env_int("KEV_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT),
        "max_questions": env_int("KEV_MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS),
        "max_options": env_int("KEV_MAX_OPTIONS", DEFAULT_MAX_OPTIONS),
        "max_state_chars": http_state_chars(),
    }


def option_count(question: object) -> int:
    if isinstance(question, Mapping):
        kind = question.get("type")
        criteria = question.get("criteria")
    else:
        kind = getattr(question, "type", None)
        criteria = getattr(question, "criteria", None)
    if kind == "choice" and isinstance(criteria, dict):
        return len(criteria)
    if kind == "score" and isinstance(criteria, list):
        return len(criteria)
    return 0


def payload_error(questions: object) -> str | None:
    if not isinstance(questions, dict):
        return "questions must be an object"
    max_questions = env_int("KEV_MAX_QUESTIONS", DEFAULT_MAX_QUESTIONS)
    if max_questions and len(questions) > max_questions:
        return f"at most {max_questions} questions"
    max_options = env_int("KEV_MAX_OPTIONS", DEFAULT_MAX_OPTIONS)
    if not max_options:
        return None
    for spec in questions.values():
        count = option_count(spec)
        if count > max_options:
            return f"at most {max_options} options per choice/score"
    return None


@dataclass
class WindowCounter:
    """Sliding window. limit 0 means unlimited."""

    _hits: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(
        self,
        key: str,
        limit: int,
        window_s: float = DEFAULT_WINDOW_S,
        now: float | None = None,
    ) -> tuple[bool, int]:
        if limit <= 0:
            return True, 0
        stamp = now if now is not None else time.time()
        cutoff = stamp - window_s
        with self._lock:
            hits = [ts for ts in self._hits.get(key, []) if ts > cutoff]
            if len(hits) >= limit:
                self._hits[key] = hits
                retry = int(max(1, hits[0] + window_s - stamp))
                return False, retry
            hits.append(stamp)
            self._hits[key] = hits
            extra = len(self._hits) - MAX_TRACKED_KEYS
            for _ in range(max(0, extra)):
                self._hits.pop(next(iter(self._hits)))
            return True, 0


@dataclass
class Inflight:
    """Non-blocking GPU slot. n 0 means unlimited."""

    n: int
    _sem: threading.Semaphore | None = field(init=False)

    def __post_init__(self) -> None:
        self._sem = threading.Semaphore(self.n) if self.n > 0 else None

    def acquire(self) -> bool:
        if self._sem is None:
            return True
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        if self._sem is not None:
            self._sem.release()
