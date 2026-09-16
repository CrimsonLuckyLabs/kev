"""Inference control flow. Code owns the loop; Kev only judges.

Many questions share one state encode / one prefix prefill.
Python assembles answers from next-token logits. Never model.generate().
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from kev.backends.base import Backend
from kev.calibration import apply_temperature
from kev.encode import estimate_request_tokens, state_to_text
from kev.primitives import Choice, Noul, Question, Score, parse_questions
from kev.prompt import ChatFormat, Qwen25ChatFormat
from kev.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    State,
    SystemOneResponse,
    Usage,
    entropy_confidence,
    nearest_level,
    new_request_id,
)

logger = logging.getLogger("kev.infer")

DEFAULT_MAX_STATE_CHARS = 120_000
YES_VARIANTS = ("YES", "Yes", "yes")
NO_VARIANTS = ("NO", "No", "no")


class LogitsEngine(Protocol):
    """Next-token scoring engine. Implementations must not decode or generate()."""

    def prefill(self, prefix_text: str) -> PrefillCache: ...

    def next_logits(self, cache: PrefillCache, suffix_text: str) -> Sequence[float]: ...

    def next_logits_batch(
        self, cache: PrefillCache, suffixes: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...

    def continuation_logprob(
        self, cache: PrefillCache, suffix_text: str, continuation: str
    ) -> float: ...

    def encode(self, text: str) -> list[int]: ...


@dataclass
class PrefillCache:
    """Shared prefix KV / handle. Suffix forwards must not mutate this for later questions."""

    prefix_text: str
    n_tokens: int
    payload: Any = None


@dataclass
class ScoredQuestion:
    question_id: str
    question: Noul | Choice | Score
    suffix: str
    token_ids: list[int] = field(default_factory=list)


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(float(logit) - peak) for logit in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def first_token_id(engine: LogitsEngine, text: str) -> int:
    ids = engine.encode(text)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for {text!r}")
    return ids[0]


def unique_first_token_ids(engine: LogitsEngine, variants: Sequence[str]) -> list[int]:
    seen: list[int] = []
    for variant in variants:
        token_id = first_token_id(engine, variant)
        if token_id not in seen:
            seen.append(token_id)
    return seen


def noul_from_logits(
    logits: Sequence[float], yes_ids: Sequence[int], no_ids: Sequence[int]
) -> float:
    """Sum vocab softmax mass for YES*/NO* variants, then renormalize to P(YES)."""
    probs = softmax(logits)
    p_yes = 0.0
    p_no = 0.0
    vocab = len(probs)
    for token_id in yes_ids:
        if 0 <= token_id < vocab:
            p_yes += probs[token_id]
    for token_id in no_ids:
        if 0 <= token_id < vocab:
            p_no += probs[token_id]
    total = p_yes + p_no
    if total <= 0.0:
        return 0.5
    return p_yes / total


def _option_first_ids(engine: LogitsEngine, options: Sequence[str]) -> list[int]:
    return [first_token_id(engine, option) for option in options]


def has_first_token_collision(engine: LogitsEngine, options: Sequence[str]) -> bool:
    ids = _option_first_ids(engine, options)
    return len(set(ids)) != len(ids)


def option_distribution(
    engine: LogitsEngine,
    cache: PrefillCache,
    suffix: str,
    options: Sequence[str],
    logits: Sequence[float] | None = None,
) -> tuple[list[float], str, list[int]]:
    """Return (probabilities, method, scored_token_ids).

    Default: first-token logits of each option key, softmax over K.
    Fallback: if two options share a first token, teacher-force the FULL option
    string and softmax length-normalized logprobs. Invalid labels stay impossible.
    """
    first_ids = _option_first_ids(engine, options)
    if len(set(first_ids)) == len(first_ids):
        if logits is None:
            logits = engine.next_logits(cache, suffix)
        scores = [
            float(logits[token_id]) if token_id < len(logits) else float("-inf")
            for token_id in first_ids
        ]
        return softmax(scores), "first_token", first_ids

    logprobs = [
        engine.continuation_logprob(cache, suffix, option) for option in options
    ]
    token_ids: list[int] = []
    for option in options:
        token_ids.extend(engine.encode(option))
    logger.info(
        "first-token collision; teacher-forcing full option strings token_ids=%s",
        token_ids,
    )
    return softmax(logprobs), "teacher_force", token_ids


def assemble_noul(probability: float) -> NoulAnswer:
    return NoulAnswer(noul=min(1.0, max(0.0, probability)))


def assemble_choice(options: Sequence[str], probabilities: Sequence[float]) -> ChoiceAnswer:
    mapping = {option: float(prob) for option, prob in zip(options, probabilities, strict=True)}
    choice = max(mapping, key=mapping.__getitem__)
    return ChoiceAnswer(
        choice=choice,
        probabilities=mapping,
        confidence=entropy_confidence(list(probabilities)),
    )


def assemble_score(levels: Sequence[str], probabilities: Sequence[float]) -> ScoreAnswer:
    expected = sum(index * float(prob) for index, prob in enumerate(probabilities))
    mapping = {level: float(prob) for level, prob in zip(levels, probabilities, strict=True)}
    return ScoreAnswer(
        score=expected,
        levels=list(levels),
        probabilities=mapping,
        confidence=entropy_confidence(list(probabilities)),
        level=nearest_level(expected, list(levels)),
    )


def score_questions(
    engine: LogitsEngine,
    state_text: str,
    questions: Mapping[str, Noul | Choice | Score],
    chat_format: ChatFormat | None = None,
    *,
    batch: bool = True,
) -> tuple[dict[str, Answer], int, list[int]]:
    """Prefill the shared prefix once, then one-step (optionally batched) suffix forwards.

    Returns answers, input token count, and the token ids that were scored.
    output_tokens is always 0 — this function never decodes.
    """
    fmt = chat_format or Qwen25ChatFormat()
    prefix = fmt.prefix(state_text)
    started = time.perf_counter()
    cache = engine.prefill(prefix)
    rows = [
        ScoredQuestion(question_id=qid, question=question, suffix=fmt.suffix_for(question))
        for qid, question in questions.items()
    ]
    suffixes = [row.suffix for row in rows]
    if batch:
        batched_logits = engine.next_logits_batch(cache, suffixes)
    else:
        batched_logits = [engine.next_logits(cache, suffix) for suffix in suffixes]

    yes_ids = unique_first_token_ids(engine, YES_VARIANTS)
    no_ids = unique_first_token_ids(engine, NO_VARIANTS)
    answers: dict[str, Answer] = {}
    scored_ids: list[int] = []
    input_tokens = cache.n_tokens
    temperature = float(getattr(engine, "temperature", 1.0) or 1.0)

    for row, logits in zip(rows, batched_logits, strict=True):
        if temperature != 1.0:
            logits = apply_temperature(logits, temperature)
        question = row.question
        if isinstance(question, Noul):
            noul = noul_from_logits(logits, yes_ids, no_ids)
            answers[row.question_id] = assemble_noul(noul)
            row.token_ids = list(yes_ids) + list(no_ids)
        elif isinstance(question, Choice):
            options = question.options
            probs, _method, token_ids = option_distribution(
                engine, cache, row.suffix, options, logits=logits
            )
            answers[row.question_id] = assemble_choice(options, probs)
            row.token_ids = token_ids
        else:
            levels = question.levels
            probs, _method, token_ids = option_distribution(
                engine, cache, row.suffix, levels, logits=logits
            )
            answers[row.question_id] = assemble_score(levels, probs)
            row.token_ids = token_ids
        scored_ids.extend(row.token_ids)
        try:
            input_tokens += len(engine.encode(row.suffix))
        except Exception:
            input_tokens += max(1, len(row.suffix) // 4)
        logger.info(
            "scored question_id=%s token_ids=%s latency_ms=%.3f",
            row.question_id,
            row.token_ids,
            (time.perf_counter() - started) * 1000.0,
        )

    logger.info(
        "prefill_tokens=%s scored_token_ids=%s latency_ms=%.3f",
        cache.n_tokens,
        scored_ids,
        (time.perf_counter() - started) * 1000.0,
    )
    return answers, input_tokens, scored_ids


def system_one(
    state: State,
    questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
    *,
    backend: Backend,
    model: str = "kev-latest",
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
) -> SystemOneResponse:
    parsed = parse_questions(questions)
    text = state_to_text(state)
    if len(text) > max_state_chars:
        raise ValueError(f"state exceeds max_state_chars ({max_state_chars})")
    started = time.perf_counter()
    answers = backend.infer(text, parsed)
    latency_ms = (time.perf_counter() - started) * 1000.0
    input_tokens = getattr(backend, "last_input_tokens", None)
    if not isinstance(input_tokens, int) or input_tokens < 0:
        input_tokens = estimate_request_tokens(text, parsed)
    from kev.backends import display_model_id

    return SystemOneResponse(
        id=new_request_id(),
        model=display_model_id(model),
        answers=answers,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=0,
            latency_ms=latency_ms,
        ),
    )
