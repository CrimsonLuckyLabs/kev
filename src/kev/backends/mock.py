"""Deterministic fake probabilities. No GPU, no network, no generate()."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from kev.primitives import Choice, Noul, Score
from kev.types import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer, entropy_confidence


def _unit(material: str) -> float:
    """Map material to (0, 1) via SHA-256. No RNG."""
    digest = hashlib.sha256(material.encode()).digest()
    number = int.from_bytes(digest[:8], "big")
    return (number + 1) / float((1 << 64) + 1)


def _hash_material(
    state_text: str,
    question_id: str,
    instructions: str,
    extra: str = "",
    seed: int | None = None,
) -> str:
    parts = [state_text, question_id, instructions]
    if extra:
        parts.append(extra)
    if seed is not None:
        parts.append(f"seed={seed}")
    return "\0".join(parts)


def _softmax(logits: Sequence[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(logit - peak) for logit in logits]
    total = sum(exps)
    return [value / total for value in exps]


def peaked_distribution(units: Sequence[float]) -> list[float]:
    """Peaked but not one-hot. Every option receives mass."""
    count = len(units)
    if count == 0:
        raise ValueError("peaked_distribution requires at least one option")
    if count == 1:
        return [1.0]
    logits = [(unit - 0.5) * 4.0 for unit in units]
    winner = max(range(count), key=lambda index: (units[index], -index))
    logits[winner] += 3.0
    probs = _softmax(logits)
    floor = min(1e-3, 0.08 / count)
    remaining = 1.0 - floor * count
    mixed = [floor + remaining * probability for probability in probs]
    total = sum(mixed)
    return [value / total for value in mixed]


def mock_noul_probability(
    state_text: str,
    question_id: str,
    instructions: str,
    seed: int | None = None,
) -> float:
    unit = _unit(_hash_material(state_text, question_id, instructions, extra="noul", seed=seed))
    probability = 0.05 + 0.90 * unit
    asap_in_instructions = "asap" in instructions.lower()
    asap_in_state = "asap" in state_text.lower()
    urgency_like = (
        asap_in_instructions
        or "urgenc" in question_id.lower()
        or "urgenc" in instructions.lower()
    )
    if asap_in_instructions or (asap_in_state and urgency_like):
        probability = 0.70 + 0.25 * unit
    return min(0.949999, max(0.050001, probability))


class MockBackend:
    """Hash-seeded peaked distributions. Python still assembles the typed answer."""

    name = "mock"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def infer(
        self,
        state_text: str,
        questions: dict[str, Noul | Choice | Score],
    ) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        for question_id, question in questions.items():
            if isinstance(question, Noul):
                answers[question_id] = self._noul(state_text, question_id, question)
            elif isinstance(question, Choice):
                answers[question_id] = self._choice(state_text, question_id, question)
            else:
                answers[question_id] = self._score(state_text, question_id, question)
        return answers

    def close(self) -> None:
        return None

    def _units(
        self,
        state_text: str,
        question_id: str,
        instructions: str,
        labels: Sequence[str],
    ) -> list[float]:
        return [
            _unit(
                _hash_material(
                    state_text,
                    question_id,
                    instructions,
                    extra=label,
                    seed=self.seed,
                )
            )
            for label in labels
        ]

    def _noul(self, state_text: str, question_id: str, question: Noul) -> NoulAnswer:
        return NoulAnswer(
            noul=mock_noul_probability(
                state_text,
                question_id,
                question.instructions,
                seed=self.seed,
            )
        )

    def _choice(self, state_text: str, question_id: str, question: Choice) -> ChoiceAnswer:
        labels = question.options
        probs = peaked_distribution(
            self._units(state_text, question_id, question.instructions, labels)
        )
        probabilities = {
            label: probability for label, probability in zip(labels, probs, strict=True)
        }
        choice = max(probabilities, key=probabilities.__getitem__)
        return ChoiceAnswer(
            choice=choice,
            probabilities=probabilities,
            confidence=entropy_confidence(probs),
        )

    def _score(self, state_text: str, question_id: str, question: Score) -> ScoreAnswer:
        levels = question.levels
        probs = peaked_distribution(
            self._units(state_text, question_id, question.instructions, levels)
        )
        probabilities = {
            level: probability for level, probability in zip(levels, probs, strict=True)
        }
        expected = sum(index * probability for index, probability in enumerate(probs))
        return ScoreAnswer(
            score=expected,
            levels=levels,
            probabilities=probabilities,
            confidence=entropy_confidence(probs),
        )
