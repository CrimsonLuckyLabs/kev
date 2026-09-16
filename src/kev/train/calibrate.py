"""Fit a scalar temperature on frozen option logits. Minimize NLL."""

from __future__ import annotations

import math
from collections.abc import Sequence

from kev.calibration import apply_temperature
from kev.infer import softmax
from kev.train.dataset import DecisionRow

LogitTarget = tuple[list[float], list[float]]


def distribution_nll(logits: Sequence[float], target: Sequence[float], temperature: float) -> float:
    probs = softmax(apply_temperature(logits, temperature))
    total = 0.0
    for mass, probability in zip(target, probs, strict=True):
        total -= float(mass) * math.log(max(float(probability), 1e-12))
    return total


def mean_nll(pairs: Sequence[LogitTarget], temperature: float) -> float:
    if not pairs:
        raise ValueError("need at least one (logits, target) pair")
    return sum(distribution_nll(logits, target, temperature) for logits, target in pairs) / len(
        pairs
    )


def fit_temperature(
    pairs: Sequence[LogitTarget],
    t_min: float = 0.05,
    t_max: float = 5.0,
    steps: int = 200,
) -> float:
    """Grid-search T to minimize mean NLL. No GPU required."""
    if t_min <= 0.0 or t_max <= t_min:
        raise ValueError("invalid temperature grid")
    best_t = 1.0
    best = mean_nll(pairs, best_t)
    delta = (t_max - t_min) / float(steps)
    temperature = t_min
    for _ in range(steps + 1):
        value = mean_nll(pairs, temperature)
        if value < best:
            best = value
            best_t = temperature
        temperature += delta
    # local refine
    half = max(delta, 0.01)
    lo = max(t_min, best_t - half)
    hi = min(t_max, best_t + half)
    refine_steps = 40
    refine_delta = (hi - lo) / float(refine_steps)
    temperature = lo
    for _ in range(refine_steps + 1):
        value = mean_nll(pairs, temperature)
        if value < best:
            best = value
            best_t = temperature
        temperature += refine_delta
    return float(best_t)


def teacher_logit_pairs(rows: Sequence[DecisionRow], peak: float = 4.0) -> list[LogitTarget]:
    """Smoke logits from teacher_probs so T~1 without a frozen backbone."""
    pairs: list[LogitTarget] = []
    for row in rows:
        target = row.target_probs()
        logits = [peak if mass == max(target) else 0.0 for mass in target]
        pairs.append((logits, target))
    return pairs
