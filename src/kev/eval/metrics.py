"""Numeric eval for Noul / Choice / Score. Stdlib only."""

from __future__ import annotations

import math
from collections.abc import Sequence


def brier_score(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("brier_score requires aligned non-empty sequences")
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        delta = float(probability) - float(label)
        total += delta * delta
    return total / len(probabilities)


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC. None if a class is missing."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("auroc requires aligned non-empty sequences")
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        return None
    greater = 0.0
    ties = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                greater += 1.0
            elif pos == neg:
                ties += 1.0
    return (greater + 0.5 * ties) / (len(positives) * len(negatives))


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("ECE requires aligned non-empty sequences")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for index, confidence in enumerate(confidences):
        clipped = min(1.0, max(0.0, float(confidence)))
        bin_index = min(n_bins - 1, int(clipped * n_bins))
        if clipped == 1.0:
            bin_index = n_bins - 1
        bins[bin_index].append(index)
    ece = 0.0
    n_total = len(confidences)
    for bucket in bins:
        if not bucket:
            continue
        acc = sum(1.0 for index in bucket if correct[index]) / len(bucket)
        conf = sum(float(confidences[index]) for index in bucket) / len(bucket)
        ece += (len(bucket) / n_total) * abs(acc - conf)
    return ece


def accuracy(predicted: Sequence[str], labels: Sequence[str]) -> float:
    if len(predicted) != len(labels) or not predicted:
        raise ValueError("accuracy requires aligned non-empty sequences")
    hits = sum(1 for pred, label in zip(predicted, labels, strict=True) if pred == label)
    return hits / len(predicted)


def macro_f1(predicted: Sequence[str], labels: Sequence[str]) -> float:
    if len(predicted) != len(labels) or not predicted:
        raise ValueError("macro_f1 requires aligned non-empty sequences")
    classes = sorted(set(predicted) | set(labels))
    scores: list[float] = []
    for name in classes:
        tp = sum(
            1
            for pred, label in zip(predicted, labels, strict=True)
            if pred == name and label == name
        )
        fp = sum(
            1
            for pred, label in zip(predicted, labels, strict=True)
            if pred == name and label != name
        )
        fn = sum(
            1
            for pred, label in zip(predicted, labels, strict=True)
            if pred != name and label == name
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def mean_absolute_error(predicted: Sequence[float], labels: Sequence[float]) -> float:
    if len(predicted) != len(labels) or not predicted:
        raise ValueError("MAE requires aligned non-empty sequences")
    total = 0.0
    for pred, label in zip(predicted, labels, strict=True):
        total += abs(float(pred) - float(label))
    return total / len(predicted)


def noul_agrees(predicted: float, gold: float, threshold: float = 0.5) -> bool:
    """Hard agreement: both on the same side of the threshold."""
    return (float(predicted) >= threshold) == (float(gold) >= threshold)


def agreement_rate(correct: Sequence[bool]) -> float:
    if not correct:
        raise ValueError("agreement_rate requires at least one judgment")
    return sum(1.0 for item in correct if item) / len(correct)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation percentile. q is in [0, 100]."""
    if not values:
        raise ValueError("percentile requires values")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = min(len(ordered) - 1, low + 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


DEFAULT_LATENCY_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


def latency_histogram(
    values: Sequence[float],
    edges: Sequence[float] = DEFAULT_LATENCY_EDGES,
) -> list[dict[str, float | int | None]]:
    """Right-open bins [lo, hi). The last bin is [last_edge, +inf)."""
    if len(edges) < 1:
        raise ValueError("histogram needs at least one edge")
    bounds = [float(edge) for edge in edges]
    counts = [0] * (len(bounds))
    for raw in values:
        value = float(raw)
        placed = False
        for index in range(len(bounds) - 1):
            if bounds[index] <= value < bounds[index + 1]:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    buckets: list[dict[str, float | int | None]] = []
    for index in range(len(bounds) - 1):
        buckets.append({"lo": bounds[index], "hi": bounds[index + 1], "count": counts[index]})
    buckets.append({"lo": bounds[-1], "hi": None, "count": counts[-1]})
    return buckets


def calibration_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Confidence buckets vs accuracy. Empty bins keep n=0."""
    if len(confidences) != len(correct):
        raise ValueError("calibration_curve requires aligned sequences")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[int]] = [[] for _ in range(n_bins)]
    for index, confidence in enumerate(confidences):
        clipped = min(1.0, max(0.0, float(confidence)))
        bin_index = min(n_bins - 1, int(clipped * n_bins))
        if clipped == 1.0:
            bin_index = n_bins - 1
        buckets[bin_index].append(index)
    curve: list[dict[str, float | int | None]] = []
    width = 1.0 / n_bins
    for bin_index, members in enumerate(buckets):
        lo = bin_index * width
        hi = 1.0 if bin_index == n_bins - 1 else (bin_index + 1) * width
        if not members:
            curve.append({"lo": lo, "hi": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        acc = sum(1.0 for index in members if correct[index]) / len(members)
        conf = sum(float(confidences[index]) for index in members) / len(members)
        curve.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(members),
                "confidence": conf,
                "accuracy": acc,
            }
        )
    return curve

