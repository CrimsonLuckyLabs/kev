"""Evaluation package. Numeric metrics and a gold-case harness."""

from kev.eval.harness import EvalCase, evaluate_path, load_cases, run_harness
from kev.eval.metrics import (
    auroc,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    latency_histogram,
)

__all__ = [
    "EvalCase",
    "auroc",
    "brier_score",
    "calibration_curve",
    "evaluate_path",
    "expected_calibration_error",
    "latency_histogram",
    "load_cases",
    "run_harness",
]
