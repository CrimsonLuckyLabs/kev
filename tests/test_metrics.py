from __future__ import annotations

from kev.eval.metrics import (
    accuracy,
    agreement_rate,
    auroc,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    latency_histogram,
    macro_f1,
    mean_absolute_error,
    noul_agrees,
    percentile,
)


def test_brier_known_vector() -> None:
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
    # (0.25^2 + 0.25^2) / 2 = 0.0625
    assert brier_score([0.75, 0.25], [1.0, 0.0]) == 0.0625
    assert brier_score([0.5, 0.5], [1.0, 0.0]) == 0.25


def test_auroc_known_vector() -> None:
    score = auroc([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert score == 1.0
    assert auroc([1, 1], [0.9, 0.2]) is None
    mixed = auroc([0, 1, 0, 1], [0.2, 0.2, 0.8, 0.9])
    assert mixed is not None
    # ties on 0.2: one pos vs two neg → 0.5; 0.9 beats both neg
    # greater=2 (0.9>0.2 twice) + 0.5*1 (0.2 tie with one neg? pos 0.2 vs neg 0.2 and 0.8)
    # pos=[0.2, 0.9] neg=[0.2, 0.8]
    # 0.2 vs 0.2 tie, 0.2 vs 0.8 lose → 0.5
    # 0.9 vs 0.2 win, 0.9 vs 0.8 win → 2
    # (2.5) / 4 = 0.625
    assert mixed == 0.625


def test_ece_known_vector() -> None:
    # two sure-correct at 1.0, two sure-correct at 0.0 → ECE = 0.5
    ece = expected_calibration_error(
        [1.0, 1.0, 0.0, 0.0],
        [True, True, True, True],
        n_bins=10,
    )
    assert ece == 0.5
    ece_ok = expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, True, False, False])
    assert 0.0 <= ece_ok < 0.2


def test_percentile_known_vector() -> None:
    assert percentile([0.0, 10.0], 0) == 0.0
    assert percentile([0.0, 10.0], 100) == 10.0
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([7.0], 95) == 7.0


def test_latency_histogram_known_vector() -> None:
    buckets = latency_histogram([0.5, 0.5, 11.0, 250.0], edges=(0.0, 10.0, 50.0, 200.0))
    assert buckets[0] == {"lo": 0.0, "hi": 10.0, "count": 2}
    assert buckets[1] == {"lo": 10.0, "hi": 50.0, "count": 1}
    assert buckets[2] == {"lo": 50.0, "hi": 200.0, "count": 0}
    assert buckets[3] == {"lo": 200.0, "hi": None, "count": 1}


def test_calibration_curve_known_vector() -> None:
    curve = calibration_curve(
        [0.05, 0.15, 0.95, 0.95],
        [False, False, True, True],
        n_bins=10,
    )
    assert len(curve) == 10
    assert curve[0]["n"] == 1
    assert curve[0]["confidence"] == 0.05
    assert curve[0]["accuracy"] == 0.0
    assert curve[1]["n"] == 1
    assert curve[1]["confidence"] == 0.15
    assert curve[1]["accuracy"] == 0.0
    assert curve[9]["n"] == 2
    assert curve[9]["confidence"] == 0.95
    assert curve[9]["accuracy"] == 1.0
    empty = curve[5]
    assert empty["n"] == 0
    assert empty["accuracy"] is None


def test_noul_agreement_and_rate() -> None:
    assert noul_agrees(0.9, 1.0) is True
    assert noul_agrees(0.1, 1.0) is False
    assert noul_agrees(0.4, 0.0) is True
    assert agreement_rate([True, True, False]) == 2 / 3
    assert accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "b", "a"], ["a", "a", "a"]) > 0.0
    assert mean_absolute_error([2.0, 0.0], [2.0, 1.0]) == 0.5
