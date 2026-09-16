from __future__ import annotations

import math
from pathlib import Path

from kev.backends.fake import FakeLogitsEngine
from kev.calibration import apply_temperature, load_temperature, save_temperature
from kev.infer import score_questions, softmax
from kev.primitives import Noul
from kev.train.calibrate import fit_temperature


def test_apply_temperature_scales_logits() -> None:
    logits = [0.0, 2.0, 4.0]
    scaled = apply_temperature(logits, 2.0)
    assert scaled == [0.0, 1.0, 2.0]
    sharp = softmax(apply_temperature(logits, 0.5))
    mild = softmax(apply_temperature(logits, 2.0))
    assert sharp[2] > mild[2]


def test_fit_temperature_recovers_near_one() -> None:
    pairs = [([4.0, 0.0, 0.0], [0.78, 0.11, 0.11]) for _ in range(8)]
    temperature = fit_temperature(pairs, t_min=0.2, t_max=3.0, steps=80)
    assert 0.3 < temperature < 2.5


def test_temperature_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "temperature.json"
    save_temperature(path, 1.25)
    assert math.isclose(load_temperature(path), 1.25)


def test_score_questions_applies_engine_temperature() -> None:
    engine = FakeLogitsEngine()
    questions = {"billing": Noul(instructions="Is this about billing?")}
    engine.temperature = 0.5
    sharp, _, _ = score_questions(engine, "charged twice ASAP", questions, batch=False)
    engine.temperature = 3.0
    mild, _, _ = score_questions(engine, "charged twice ASAP", questions, batch=False)
    assert sharp["billing"].noul != mild["billing"].noul
