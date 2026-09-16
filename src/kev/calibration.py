"""Temperature scaling. Confidence stays a number software can threshold."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

TEMPERATURE_NAME = "temperature.json"


def apply_temperature(logits: Sequence[float], temperature: float) -> list[float]:
    """Return logits / T. Caller softmaxes. T must be > 0."""
    scale = float(temperature)
    if scale <= 0.0:
        raise ValueError("temperature must be > 0")
    return [float(logit) / scale for logit in logits]


def load_temperature(path: str | Path) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("temperature file must be a JSON object")
    raw = payload.get("temperature", payload.get("T"))
    if raw is None:
        raise ValueError("temperature file missing 'temperature'")
    value = float(raw)
    if value <= 0.0:
        raise ValueError("temperature must be > 0")
    return value


def save_temperature(path: str | Path, temperature: float) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"temperature": float(temperature)}, indent=2) + "\n",
        encoding="utf-8",
    )


def temperature_path(adapter_dir: str | Path) -> Path:
    return Path(adapter_dir) / TEMPERATURE_NAME
