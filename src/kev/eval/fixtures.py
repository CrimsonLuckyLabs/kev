"""Shared eval fixtures. GPU-free toy JSONL."""

from __future__ import annotations

from pathlib import Path

from kev.train.dataset import DecisionDataset

DEFAULT_EVAL = Path("data/eval/toy.jsonl")
DEFAULT_EXAMPLE = Path("data/sft/example.jsonl")


def load_fixtures(path: str | Path | None = None) -> DecisionDataset:
    target = Path(path) if path is not None else DEFAULT_EVAL
    if not target.is_file():
        target = DEFAULT_EXAMPLE
    return DecisionDataset.from_jsonl(target)
