"""Kev must never call generate() to produce answers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src" / "kev", ROOT / "scripts")


def _generate_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "generate":
            hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        elif isinstance(func, ast.Name) and func.id == "generate":
            hits.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return hits


def test_no_generate_calls_in_src_or_scripts() -> None:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            hits.extend(_generate_calls(path))
    assert hits == [], "delete generate()-for-answers paths: " + ", ".join(hits)
