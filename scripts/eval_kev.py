"""Evaluate Kev on smoke/toy cases: accuracy, Brier/ECE, latency."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from kev.eval.harness import evaluate_path

logger = logging.getLogger("kev.eval")


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| split | metric | value |",
        "| --- | --- | --- |",
    ]
    latency = report.get("latency_ms") if isinstance(report.get("latency_ms"), dict) else {}
    calibration = report.get("calibration") if isinstance(report.get("calibration"), dict) else {}
    rows: list[tuple[str, str, object]] = [
        ("overall", "n_cases", report.get("n_cases")),
        ("overall", "n_answers", report.get("n_answers")),
        ("overall", "accuracy", report.get("accuracy")),
        ("overall", "brier", report.get("brier")),
        ("overall", "ece", calibration.get("ece") if isinstance(calibration, dict) else None),
        ("overall", "latency_p50_ms", latency.get("p50") if isinstance(latency, dict) else None),
        ("overall", "latency_p95_ms", latency.get("p95") if isinstance(latency, dict) else None),
    ]
    agreement = report.get("agreement") if isinstance(report.get("agreement"), dict) else {}
    if isinstance(agreement, dict):
        for kind in ("noul", "choice", "score"):
            rows.append((kind, "accuracy", agreement.get(kind)))
    for kind, name, value in rows:
        if value is None:
            rendered = "—"
        elif isinstance(value, float):
            rendered = f"{value:.4f}"
        else:
            rendered = str(value)
        lines.append(f"| {kind} | {name} | {rendered} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Kev on smoke/toy cases.")
    parser.add_argument("--data", default="data/eval/smoke.jsonl")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="artifacts/eval_report.json")
    parser.add_argument("--out-dir", default="artifacts/eval")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report, harness_path = evaluate_path(
        args.data,
        model=args.model,
        adapter=args.adapter,
        out_dir=args.out_dir,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text(markdown_table(report), encoding="utf-8")
    print(markdown_table(report), end="")
    print(f"wrote {out}, {md_path}, and {harness_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
