"""Eval harness. Cases in, agreement / latency / calibration out. Kev does not chat."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kev.client import KevClient
from kev.eval.metrics import (
    agreement_rate,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    latency_histogram,
    noul_agrees,
    percentile,
)
from kev.primitives import Choice, Noul, Score, parse_questions
from kev.types import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    State,
    SystemOneResponse,
)

GoldKind = Literal["noul", "choice", "score"]


class GoldNoul(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class GoldChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    choice: str


class GoldScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    level: str | None = None
    score: float | None = None
    index: int | None = None

    @model_validator(mode="after")
    def _has_target(self) -> GoldScore:
        if self.level is None and self.score is None and self.index is None:
            raise ValueError("score gold needs level, score, or index")
        return self


GoldLabel = GoldNoul | GoldChoice | GoldScore


class EvalCase(BaseModel):
    """One unlabeled state plus gold labels for a subset of questions."""

    model_config = ConfigDict(extra="forbid")

    id: str
    state: State
    questions: dict[str, dict[str, Any]]
    gold: dict[str, dict[str, Any]]

    def parsed_questions(self) -> dict[str, Noul | Choice | Score]:
        return parse_questions(self.questions)

    def parsed_gold(self) -> dict[str, GoldLabel]:
        parsed: dict[str, GoldLabel] = {}
        for name, payload in self.gold.items():
            kind = payload.get("type")
            if kind == "noul":
                parsed[name] = GoldNoul.model_validate(payload)
            elif kind == "choice":
                parsed[name] = GoldChoice.model_validate(payload)
            elif kind == "score":
                parsed[name] = GoldScore.model_validate(payload)
            else:
                raise ValueError(f"gold {name!r} missing a valid type")
        return parsed

    @model_validator(mode="after")
    def _gold_keys(self) -> EvalCase:
        if not self.questions:
            raise ValueError("case needs questions")
        if not self.gold:
            raise ValueError("case needs gold")
        extra = set(self.gold) - set(self.questions)
        if extra:
            raise ValueError(f"gold keys not in questions: {sorted(extra)}")
        return self


def _sft_gold(row: Any) -> dict[str, Any]:
    question = row.question
    if question.type == "noul":
        return {"type": "noul", "noul": float(row.label.noul or 0.0)}
    if question.type == "choice":
        return {"type": "choice", "choice": str(row.label.choice)}
    return {"type": "score", "level": str(row.label.level), "index": row.label.index}


def _cases_from_sft(path: Path) -> list[EvalCase]:
    from kev.primitives import dump_questions
    from kev.train.dataset import DecisionDataset

    dataset = DecisionDataset.from_jsonl(path)
    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    for row in dataset.rows:
        key = json.dumps(row.state, sort_keys=True, default=str)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(row)
    cases: list[EvalCase] = []
    for key in order:
        rows = groups[key]
        questions = {row.question.id: row.question.as_primitive() for row in rows}
        gold = {row.question.id: _sft_gold(row) for row in rows}
        prefix = rows[0].id.rsplit("-", 1)[0] if "-" in rows[0].id else rows[0].id
        cases.append(
            EvalCase(
                id=prefix,
                state=rows[0].state,
                questions=dump_questions(questions),
                gold=gold,
            )
        )
    return cases


def load_cases(path: str | Path) -> list[EvalCase]:
    target = Path(path)
    lines = [
        json.loads(raw)
        for raw in target.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if not lines:
        raise ValueError(f"no cases in {target}")
    sample = lines[0]
    if "questions" in sample and "gold" in sample:
        return [EvalCase.model_validate(item) for item in lines]
    if "question" in sample and "label" in sample:
        return _cases_from_sft(target)
    raise ValueError(
        f"{target} is not an eval case file or a labeled Decision JSONL"
    )


def gold_agrees(answer: Answer, gold: GoldLabel) -> bool:
    if isinstance(gold, GoldNoul) and isinstance(answer, NoulAnswer):
        return noul_agrees(answer.noul, gold.noul)
    if isinstance(gold, GoldChoice) and isinstance(answer, ChoiceAnswer):
        return answer.choice == gold.choice
    if isinstance(gold, GoldScore) and isinstance(answer, ScoreAnswer):
        if gold.level is not None:
            return answer.level == gold.level
        if gold.index is not None:
            predicted = int(round(answer.score))
            return predicted == int(gold.index)
        assert gold.score is not None
        return abs(answer.score - gold.score) < 0.5
    return False


def answer_confidence(answer: Answer) -> float:
    if isinstance(answer, NoulAnswer):
        return max(answer.noul, 1.0 - answer.noul)
    return float(answer.confidence)


def score_case(case: EvalCase, response: SystemOneResponse) -> dict[str, Any]:
    gold = case.parsed_gold()
    items: list[dict[str, Any]] = []
    for name, label in gold.items():
        if name not in response.answers:
            raise ValueError(f"response missing answer {name!r} for case {case.id}")
        answer = response.answers[name]
        correct = gold_agrees(answer, label)
        payload: dict[str, Any] = {
            "id": name,
            "type": label.type,
            "correct": correct,
            "confidence": answer_confidence(answer),
        }
        if isinstance(answer, NoulAnswer) and isinstance(label, GoldNoul):
            payload["p"] = float(answer.noul)
            payload["y"] = float(label.noul)
        items.append(payload)
    return {
        "id": case.id,
        "latency_ms": response.usage.latency_ms,
        "items": items,
    }


def summarize_run(
    case_rows: list[dict[str, Any]],
    *,
    model: str,
    cases_path: str,
    base_url: str | None,
) -> dict[str, Any]:
    correct: list[bool] = []
    by_type: dict[str, list[bool]] = {"noul": [], "choice": [], "score": []}
    confidences: list[float] = []
    latencies = [float(row["latency_ms"]) for row in case_rows]
    for row in case_rows:
        for item in row["items"]:
            flag = bool(item["correct"])
            correct.append(flag)
            by_type[str(item["type"])].append(flag)
            confidences.append(float(item["confidence"]))
    agreement: dict[str, float | None] = {
        "overall": agreement_rate(correct) if correct else None,
    }
    for kind, flags in by_type.items():
        agreement[kind] = agreement_rate(flags) if flags else None
    ece = (
        expected_calibration_error(confidences, correct)
        if confidences
        else None
    )
    noul_p = [
        float(item["p"])
        for row in case_rows
        for item in row["items"]
        if item.get("type") == "noul" and "p" in item and "y" in item
    ]
    noul_y = [
        float(item["y"])
        for row in case_rows
        for item in row["items"]
        if item.get("type") == "noul" and "p" in item and "y" in item
    ]
    brier = brier_score(noul_p, noul_y) if noul_p else None
    accuracy = agreement["overall"]
    return {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "base_url": base_url,
        "cases_path": cases_path,
        "n_cases": len(case_rows),
        "n_answers": len(correct),
        "accuracy": accuracy,
        "brier": brier,
        "agreement": agreement,
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": percentile(latencies, 50) if latencies else None,
            "p95": percentile(latencies, 95) if latencies else None,
            "histogram": latency_histogram(latencies),
        },
        "calibration": {
            "ece": ece,
            "buckets": calibration_curve(confidences, correct) if confidences else [],
        },
        "per_case": case_rows,
    }


def save_report(report: dict[str, Any], out_dir: str | Path) -> Path:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{stamp}.json"
    suffix = 0
    while path.exists():
        suffix += 1
        path = target_dir / f"{stamp}-{suffix}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def run_harness(
    cases: list[EvalCase],
    client: KevClient,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        response = client.system_one(case.state, case.parsed_questions())
        rows.append(score_case(case, response))
    return rows


def evaluate_path(
    cases_path: str | Path,
    *,
    model: str = "mock",
    base_url: str | None = None,
    adapter: str | None = None,
    out_dir: str | Path = "artifacts/eval",
) -> tuple[dict[str, Any], Path]:
    cases = load_cases(cases_path)
    with KevClient(model=model, base_url=base_url, adapter=adapter) as client:
        rows = run_harness(cases, client)
        report_model = client.model
    report = summarize_run(
        rows,
        model=report_model,
        cases_path=str(cases_path),
        base_url=base_url,
    )
    path = save_report(report, out_dir)
    return report, path
