from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from kev import KevClient
from kev.cli import app
from kev.eval.harness import load_cases, run_harness, save_report, summarize_run

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "eval" / "smoke.jsonl"
TOY = ROOT / "data" / "eval" / "toy.jsonl"
runner = CliRunner()


def test_smoke_has_twelve_hand_written_cases() -> None:
    cases = load_cases(SMOKE)
    assert len(cases) == 12
    ids = [case.id for case in cases]
    assert any(item.startswith("easy-") for item in ids)
    assert any(item.startswith("amb-") for item in ids)
    first = cases[0]
    assert "billing" in first.questions
    assert first.gold["billing"]["type"] == "noul"


def test_toy_sft_jsonl_groups_into_cases() -> None:
    cases = load_cases(TOY)
    assert len(cases) >= 1
    assert all(case.gold for case in cases)
    assert any(len(case.questions) > 1 for case in cases)


def test_harness_mock_writes_timestamped_report(tmp_path: Path) -> None:
    cases = load_cases(SMOKE)
    with KevClient(model="mock") as client:
        rows = run_harness(cases, client)
        report = summarize_run(
            rows,
            model="mock",
            cases_path=str(SMOKE),
            base_url=None,
        )
    path = save_report(report, tmp_path)
    assert path.name.endswith(".json")
    assert report["n_cases"] == 12
    assert report["n_answers"] == 36
    assert 0.0 <= report["agreement"]["overall"] <= 1.0
    assert report["accuracy"] == report["agreement"]["overall"]
    assert report["brier"] is None or 0.0 <= report["brier"] <= 1.0
    assert report["latency_ms"]["histogram"]
    assert report["calibration"]["buckets"]
    assert path.is_file()


def test_cli_eval_mock(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "eval",
            "--cases",
            str(SMOKE),
            "--model",
            "mock",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.stdout
    assert list(tmp_path.glob("*.json"))


def test_cli_eval_teacher_openai_is_stub() -> None:
    result = runner.invoke(app, ["eval", "--teacher-openai", "--model", "gpt-4o-mini"])
    assert result.exit_code == 1
    assert "stub" in result.stdout.lower()
    assert "openai" in result.stdout.lower()
