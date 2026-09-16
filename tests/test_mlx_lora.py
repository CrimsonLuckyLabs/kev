from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kev.train.dataset import DecisionDataset, HashTokenizer
from kev.train.mlx_lora import (
    DEFAULT_GRAD_ACCUM,
    DEFAULT_ITERS,
    DEFAULT_LR,
    DEFAULT_MAX_SEQ,
    MLX_LORA_ALPHA,
    MLX_LORA_RANK,
    adapter_config,
    assert_qwen_1_5b,
    clip_prompt_ids,
    python_option_ce,
    row_training_example,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "sft" / "example.jsonl"


def _load_script(name: str) -> object:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lora_defaults_are_1_5b_smoke() -> None:
    config = adapter_config()
    assert config["fine_tune_type"] == "lora"
    assert config["num_layers"] == -1
    params = config["lora_parameters"]
    assert params["rank"] == MLX_LORA_RANK == 8
    assert params["scale"] == float(MLX_LORA_ALPHA) == 16.0
    assert "self_attn.q_proj" in params["keys"]
    assert DEFAULT_ITERS == 200
    assert DEFAULT_GRAD_ACCUM == 4
    assert DEFAULT_MAX_SEQ == 1024
    assert DEFAULT_LR == 1e-5


def test_assert_qwen_1_5b_rejects_3b() -> None:
    assert_qwen_1_5b("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    with pytest.raises(ValueError, match="1.5B"):
        assert_qwen_1_5b("mlx-community/Qwen2.5-3B-Instruct-4bit")


def test_clip_keeps_decision_suffix() -> None:
    ids = list(range(20))
    assert clip_prompt_ids(ids, 8) == list(range(12, 20))
    assert clip_prompt_ids(ids, 20) == ids


def test_row_training_example_option_tokens_not_json() -> None:
    dataset = DecisionDataset.from_jsonl(EXAMPLE)
    tokenizer = HashTokenizer.from_rows(dataset.rows)
    for row in dataset.rows:
        ids, option_ids, target = row_training_example(row, tokenizer, max_seq=64)
        assert ids
        assert len(option_ids) == len(target) == len(row.option_labels())
        assert abs(sum(target) - 1.0) < 1e-9
        prompt = row.prompt_text()
        assert "{" not in prompt.split("assistant")[-1]
        ce = python_option_ce([0.0] * (max(option_ids) + 2), option_ids, target)
        assert ce > 0.0


def test_train_sft_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script("train_sft.py")
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--dry-run", "--data", str(EXAMPLE), "--out", "artifacts/kev-1p5-lora"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Skipping LoRA training" in captured.out


def test_train_calibrate_smoke_without_mlx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _load_script("train_calibrate.py")
    monkeypatch.setattr(script, "has_mlx", lambda: False)
    script.main(["--data", str(EXAMPLE), "--out", str(tmp_path)])
    written = tmp_path / "temperature.json"
    assert written.is_file()
    payload = written.read_text(encoding="utf-8")
    assert "temperature" in payload


def test_eval_kev_smoke_mock(tmp_path: Path) -> None:
    script = _load_script("eval_kev.py")
    out = tmp_path / "report.json"
    script.main(
        [
            "--data",
            str(ROOT / "data" / "eval" / "smoke.jsonl"),
            "--model",
            "mock",
            "--out",
            str(out),
            "--out-dir",
            str(tmp_path / "eval"),
        ]
    )
    report = out.read_text(encoding="utf-8")
    assert "accuracy" in report
    assert "brier" in report
    assert "latency_ms" in report
