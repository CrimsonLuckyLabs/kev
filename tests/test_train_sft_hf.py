from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from kev.train.dataset import DecisionDataset

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "sft" / "example.jsonl"


def _load_script() -> object:
    path = ROOT / "scripts" / "train_sft_hf.py"
    spec = importlib.util.spec_from_file_location("train_sft_hf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_does_not_import_mlx() -> None:
    text = (ROOT / "scripts" / "train_sft_hf.py").read_text(encoding="utf-8")
    assert "import mlx" not in text
    assert "mlx_lm" not in text
    assert "mlx.core" not in text
    assert "mlx_lora" not in text


def test_hf_sft_flag_defaults() -> None:
    script = _load_script()
    args = script.parse_args([])
    assert args.model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.data == "data/sft/toy.jsonl"
    assert args.out == "artifacts/kev-7b-lora"
    assert args.iters == 200
    assert args.max_seq == 1024
    assert args.lora_rank == 8
    assert args.lr == 1e-5


def test_option_token_loss_is_closed_set_not_full_sequence() -> None:
    from kev.train.sft import gather_option_logits, option_cross_entropy

    script = _load_script()
    vocab = [0.0] * 8
    vocab[2] = 4.0
    vocab[5] = 1.0
    gathered = gather_option_logits(vocab, [2, 5])
    win = option_cross_entropy(gathered, [1.0, 0.0])
    lose = option_cross_entropy(gathered, [0.0, 1.0])
    assert win < lose
    assert callable(script.option_token_loss)


def test_row_training_example_is_prompt_and_option_ids() -> None:
    script = _load_script()
    dataset = DecisionDataset.from_jsonl(EXAMPLE)

    class Tok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) % 97 + 1 for ch in text[:12]] or [1]

    tok = Tok()
    for row in dataset.rows:
        ids, option_ids, target = script.row_training_example(row, tok, max_seq=64)
        assert ids
        assert "{" not in row.prompt_text().split("assistant")[-1]
        assert len(option_ids) == len(target) == len(row.option_labels())
        assert abs(sum(target) - 1.0) < 1e-9


def test_row_text_is_prompt_plus_option_label_not_json() -> None:
    script = _load_script()
    dataset = DecisionDataset.from_jsonl(EXAMPLE)
    for row in dataset.rows:
        text = script.row_text(row)
        assert text.startswith(row.prompt_text())
        assert text.endswith(row.hard_label())
        assert row.hard_label() in row.option_labels()
        assistant = text.split("assistant")[-1]
        assert "{" not in assistant
        assert "json" not in assistant.lower()


def test_encode_ids_keeps_label_suffix() -> None:
    script = _load_script()

    class Tok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(len(text)))

    ids = script.encode_ids(Tok(), "abcdefghij", max_seq=4)
    assert ids == [6, 7, 8, 9]


def test_train_sft_hf_skips_without_cuda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    monkeypatch.setattr(script, "has_cuda", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        script.main(["--data", str(EXAMPLE), "--out", "artifacts/kev-7b-lora"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Skipping PEFT training" in captured.out
    assert "dataset rows=" in captured.out
