from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

from kev.backends import (
    MLX_DEFAULT_REPO,
    MLX_REPO_3B,
    display_model_id,
    on_darwin,
    resolve_model,
)
from kev.backends.mlx_qwen import (
    MLX_INSTALL,
    MLXQwenBackend,
    download_command,
    find_local_snapshot,
    require_local_snapshot,
    weights_missing_message,
)
from kev.cli import app
from kev.client import KevClient

runner = CliRunner()


def _mlx_specs_present() -> bool:
    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def test_darwin_kev_latest_uses_mlx() -> None:
    spec = resolve_model("kev-latest")
    spec15 = resolve_model("qwen2.5-1.5b")
    spec3 = resolve_model("qwen2.5-3b")
    if on_darwin():
        assert spec.backend == "mlx_qwen"
        assert spec.repo == MLX_DEFAULT_REPO
        assert spec15.backend == "mlx_qwen"
        assert spec15.repo == MLX_DEFAULT_REPO
        assert spec3.backend == "mlx_qwen"
        assert spec3.repo == MLX_REPO_3B
    assert display_model_id("kev-latest") == "qwen2.5-1.5b"
    assert display_model_id("qwen2.5-1.5b") == "qwen2.5-1.5b"
    assert display_model_id("qwen2.5-3b") == "qwen2.5-3b"
    assert display_model_id("mock") == "mock"


def test_slash_mlx_repo_selects_mlx_backend() -> None:
    spec = resolve_model(MLX_DEFAULT_REPO)
    assert spec.backend == "mlx_qwen"
    assert spec.repo == MLX_DEFAULT_REPO


def test_mock_backend_still_offline() -> None:
    client = KevClient(model="mock")
    assert client.backend is not None
    assert client.backend.name == "mock"
    res = client.system_one(
        "I was charged twice ASAP",
        {"billing": {"type": "noul", "instructions": "Is this about billing?"}},
    )
    assert res.model == "mock"
    assert res.usage.output_tokens == 0


def test_mlx_missing_fails_loud_without_importing_metal() -> None:
    if _mlx_specs_present():
        pytest.skip("mlx is installed; load coverage is @pytest.mark.gpu")
    with pytest.raises(RuntimeError, match="pip install mlx mlx-lm"):
        MLXQwenBackend(repo=MLX_DEFAULT_REPO)


def test_missing_snapshot_prints_download_command() -> None:
    with pytest.raises(RuntimeError, match="download_model"):
        require_local_snapshot("mlx-community/kev-not-a-real-repo")
    text = weights_missing_message(MLX_DEFAULT_REPO)
    assert "download_model.py" in text
    assert MLX_DEFAULT_REPO in text
    assert download_command(MLX_DEFAULT_REPO) in text


def test_cli_ask_real_model_without_mlx_exits() -> None:
    if _mlx_specs_present():
        pytest.skip("mlx is installed; load coverage is @pytest.mark.gpu")
    result = runner.invoke(
        app,
        [
            "ask",
            "--model",
            "qwen2.5-1.5b",
            "--state",
            "I was charged twice. Help ASAP.",
            "--noul",
            "billing=Is this about billing?",
        ],
    )
    assert result.exit_code == 1
    assert MLX_INSTALL in result.output


def test_cli_ask_adapter_flag_is_accepted_for_mock() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "--model",
            "mock",
            "--adapter",
            "artifacts/missing-lora",
            "--state",
            "hello",
            "--noul",
            "billing=Is this about billing?",
        ],
    )
    assert result.exit_code == 0, result.output


@pytest.mark.gpu
def test_mlx_qwen_logits_not_generate() -> None:
    pytest.importorskip("mlx")
    pytest.importorskip("mlx_lm")
    if find_local_snapshot(MLX_DEFAULT_REPO) is None:
        pytest.skip("MLX 1.5B 4-bit weights are not cached")
    from kev.primitives import Choice, Noul

    backend = MLXQwenBackend(repo=MLX_DEFAULT_REPO)
    try:
        generate = backend.engine.model.generate
        with pytest.raises(RuntimeError, match="must not call generate"):
            generate()
        answers = backend.infer(
            "I was charged twice ASAP",
            {
                "billing": Noul(instructions="Is this about billing?"),
                "tone": Choice(
                    instructions="Tone?",
                    criteria={"calm": None, "angry": None},
                ),
            },
        )
        assert 0.0 <= answers["billing"].noul <= 1.0
        assert answers["tone"].choice in {"calm", "angry"}
        assert backend.last_input_tokens > 0
    finally:
        backend.close()


def test_download_default_is_mlx_1_5b() -> None:
    from kev.backends import MLX_DEFAULT_REPO as repo

    assert "Qwen2.5-1.5B-Instruct-4bit" in repo
