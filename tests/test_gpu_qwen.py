from __future__ import annotations

import pytest

from kev.backends.hf_qwen import LOAD_ERROR, HFQwenBackend


def _weights_cached(repo: str) -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.file_download import try_to_load_from_cache
    except Exception:
        try:
            from huggingface_hub import try_to_load_from_cache
        except Exception:
            return False
        HF_HUB_CACHE = None  # noqa: F841
    try:
        path = try_to_load_from_cache(repo, "config.json")
    except Exception:
        return False
    return bool(path) and str(path) != ".no_exist"


@pytest.mark.gpu
def test_real_qwen_logits_not_generate() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    gpu = torch.cuda.is_available() or (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    )
    if not gpu:
        pytest.skip("no CUDA/MPS GPU")
    repo = "Qwen/Qwen2.5-1.5B-Instruct"
    if not _weights_cached(repo):
        pytest.skip("Qwen weights are not cached")
    from kev import Choice, Noul
    from kev.backends.hf_qwen import HFQwenBackend as Backend

    backend = Backend(repo=repo, device="auto")
    try:
        generate = backend.engine.model.generate
        with pytest.raises(RuntimeError, match="must not call model.generate"):
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


def test_hf_backend_clear_error_without_stack() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="mock"):
            HFQwenBackend(repo="Qwen/Qwen2.5-1.5B-Instruct")
        assert "mock" in LOAD_ERROR
        return
    pytest.skip("torch is installed; GPU coverage lives in test_real_qwen_logits_not_generate")
