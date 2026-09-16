"""Apple MLX Qwen backend. Scores next-token logits. Never generate()."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kev.infer import PrefillCache, score_questions
from kev.primitives import Choice, Noul, Score
from kev.prompt import ChatFormat, get_chat_format
from kev.types import Answer

logger = logging.getLogger("kev.backends.mlx_qwen")

MLX_INSTALL = "pip install mlx mlx-lm"
DEFAULT_MLX_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MLX_REPO_3B = "mlx-community/Qwen2.5-3B-Instruct-4bit"


def download_command(repo: str = DEFAULT_MLX_REPO) -> str:
    return f"python scripts/download_model.py --repo {repo}"


def weights_missing_message(repo: str) -> str:
    return f"MLX weights not found for {repo}.\nRun: {download_command(repo)}"


def _import_mlx() -> tuple[Any, Any, Any]:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as exc:
        raise RuntimeError(MLX_INSTALL) from exc
    return mx, load, make_prompt_cache


def find_local_snapshot(repo: str) -> str | None:
    path = Path(repo).expanduser()
    if path.exists() and (path / "config.json").is_file():
        return str(path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None
        cached = try_to_load_from_cache(repo_id=repo, filename="config.json")
        if cached and str(cached) != ".no_exist":
            return str(Path(str(cached)).parent)
        return None
    try:
        found = snapshot_download(repo_id=repo, local_files_only=True)
        return str(found)
    except Exception:
        return None


def require_local_snapshot(repo: str) -> str:
    local = find_local_snapshot(repo)
    if local is None:
        raise RuntimeError(weights_missing_message(repo))
    return local


def _copy_state(value: Any, mx: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_copy_state(item, mx) for item in value)
    if isinstance(value, list):
        return [_copy_state(item, mx) for item in value]
    if isinstance(value, dict):
        return {key: _copy_state(item, mx) for key, item in value.items()}
    return value


def clone_prompt_cache(cache: Sequence[Any], mx: Any) -> list[Any]:
    try:
        cloned_copy = copy.deepcopy(list(cache))
        if cloned_copy:
            return cloned_copy
    except Exception:
        logger.debug("deepcopy of MLX cache failed; copying state")
    cloned: list[Any] = []
    for item in cache:
        state = _copy_state(item.state, mx)
        meta = getattr(item, "meta_state", "")
        try:
            cloned.append(type(item).from_state(state, meta))
        except TypeError:
            cloned.append(type(item).from_state(state))
    return cloned


def _logits_row(logits: Any, mx: Any) -> list[float]:
    last = logits[0, -1, :].astype(mx.float32)
    mx.eval(last)
    return [float(value) for value in last.tolist()]


def _forbid_generate(model: Any) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Kev must not call generate(); answers are assembled from logits.")

    try:
        model.generate = _blocked
    except Exception:
        logger.debug("could not patch model.generate")


class MLXLogitsEngine:
    """One prefill + suffix forwards on MLX. Teacher-forcing is forward passes."""

    def __init__(self, model: Any, tokenizer: Any, mx: Any, make_prompt_cache: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mx = mx
        self._make_prompt_cache = make_prompt_cache
        self.temperature = 1.0
        _forbid_generate(self.model)

    def encode(self, text: str) -> list[int]:
        inner = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        try:
            ids = inner.encode(text, add_special_tokens=False)
        except TypeError:
            ids = inner.encode(text)
        return [int(token) for token in ids]

    def prefill(self, prefix_text: str) -> PrefillCache:
        ids = self.encode(prefix_text)
        if not ids:
            raise ValueError("tokenizer produced no prefix ids")
        cache = self._make_prompt_cache(self.model)
        tokens = self.mx.array(ids)[None]
        logits = self.model(tokens, cache=cache)
        last = _logits_row(logits, self.mx)
        self.mx.eval(*[layer.state for layer in cache])
        logger.info("mlx prefill n_tokens=%s", len(ids))
        return PrefillCache(
            prefix_text=prefix_text,
            n_tokens=len(ids),
            payload={"cache": cache, "last_logits": last},
        )

    def next_logits(self, cache: PrefillCache, suffix_text: str) -> list[float]:
        suffix_ids = self.encode(suffix_text)
        if not suffix_ids:
            stored = cache.payload.get("last_logits") if cache.payload else None
            if isinstance(stored, list):
                return stored
            raise ValueError("empty suffix and no cached prefix logits")
        work = clone_prompt_cache(cache.payload["cache"], self.mx)
        tokens = self.mx.array(suffix_ids)[None]
        logits = self.model(tokens, cache=work)
        logger.info("mlx forward suffix_token_ids=%s", suffix_ids[:32])
        return _logits_row(logits, self.mx)

    def next_logits_batch(self, cache: PrefillCache, suffixes: Sequence[str]) -> list[list[float]]:
        return [self.next_logits(cache, suffix) for suffix in suffixes]

    def continuation_logprob(
        self, cache: PrefillCache, suffix_text: str, continuation: str
    ) -> float:
        import mlx.nn as mlx_nn

        token_ids = self.encode(continuation)
        if not token_ids:
            return float("-inf")
        work = clone_prompt_cache(cache.payload["cache"], self.mx)
        suffix_ids = self.encode(suffix_text)
        mx = self.mx
        scale = max(float(self.temperature), 1e-6)
        if suffix_ids:
            logits = self.model(mx.array(suffix_ids)[None], cache=work)
            last = logits[0, -1, :]
        else:
            last = mx.array(cache.payload["last_logits"])
        logprobs: list[float] = []
        for index, token_id in enumerate(token_ids):
            logp = mlx_nn.log_softmax(last.astype(mx.float32) / scale, axis=-1)
            picked = logp[token_id]
            mx.eval(picked)
            logprobs.append(float(picked.item()))
            if index == len(token_ids) - 1:
                break
            logits = self.model(mx.array([[token_id]]), cache=work)
            last = logits[0, -1, :]
        logger.info("mlx teacher_force token_ids=%s n=%s", token_ids, len(token_ids))
        return sum(logprobs) / len(logprobs)


def _adapter_dir(adapter: str | None) -> Path | None:
    if not adapter:
        return None
    path = Path(adapter).expanduser()
    if path.is_dir():
        return path
    logger.warning("adapter %s missing; continuing without LoRA", adapter)
    return None


class MLXQwenBackend:
    """Qwen2.5 Instruct via mlx-lm. Logits only. No generate()."""

    name = "mlx_qwen"

    def __init__(
        self,
        repo: str = DEFAULT_MLX_REPO,
        device: str = "auto",
        dtype: str = "auto",
        chat_format: str | ChatFormat | None = None,
        alias: str | None = None,
        adapter: str | None = None,
    ) -> None:
        del device, dtype  # MLX places tensors; Apple GPU is implicit.
        mx, load, make_prompt_cache = _import_mlx()
        self.repo = repo
        self.alias = alias or repo
        local = require_local_snapshot(repo)
        adapter_path = _adapter_dir(adapter)
        load_kwargs: dict[str, Any] = {}
        if adapter_path is not None and (adapter_path / "adapter_config.json").is_file():
            load_kwargs["adapter_path"] = str(adapter_path)
        elif adapter_path is not None:
            logger.warning("adapter %s has no adapter_config.json; skipping LoRA", adapter_path)
        try:
            model, tokenizer = load(local, **load_kwargs)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc) in {MLX_INSTALL}:
                raise
            raise RuntimeError(weights_missing_message(repo)) from exc
        fmt_name = chat_format.name if isinstance(chat_format, ChatFormat) else chat_format
        fmt = chat_format if isinstance(chat_format, ChatFormat) else None
        self.chat_format = fmt or get_chat_format(fmt_name or repo)
        self.engine = MLXLogitsEngine(model, tokenizer, mx, make_prompt_cache)
        self.last_input_tokens = 0
        self.adapter = adapter
        if adapter_path is not None:
            from kev.calibration import load_temperature, temperature_path

            temp_file = temperature_path(adapter_path)
            if temp_file.is_file():
                self.engine.temperature = load_temperature(temp_file)

    def infer(
        self,
        state_text: str,
        questions: dict[str, Noul | Choice | Score],
    ) -> dict[str, Answer]:
        answers, input_tokens, scored_ids = score_questions(
            self.engine,
            state_text,
            questions,
            self.chat_format,
            batch=True,
        )
        self.last_input_tokens = input_tokens
        logger.info("scored_token_ids=%s input_tokens=%s output_tokens=0", scored_ids, input_tokens)
        return answers

    def close(self) -> None:
        self.engine = None  # type: ignore[assignment]
