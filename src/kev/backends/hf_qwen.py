"""Hugging Face Qwen backend. Scores next-token logits. Never model.generate()."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

from kev.infer import PrefillCache, score_questions
from kev.primitives import Choice, Noul, Score
from kev.prompt import ChatFormat, get_chat_format
from kev.types import Answer

logger = logging.getLogger("kev.backends.hf_qwen")

LOAD_ERROR = (
    "Could not load the Qwen backend (torch/transformers or weights). "
    "Use --model mock, or run scripts/download_model.py after installing torch and transformers."
)


def _import_torch() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(LOAD_ERROR) from exc
    return torch, transformers


def resolve_device(device: str, torch: Any) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"
    warnings.warn("CUDA and MPS missing; running Kev on CPU", RuntimeWarning, stacklevel=2)
    return "cpu"


def _clone_past(past: Any) -> Any:
    if past is None:
        return None
    copier = getattr(past, "copy", None)
    if callable(copier):
        try:
            return copier()
        except Exception:
            pass
    try:
        return tuple(tuple(tensor.clone() for tensor in layer) for layer in past)
    except Exception:
        return past


def _forbid_generate(model: Any) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Kev must not call model.generate(); answers are assembled from logits.")

    model.generate = _blocked


class HFLogitsEngine:
    """One prefill + suffix forwards. Teacher-forcing is forward passes, not decode."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = 1.0
        self.model.eval()
        _forbid_generate(self.model)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos = getattr(self.tokenizer, "eos_token", None)
            if eos is not None:
                self.tokenizer.pad_token = eos

    def _device(self) -> Any:
        return next(self.model.parameters()).device

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(token) for token in ids]

    def prefill(self, prefix_text: str) -> PrefillCache:
        torch, _transformers = _import_torch()
        encoded = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self._device()) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded, use_cache=True)
        n_tokens = int(encoded["input_ids"].shape[-1])
        logger.info(
            "prefill token_ids=%s n_tokens=%s",
            encoded["input_ids"][0].tolist()[:32],
            n_tokens,
        )
        return PrefillCache(
            prefix_text=prefix_text,
            n_tokens=n_tokens,
            payload={"input_ids": encoded["input_ids"], "past": output.past_key_values},
        )

    def next_logits(self, cache: PrefillCache, suffix_text: str) -> list[float]:
        torch, _transformers = _import_torch()
        suffix = self.tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
        suffix_ids = suffix["input_ids"].to(self._device())
        past = _clone_past(cache.payload["past"] if cache.payload else None)
        with torch.inference_mode():
            try:
                if past is None:
                    raise RuntimeError("empty prefix cache")
                output = self.model(input_ids=suffix_ids, past_key_values=past, use_cache=True)
            except Exception:
                prefix_ids = cache.payload["input_ids"]
                full = torch.cat([prefix_ids, suffix_ids], dim=1)
                output = self.model(input_ids=full, use_cache=False)
        logits = output.logits[0, -1, :].detach().float().cpu()
        logger.info("forward suffix_token_ids=%s", suffix_ids[0].tolist()[:32])
        return [float(value) for value in logits.tolist()]

    def next_logits_batch(self, cache: PrefillCache, suffixes: Sequence[str]) -> list[list[float]]:
        if not suffixes:
            return []
        torch, _transformers = _import_torch()
        prefix_ids = cache.payload["input_ids"]
        device = prefix_ids.device
        suffix_rows = [self.encode(suffix) for suffix in suffixes]
        max_suffix = max((len(row) for row in suffix_rows), default=0)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        prefix_list = prefix_ids[0].tolist()
        prefix_len = len(prefix_list)
        batch_ids: list[list[int]] = []
        for suffix in suffix_rows:
            pad = [pad_id] * (max_suffix - len(suffix))
            batch_ids.append(prefix_list + pad + suffix)
        batch = torch.tensor(batch_ids, dtype=torch.long, device=device)
        attention = torch.ones_like(batch)
        for index, suffix in enumerate(suffix_rows):
            extra = max_suffix - len(suffix)
            if extra:
                attention[index, prefix_len : prefix_len + extra] = 0
        with torch.inference_mode():
            output = self.model(input_ids=batch, attention_mask=attention, use_cache=False)
        gathered: list[list[float]] = []
        for index, suffix in enumerate(suffix_rows):
            last = prefix_len + (max_suffix - len(suffix)) + len(suffix) - 1
            gathered.append(output.logits[index, last, :].detach().float().cpu().tolist())
        logger.info("batched forward rows=%s suffix_token_ids=%s", len(suffixes), suffix_rows)
        return gathered

    def continuation_logprob(
        self, cache: PrefillCache, suffix_text: str, continuation: str
    ) -> float:
        """Length-normalized logprob of the full option string via teacher-forcing."""
        torch, _transformers = _import_torch()
        token_ids = self.encode(continuation)
        if not token_ids:
            return float("-inf")
        suffix = self.tokenizer(suffix_text, return_tensors="pt", add_special_tokens=False)
        suffix_ids = suffix["input_ids"].to(self._device())
        prefix_ids = cache.payload["input_ids"]
        context = torch.cat([prefix_ids, suffix_ids], dim=1)
        logprobs: list[float] = []
        with torch.inference_mode():
            output = self.model(input_ids=context, use_cache=True)
            logits = output.logits[0, -1, :]
            past = output.past_key_values
            for index, token_id in enumerate(token_ids):
                scaled = logits.float() / max(float(self.temperature), 1e-6)
                log_probs = torch.log_softmax(scaled, dim=-1)
                logprobs.append(float(log_probs[token_id]))
                if index == len(token_ids) - 1:
                    break
                nxt = torch.tensor([[token_id]], device=self._device())
                output = self.model(input_ids=nxt, past_key_values=past, use_cache=True)
                logits = output.logits[0, -1, :]
                past = output.past_key_values
        logger.info("teacher_force token_ids=%s n=%s", token_ids, len(token_ids))
        return sum(logprobs) / len(logprobs)


class HFQwenBackend:
    """Qwen2.5 / Qwen3 Instruct via transformers. Logits only."""

    name = "hf_qwen"

    def __init__(
        self,
        repo: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        dtype: str = "auto",
        chat_format: str | ChatFormat | None = None,
        alias: str | None = None,
        adapter: str | None = None,
    ) -> None:
        torch, transformers = _import_torch()
        self.repo = repo
        self.alias = alias or repo
        device_name = resolve_device(device, torch)
        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if dtype == "auto":
            load_kwargs["torch_dtype"] = torch.float32 if device_name == "cpu" else "auto"
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)
        load_kwargs["device_map"] = {"": "cpu"} if device_name == "cpu" else "auto"
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = transformers.AutoModelForCausalLM.from_pretrained(repo, **load_kwargs)
            if adapter:
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError("peft is required to load a LoRA adapter") from exc
                model = PeftModel.from_pretrained(model, adapter)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "peft" in str(exc):
                raise
            raise RuntimeError(LOAD_ERROR) from exc
        fmt_name = chat_format.name if isinstance(chat_format, ChatFormat) else chat_format
        self.chat_format = chat_format if isinstance(chat_format, ChatFormat) else get_chat_format(
            fmt_name or repo
        )
        self.engine = HFLogitsEngine(model, tokenizer)
        self.last_input_tokens = 0
        self.device_name = device_name
        self.adapter = adapter
        if adapter:
            from kev.calibration import load_temperature, temperature_path

            temp_file = temperature_path(adapter)
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
        model = getattr(self, "engine", None)
        if model is None:
            return
        inner = getattr(model, "model", None)
        if inner is not None:
            del inner
        self.engine = None  # type: ignore[assignment]
