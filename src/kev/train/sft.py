"""Option-token SFT. The model is never trained to emit JSON or prose."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from kev.infer import softmax
from kev.train.dataset import CollatedBatch

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def option_cross_entropy(logits: Sequence[float], target: Sequence[float]) -> float:
    """Soft CE over a closed option set. Invalid labels cannot appear."""
    probs = softmax(logits)
    total = 0.0
    for mass, probability in zip(target, probs, strict=True):
        total -= float(mass) * math.log(max(float(probability), 1e-12))
    return total


def gather_option_logits(
    vocab_logits: Sequence[float], option_token_ids: Sequence[int]
) -> list[float]:
    gathered: list[float] = []
    vocab = len(vocab_logits)
    for token_id in option_token_ids:
        if 0 <= int(token_id) < vocab:
            gathered.append(float(vocab_logits[int(token_id)]))
        else:
            gathered.append(float("-inf"))
    return gathered


def batch_option_loss(vocab_logits: Sequence[Sequence[float]], batch: CollatedBatch) -> float:
    if len(vocab_logits) != len(batch):
        raise ValueError("logits rows must match the collated batch")
    total = 0.0
    for row_logits, token_ids, target in zip(
        vocab_logits, batch.option_token_ids, batch.target_probs, strict=True
    ):
        option_logits = gather_option_logits(row_logits, token_ids)
        total += option_cross_entropy(option_logits, target)
    return total / len(batch)


def lora_config() -> Any:
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise RuntimeError("peft is required for LoRA SFT") from exc
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGETS),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def has_gpu() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    checker = getattr(mps, "is_available", None)
    return bool(callable(checker) and checker())


def last_token_logits(outputs: Any, last_index: Sequence[int]) -> Any:
    """Select the next-token distribution at the last real prompt position."""
    logits = outputs.logits
    rows = []
    for index, position in enumerate(last_index):
        rows.append(logits[index, int(position), :])
    import torch

    return torch.stack(rows, dim=0)


def torch_option_loss(last_logits: Any, batch: CollatedBatch) -> Any:
    import torch
    import torch.nn.functional as F

    losses = []
    for index, (token_ids, target) in enumerate(
        zip(batch.option_token_ids, batch.target_probs, strict=True)
    ):
        ids = torch.tensor(token_ids, device=last_logits.device, dtype=torch.long)
        gathered = last_logits[index].index_select(0, ids)
        log_probs = F.log_softmax(gathered.float(), dim=-1)
        target_t = torch.tensor(target, device=last_logits.device, dtype=log_probs.dtype)
        losses.append(-(target_t * log_probs).sum())
    return torch.stack(losses).mean()
