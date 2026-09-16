"""PEFT LoRA SFT on CUDA with bitsandbytes 4-bit. No MLX.

Loads atomic-decision JSONL. Each step samples one row. Loss is
option-token CE at the last prompt position (Noul over YES/NO;
Choice/Score over the supplied keys). The student is not trained
to emit JSON.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kev.train.dataset import DecisionDataset, DecisionRow, _first_token

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA = "data/sft/toy.jsonl"
DEFAULT_OUT = "artifacts/kev-7b-lora"
DEFAULT_ITERS = 200
DEFAULT_MAX_SEQ = 1024
DEFAULT_LORA_RANK = 8
DEFAULT_LR = 1e-5
BNB_INSTALL = "pip install bitsandbytes"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def row_text(row: DecisionRow) -> str:
    """Prompt plus the closed-set label. Never a JSON payload."""
    return row.prompt_text() + row.hard_label()


def encode_ids(tokenizer: Any, text: str, max_seq: int) -> list[int]:
    if max_seq < 1:
        raise ValueError("max_seq must be >= 1")
    ids = tokenizer.encode(text, add_special_tokens=False)
    values = [int(token) for token in ids]
    if not values:
        raise ValueError("tokenizer produced no ids")
    if len(values) > max_seq:
        return values[-max_seq:]
    return values


def row_training_example(
    row: DecisionRow,
    tokenizer: Any,
    max_seq: int,
) -> tuple[list[int], list[int], list[float]]:
    """Prompt ids plus first-token ids for the closed option set."""

    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(token) for token in ids]

    ids = encode_ids(tokenizer, row.prompt_text(), max_seq)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for row {row.id}")
    option_ids = [_first_token(encode, label) for label in row.option_labels()]
    return ids, option_ids, row.target_probs()


def option_token_loss(logits: Any, option_ids: Sequence[int], target: Sequence[float]) -> Any:
    """CE over YES/NO or option-key ids at the last prompt position."""
    import torch

    last = logits[0, -1]
    index = torch.tensor(list(option_ids), device=last.device, dtype=torch.long)
    gathered = last.index_select(0, index)
    logp = torch.log_softmax(gathered.float(), dim=-1)
    mass = torch.tensor(list(target), device=last.device, dtype=torch.float32)
    return -(mass * logp).sum()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kev PEFT option-token LoRA SFT (Qwen Instruct, bitsandbytes 4-bit CUDA)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    return parser.parse_args(argv)


def _train(args: argparse.Namespace, dataset: DecisionDataset) -> Path:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.iters < 1:
        raise ValueError("iters must be >= 1")
    if args.lora_rank < 1:
        raise ValueError("lora_rank must be >= 1")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_rank) * 2,
            lora_dropout=0.0,
            target_modules=list(LORA_TARGETS),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.train()

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Kev SFT must not call generate()")

    try:
        model.generate = _blocked
    except Exception:
        pass

    device = next(p for p in model.parameters() if p.requires_grad).device
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(args.lr),
    )
    n_rows = len(dataset)
    rng = random.Random()
    loss_value = 0.0
    for step in range(1, int(args.iters) + 1):
        row = dataset.rows[rng.randrange(n_rows)]
        token_ids, option_ids, target = row_training_example(row, tokenizer, int(args.max_seq))
        input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
        outputs = model(input_ids=input_ids)
        loss = option_token_loss(outputs.logits, option_ids, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value = float(loss.item())
        if step % 20 == 0 or step == int(args.iters):
            print(f"iter={step}/{args.iters} loss={loss_value:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    print(f"saved adapter {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = DecisionDataset.from_jsonl(args.data)
    kinds = {row.question.type for row in dataset}
    print(f"dataset rows={len(dataset)} kinds={sorted(kinds)} model={args.model}")
    if not has_cuda():
        print("Skipping PEFT training (CUDA required). Dataset validated.")
        raise SystemExit(0)
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print(f"bitsandbytes is required for 4-bit CUDA LoRA.\nRun: {BNB_INSTALL}")
        raise SystemExit(1)
    _train(args, dataset)


if __name__ == "__main__":
    main(sys.argv[1:])
