"""MLX LoRA SFT on option-token distributions. Does not teach JSON generation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kev.backends.mlx_qwen import DEFAULT_MLX_REPO, MLX_INSTALL, download_command
from kev.train.dataset import DecisionDataset, HashTokenizer
from kev.train.mlx_lora import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM,
    DEFAULT_ITERS,
    DEFAULT_LR,
    DEFAULT_MAX_SEQ,
    DEFAULT_OUT,
    MLX_LORA_ALPHA,
    MLX_LORA_RANK,
    assert_qwen_1_5b,
    has_mlx,
    train_option_lora,
)
from kev.train.sft import batch_option_loss

logger = logging.getLogger("kev.train_sft")


def _dry_run_batch(dataset: DecisionDataset) -> tuple[int, int]:
    tokenizer = HashTokenizer.from_rows(dataset.rows)
    sample = dataset.rows[: min(4, len(dataset))]
    batch = dataset.collate(sample, tokenizer=tokenizer)
    vocab_logits = [[0.0] * (max(ids) + 1 if ids else 1) for ids in batch.option_token_ids]
    for row_logits, token_ids in zip(vocab_logits, batch.option_token_ids, strict=True):
        for token_id in token_ids:
            if token_id >= len(row_logits):
                row_logits.extend([0.0] * (token_id + 1 - len(row_logits)))
            row_logits[token_id] = 1.0
    loss = batch_option_loss(vocab_logits, batch)
    logger.info("dry-run option CE=%.4f tokenizer=hash", loss)
    return len(batch), batch.seq_len


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kev MLX option-token LoRA SFT (Qwen2.5-1.5B).")
    parser.add_argument("--model", default=DEFAULT_MLX_REPO)
    parser.add_argument("--data", default="data/sft/toy.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lora-rank", type=int, default=MLX_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=MLX_LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert_qwen_1_5b(args.model)
    dataset = DecisionDataset.from_jsonl(args.data)
    kinds = {row.question.type for row in dataset}
    logger.info("dataset rows=%s kinds=%s", len(dataset), sorted(kinds))
    batch_size, seq_len = _dry_run_batch(dataset)
    logger.info("collated batch_size=%s seq_len=%s", batch_size, seq_len)
    if args.dry_run or not has_mlx():
        hint = MLX_INSTALL if not has_mlx() else "dry-run"
        print(
            f"Skipping LoRA training ({hint}). Dataset validated and one batch collated "
            f"(size={batch_size}, seq_len={seq_len})."
        )
        raise SystemExit(0)
    from kev.backends.mlx_qwen import find_local_snapshot

    if find_local_snapshot(args.model) is None:
        print(f"MLX weights not found for {args.model}.\nRun: {download_command(args.model)}")
        raise SystemExit(1)
    out = train_option_lora(
        dataset,
        model_name=args.model,
        out_dir=Path(args.out),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq=args.max_seq,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        iters=args.iters,
        seed=args.seed,
    )
    print(f"saved adapter {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
