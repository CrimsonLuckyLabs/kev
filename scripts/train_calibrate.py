"""Fit temperature T on frozen option logits. Minimize NLL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kev.backends.mlx_qwen import DEFAULT_MLX_REPO
from kev.calibration import save_temperature, temperature_path
from kev.train.calibrate import fit_temperature, teacher_logit_pairs
from kev.train.dataset import DecisionDataset
from kev.train.mlx_lora import DEFAULT_MAX_SEQ, DEFAULT_OUT, assert_qwen_1_5b, has_mlx

logger = logging.getLogger("kev.train_calibrate")


def _collect_mlx_pairs(
    dataset: DecisionDataset,
    *,
    model_name: str,
    adapter: Path,
    max_seq: int,
) -> list[tuple[list[float], list[float]]]:
    from kev.backends.mlx_qwen import MLXQwenBackend, find_local_snapshot
    from kev.train.mlx_lora import collect_option_logit_pairs

    if find_local_snapshot(model_name) is None:
        raise RuntimeError("mlx weights missing")
    adapter_arg = str(adapter) if (adapter / "adapter_config.json").is_file() else None
    backend = MLXQwenBackend(repo=model_name, adapter=adapter_arg)
    backend.engine.temperature = 1.0
    try:
        return collect_option_logit_pairs(
            dataset.rows,
            model=backend.engine.model,
            tokenizer=backend.engine.tokenizer,
            mx=backend.engine.mx,
            max_seq=max_seq,
        )
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit Kev temperature on eval JSONL.")
    parser.add_argument("--data", default="data/eval/toy.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MLX_REPO)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    assert_qwen_1_5b(args.model)
    dataset = DecisionDataset.from_jsonl(args.data)
    logger.info("eval rows=%s", len(dataset))
    out_dir = Path(args.out)
    adapter = Path(args.adapter) if args.adapter else out_dir
    pairs: list[tuple[list[float], list[float]]] | None = None
    if has_mlx():
        try:
            pairs = _collect_mlx_pairs(
                dataset,
                model_name=args.model,
                adapter=adapter,
                max_seq=args.max_seq,
            )
            logger.info("fitted T on frozen MLX option logits rows=%s", len(pairs))
        except Exception as exc:
            logger.info("MLX logits unavailable (%s); using teacher-derived smoke logits.", exc)
    if pairs is None:
        pairs = teacher_logit_pairs(dataset.rows)
        logger.info("fitting T on teacher-derived option logits (smoke).")
    temperature = fit_temperature(pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = temperature_path(out_dir)
    save_temperature(dest, temperature)
    print(f"temperature={temperature:.4f} wrote {dest}")


if __name__ == "__main__":
    main(sys.argv[1:])
