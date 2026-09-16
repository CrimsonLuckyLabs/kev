"""MLX LoRA on option-token logits. Qwen2.5-1.5B only. Never generate()."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kev.backends.mlx_qwen import DEFAULT_MLX_REPO, MLX_INSTALL, require_local_snapshot
from kev.prompt import Qwen25ChatFormat
from kev.train.dataset import DecisionDataset, DecisionRow, _first_token
from kev.train.sft import option_cross_entropy

logger = logging.getLogger("kev.train.mlx_lora")

MLX_LORA_RANK = 8
MLX_LORA_ALPHA = 16
MLX_LORA_DROPOUT = 0.0
MLX_LORA_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_SEQ = 1024
DEFAULT_LR = 1e-5
DEFAULT_ITERS = 200
DEFAULT_OUT = "artifacts/kev-1p5-lora"
ALLOWED_MODEL_MARKERS = ("qwen2.5-1.5b", "qwen2.5-1.5b-instruct", "1.5b-instruct")


def has_mlx() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlx_lm") is not None
    )


def assert_qwen_1_5b(model: str) -> None:
    lowered = model.lower().replace("_", "-")
    if any(marker in lowered for marker in ALLOWED_MODEL_MARKERS):
        return
    if lowered in {"kev-latest", "qwen2.5-1.5b", "qwen2.5-instruct"}:
        return
    raise ValueError(
        "MLX LoRA trains Qwen2.5-1.5B only. "
        f"Got {model!r}. Use mlx-community/Qwen2.5-1.5B-Instruct-4bit."
    )


def adapter_config(
    *,
    rank: int = MLX_LORA_RANK,
    alpha: int = MLX_LORA_ALPHA,
    dropout: float = MLX_LORA_DROPOUT,
    num_layers: int = -1,
    keys: Sequence[str] = MLX_LORA_KEYS,
) -> dict[str, Any]:
    """mlx-lm load_adapters() shape. scale is LoRA alpha."""
    return {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": int(rank),
            "scale": float(alpha),
            "dropout": float(dropout),
            "keys": list(keys),
        },
    }


def clip_prompt_ids(ids: Sequence[int], max_seq: int) -> list[int]:
    if max_seq < 1:
        raise ValueError("max_seq must be >= 1")
    values = [int(token) for token in ids]
    if len(values) <= max_seq:
        return values
    return values[-max_seq:]


def encode_text(tokenizer: Any, text: str) -> list[int]:
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        ids = inner.encode(text, add_special_tokens=False)
    except TypeError:
        ids = inner.encode(text)
    return [int(token) for token in ids]


def row_training_example(
    row: DecisionRow,
    tokenizer: Any,
    max_seq: int = DEFAULT_MAX_SEQ,
) -> tuple[list[int], list[int], list[float]]:
    def encode(text: str) -> list[int]:
        return encode_text(tokenizer, text)

    ids = clip_prompt_ids(encode(row.prompt_text(Qwen25ChatFormat())), max_seq)
    if not ids:
        raise ValueError(f"tokenizer produced no ids for row {row.id}")
    option_ids = [_first_token(encode, label) for label in row.option_labels()]
    return ids, option_ids, row.target_probs()


def write_adapter(out_dir: str | Path, model: Any, config: dict[str, Any]) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(target / "adapters.safetensors"), weights)
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    logger.info("saved MLX LoRA adapter to %s", target)
    return target


def _import_train_stack() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_map
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        raise RuntimeError(MLX_INSTALL) from exc
    return mx, nn, optim, tree_map, load, linear_to_lora_layers


def _forbid_generate(model: Any) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Kev SFT must not call generate()")

    try:
        model.generate = _blocked
    except Exception:
        logger.debug("could not patch model.generate")


def option_token_loss(
    model: Any,
    mx: Any,
    nn: Any,
    tokens: Any,
    option_ids: Any,
    target: Any,
) -> Any:
    """CE over YES/NO or option-key ids at the last prompt position."""
    logits = model(tokens)
    last = logits[0, -1, :]
    gathered = last[option_ids]
    logp = nn.log_softmax(gathered.astype(mx.float32), axis=-1)
    return -(target.astype(mx.float32) * logp).sum()


def collect_option_logit_pairs(
    rows: Sequence[DecisionRow],
    *,
    model: Any,
    tokenizer: Any,
    mx: Any,
    max_seq: int = DEFAULT_MAX_SEQ,
) -> list[tuple[list[float], list[float]]]:
    pairs: list[tuple[list[float], list[float]]] = []
    for row in rows:
        token_ids, option_ids, target = row_training_example(row, tokenizer, max_seq)
        tokens = mx.array(token_ids)[None]
        logits = model(tokens)
        last = logits[0, -1, :].astype(mx.float32)
        gathered = last[mx.array(option_ids)]
        mx.eval(gathered)
        pairs.append(([float(value) for value in gathered.tolist()], list(target)))
    return pairs


def train_option_lora(
    dataset: DecisionDataset,
    *,
    model_name: str = DEFAULT_MLX_REPO,
    out_dir: str | Path = DEFAULT_OUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    max_seq: int = DEFAULT_MAX_SEQ,
    lora_rank: int = MLX_LORA_RANK,
    lora_alpha: int = MLX_LORA_ALPHA,
    lr: float = DEFAULT_LR,
    iters: int = DEFAULT_ITERS,
    seed: int = 0,
) -> Path:
    """Train LoRA by scoring option tokens. Does not train JSON generation."""
    del batch_size  # always 1 example per microbatch; accum is the effective batch
    assert_qwen_1_5b(model_name)
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    imported = _import_train_stack()
    mx, nn, optim, tree_map, load, linear_to_lora_layers = imported
    mx.random.seed(seed)
    local = require_local_snapshot(model_name)
    model, tokenizer = load(local)
    _forbid_generate(model)
    model.freeze()
    config = adapter_config(rank=lora_rank, alpha=lora_alpha)
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    model.train()
    optimizer = optim.AdamW(learning_rate=lr)

    def loss_fn(mdl: Any, tokens: Any, option_ids: Any, target: Any) -> Any:
        return option_token_loss(mdl, mx, nn, tokens, option_ids, target)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    n_rows = len(dataset)
    step_loss = 0.0
    for iteration in range(1, iters + 1):
        acc_grads: Any | None = None
        micro_losses: list[float] = []
        for micro in range(grad_accum):
            row = dataset.rows[(iteration * grad_accum + micro) % n_rows]
            token_ids, option_ids, target = row_training_example(row, tokenizer, max_seq)
            tokens = mx.array(token_ids)[None]
            opt_ids = mx.array(option_ids)
            tgt = mx.array(target)
            loss, grads = loss_and_grad(model, tokens, opt_ids, tgt)
            mx.eval(loss)
            micro_losses.append(float(loss.item()))
            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = tree_map(lambda left, right: left + right, acc_grads, grads)
        assert acc_grads is not None
        scale = 1.0 / float(grad_accum)
        acc_grads = tree_map(lambda grad: grad * scale, acc_grads)
        optimizer.update(model, acc_grads)
        mx.eval(model.parameters(), optimizer.state)
        step_loss = sum(micro_losses) / len(micro_losses)
        if iteration == 1 or iteration % 10 == 0 or iteration == iters:
            logger.info("iter=%s/%s option_ce=%.4f", iteration, iters, step_loss)
    out = write_adapter(out_dir, model, config)
    logger.info("final option_ce=%.4f out=%s", step_loss, out)
    return out


def python_option_ce(
    last_vocab: Sequence[float],
    option_ids: Sequence[int],
    target: Sequence[float],
) -> float:
    gathered = []
    vocab = len(last_vocab)
    for token_id in option_ids:
        if 0 <= int(token_id) < vocab:
            gathered.append(float(last_vocab[int(token_id)]))
        else:
            gathered.append(float("-inf"))
    return option_cross_entropy(gathered, target)
