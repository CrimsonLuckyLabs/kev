from __future__ import annotations

from pathlib import Path

from kev.train.dataset import DecisionDataset, HashTokenizer
from kev.train.sft import batch_option_loss, option_cross_entropy
from kev.train.toy import (
    angry_ish_text,
    generate_toy_tickets,
    is_billing_text,
    is_high_urgency_text,
    tickets_to_rows,
    write_toy_splits,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "sft" / "example.jsonl"


def test_example_jsonl_schema() -> None:
    dataset = DecisionDataset.from_jsonl(EXAMPLE)
    assert len(dataset) == 3
    kinds = [row.question.type for row in dataset]
    assert kinds == ["noul", "choice", "score"]
    noul = dataset.rows[0]
    assert noul.label.noul == 1.0
    assert noul.teacher_p == 0.93
    assert noul.target_probs()[0] == 0.93
    choice = dataset.rows[1]
    assert choice.label.choice == "angry"
    assert choice.teacher_probs is not None
    assert abs(sum(choice.teacher_probs.values()) - 1.0) < 1e-9
    score = dataset.rows[2]
    assert score.label.level == "today"
    assert score.label.index == 2


def test_toy_rules_are_deterministic() -> None:
    tickets = generate_toy_tickets(27)
    billing = next(item for item in tickets if item["topic"] == "billing")
    high = next(item for item in tickets if item["urgency"] == "high")
    angry = next(item for item in tickets if item["tone"] == "angry")
    assert is_billing_text(billing["ticket"])
    assert is_high_urgency_text(high["ticket"])
    assert angry_ish_text(angry["ticket"])
    rows = tickets_to_rows([billing, high, angry])
    billing_row = next(row for row in rows if row.question.id == "billing")
    assert billing_row.label.noul == 1.0
    urgency_row = next(
        row for row in rows if row.id.startswith(high["id"]) and row.question.type == "score"
    )
    assert urgency_row.label.level == "today"
    tone_row = next(
        row for row in rows if row.id.startswith(angry["id"]) and row.question.type == "choice"
    )
    assert tone_row.label.choice == "angry"


def test_collate_one_batch(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_ds, eval_ds = write_toy_splits(
        train_path=train_path,
        eval_path=eval_path,
        n_tickets=12,
        eval_tickets=3,
    )
    assert len(train_ds) == 9 * 3
    assert len(eval_ds) == 3 * 3
    loaded = DecisionDataset.from_jsonl(train_path)
    tokenizer = HashTokenizer.from_rows(loaded.rows)
    batch = loaded.collate(loaded.rows[:6], tokenizer=tokenizer)
    assert len(batch) == 6
    assert batch.seq_len == max(len(ids) for ids in batch.input_ids)
    paired = zip(batch.attention_mask, batch.last_index, strict=True)
    assert all(mask[position] == 1 for mask, position in paired)
    assert "noul" in batch.kinds
    noul_index = batch.kinds.index("noul")
    assert batch.option_labels[noul_index] == ["YES", "NO"]
    assert abs(sum(batch.target_probs[noul_index]) - 1.0) < 1e-9
    vocab = max(max(ids) for ids in batch.option_token_ids) + 1
    logits = [[0.0] * vocab for _ in range(len(batch))]
    for row, token_ids, target in zip(
        logits, batch.option_token_ids, batch.target_probs, strict=True
    ):
        winner = target.index(max(target))
        row[token_ids[winner]] = 3.0
    loss = batch_option_loss(logits, batch)
    assert 0.0 < loss < 2.0
    assert option_cross_entropy([3.0, 0.0], [1.0, 0.0]) < option_cross_entropy(
        [0.0, 3.0], [1.0, 0.0]
    )
