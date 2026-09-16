"""Build the toy System One JSONL splits. Labels are rule-based."""

from __future__ import annotations

import argparse

from kev.train.toy import write_toy_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Write rule-labeled toy decision JSONL.")
    parser.add_argument("--train-out", default="data/sft/toy.jsonl")
    parser.add_argument("--eval-out", default="data/eval/toy.jsonl")
    parser.add_argument("--tickets", type=int, default=200)
    parser.add_argument("--eval-tickets", type=int, default=40)
    args = parser.parse_args()
    train_ds, eval_ds = write_toy_splits(
        train_path=args.train_out,
        eval_path=args.eval_out,
        n_tickets=args.tickets,
        eval_tickets=args.eval_tickets,
    )
    print(
        f"wrote {args.train_out} rows={len(train_ds)} "
        f"and {args.eval_out} rows={len(eval_ds)}"
    )


if __name__ == "__main__":
    main()
