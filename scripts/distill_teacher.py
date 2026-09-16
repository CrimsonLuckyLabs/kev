"""Ask an OpenAI-compatible teacher for Kev labels. Kev still does not generate text."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from kev.train.distill import run_distill


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Distill typed Kev labels from an OpenAI-compatible teacher."
    )
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--in", dest="input_path", default="data/raw/tickets.jsonl")
    parser.add_argument("--questions", default="examples/triage_questions.yaml")
    parser.add_argument("--out", default="data/sft/teacher.jsonl")
    parser.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    api_key = (os.environ.get(args.api_key_env) or "").strip()
    rows = run_distill(
        questions_path=args.questions,
        out_path=args.out,
        input_path=args.input_path,
        max_rows=args.max_rows,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key or None,
    )
    if not api_key:
        print(
            f"No API key in {args.api_key_env}. Dry-run wrote {len(rows)} fake rows "
            f"to {args.out}. Rationale discarded; student still does not generate text."
        )
        return
    print(f"wrote {args.out} rows={len(rows)} source=teacher:{args.model}")


if __name__ == "__main__":
    main(sys.argv[1:])
