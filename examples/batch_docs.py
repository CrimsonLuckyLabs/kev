"""Batch documents: the same 8 questions over 5 short docs. Latency demo."""

from __future__ import annotations

import time

from rich.console import Console
from rich.table import Table

from kev import Choice, KevClient, Noul, Score

console = Console()

DOCS = [
    {"id": "d1", "text": "Q3 revenue grew 12% year over year."},
    {"id": "d2", "text": "Please reset my password. I cannot log in."},
    {"id": "d3", "text": "The reactor pressure exceeded the safety threshold."},
    {"id": "d4", "text": "Team lunch is at noon in the cafeteria."},
    {"id": "d5", "text": "SSN 000-00-0000 appeared in the attached spreadsheet."},
]

QUESTIONS = {
    "sensitive": Noul(instructions="Does this document contain sensitive operational risk?"),
    "pii": Noul(instructions="Does this document contain personal data or credentials?"),
    "actionable": Noul(instructions="Does this document require an operator action?"),
    "needs_human": Noul(instructions="Should a human review this document before routing?"),
    "kind": Choice(
        instructions="What kind of document is this?",
        criteria=["finance", "support", "safety", "other"],
    ),
    "audience": Choice(
        instructions="Who is the primary audience?",
        criteria=["internal", "customer", "regulator", "unknown"],
    ),
    "severity": Score(
        instructions="How severe are the consequences of ignoring this?",
        criteria=["negligible", "moderate", "severe"],
    ),
    "clarity": Score(
        instructions="How clearly is the request or fact stated?",
        criteria=["vague", "usable", "precise"],
    ),
}


def main() -> None:
    table = Table(title="batch docs")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("sensitive")
    table.add_column("latency_ms")
    total_ms = 0.0
    with KevClient(model="mock") as client:
        for doc in DOCS:
            started = time.perf_counter()
            response = client.system_one(state=doc, questions=QUESTIONS)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            total_ms += elapsed_ms
            table.add_row(
                str(doc["id"]),
                response.choices["kind"].choice,
                f"{response.nouls['sensitive'].noul:.3f}",
                f"{elapsed_ms:.2f}",
            )
    console.print(table)
    console.print(f"docs={len(DOCS)} questions={len(QUESTIONS)} total_ms={total_ms:.2f}")


if __name__ == "__main__":
    main()
