"""Ticket triage: deterministic rules first, then Kev, then code-owned routing."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from kev import Choice, KevClient, Noul, Score, SystemOneResponse

console = Console()

QUESTIONS = {
    "billing": Noul(instructions="Is this about billing?"),
    "tone": Choice(
        instructions="What is the customer's tone?",
        criteria={
            "calm": "neutral factual",
            "frustrated": "annoyed but civil",
            "angry": "hostile",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this?",
        criteria=["can wait", "this week", "today"],
    ),
}


def deterministic_route(state: dict[str, Any]) -> str | None:
    status = str(state.get("status", "")).lower()
    if status == "closed":
        return "no_action"
    return None


def compose_route(state: dict[str, Any], response: SystemOneResponse | None) -> str:
    locked = deterministic_route(state)
    if locked is not None:
        return locked
    if response is None:
        return "default"
    billing = response.nouls["billing"].noul
    urgency = response.scores["urgency"].score
    tone = response.choices["tone"]
    if billing > 0.8 and urgency >= 1.5:
        return "billing_prio"
    if tone.choice == "angry" and tone.confidence > 0.5:
        return "human"
    return "default"


def evaluate(client: KevClient, state: dict[str, Any]) -> tuple[str, SystemOneResponse | None]:
    if deterministic_route(state) is not None:
        return compose_route(state, None), None
    response = client.system_one(state, QUESTIONS)
    return compose_route(state, response), response


def main() -> None:
    tickets = [
        {
            "id": "T-100",
            "status": "closed",
            "ticket": "Old duplicate charge, already resolved.",
            "plan": "pro",
        },
        {
            "id": "T-101",
            "status": "open",
            "ticket": "I was charged twice. Please help ASAP.",
            "plan": "pro",
        },
        {
            "id": "T-102",
            "status": "open",
            "ticket": "The app crashed once yesterday. No rush.",
            "plan": "free",
        },
    ]
    table = Table(title="ticket triage")
    table.add_column("ticket")
    table.add_column("status")
    table.add_column("billing")
    table.add_column("tone")
    table.add_column("urgency")
    table.add_column("route")
    with KevClient(model="mock") as client:
        for state in tickets:
            route, response = evaluate(client, state)
            if response is None:
                table.add_row(str(state["id"]), str(state["status"]), "—", "—", "—", route)
                continue
            tone = response.choices["tone"]
            table.add_row(
                str(state["id"]),
                str(state["status"]),
                f"{response.nouls['billing'].noul:.2f}",
                f"{tone.choice} ({tone.confidence:.2f})",
                f"{response.scores['urgency'].score:.2f}",
                route,
            )
    console.print(table)


if __name__ == "__main__":
    main()
