"""One-tick text-game latency demo. Not a Doom port."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from kev import Choice, KevClient, Noul

console = Console()

STATE = {
    "health": 27,
    "enemy_near": True,
    "ammo": 4,
    "facing": "east",
}

QUESTIONS = {
    "hold_trigger": Noul(instructions="Should the agent hold the trigger this tick?"),
    "move": Choice(
        instructions="Which move is safest this tick?",
        criteria=["forward", "left", "right", "back"],
    ),
    "goal": Choice(
        instructions="What is the immediate goal?",
        criteria=["survive", "hunt", "loot", "exit"],
    ),
}


def main() -> None:
    with KevClient(model="mock") as client:
        response = client.system_one(STATE, QUESTIONS)
    table = Table(title="doom toy — one tick")
    table.add_column("field")
    table.add_column("value")
    table.add_row("health", str(STATE["health"]))
    table.add_row("enemy_near", str(STATE["enemy_near"]))
    table.add_row("ammo", str(STATE["ammo"]))
    table.add_row("facing", str(STATE["facing"]))
    table.add_row("hold_trigger", f"{response.nouls['hold_trigger'].noul:.3f}")
    table.add_row("move", response.choices["move"].choice)
    table.add_row("goal", response.choices["goal"].choice)
    table.add_row("latency_ms", f"{response.usage.latency_ms:.2f}")
    table.add_row("input_tokens", str(response.usage.input_tokens))
    table.add_row("output_tokens", str(response.usage.output_tokens))
    console.print(table)


if __name__ == "__main__":
    main()
