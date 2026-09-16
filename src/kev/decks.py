"""Question packs for the consumer console. Code still owns the labels."""

from __future__ import annotations

from typing import Any

Deck = dict[str, Any]

TRIAGE: Deck = {
    "id": "triage",
    "title": "Triage",
    "blurb": "Support ticket → billing / tone / urgency.",
    "state_key": "ticket",
    "sample": (
        "I was charged twice on invoice A-19. This is furious. Refund now, ASAP."
    ),
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the customer's tone?",
            "criteria": {
                "calm": "neutral factual",
                "frustrated": "annoyed but civil",
                "angry": "hostile",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait", "this week", "today"],
        },
    },
}

TRADE: Deck = {
    "id": "trade",
    "title": "Trade",
    "blurb": "Blotter in → tradeable / side / size. Not a chatbot.",
    "state_key": "blotter",
    "sample": (
        "AAPL 14:12 ET. Spot 228.40, VWAP 227.10, ORH 229.80.\n"
        "Tape: offer thinning, no news catalyst.\n"
        "Account: 1R=$250, already -0.4R today. Max 1R.\n"
        "Plan: fade first break of ORH only if volume dies. Else stand down."
    ),
    "questions": {
        "tradeable": {
            "type": "noul",
            "instructions": "Is this a valid, executable setup right now?",
        },
        "side": {
            "type": "choice",
            "instructions": "Which side should software take?",
            "criteria": {
                "long": "buy / cover",
                "short": "sell / fade",
                "stand_down": "no trade",
            },
        },
        "size": {
            "type": "score",
            "instructions": "How large should the order be?",
            "criteria": ["skip", "quarter", "half", "full"],
        },
        "haste": {
            "type": "noul",
            "instructions": "Must this be acted on in the next few minutes?",
        },
    },
}

DECKS: dict[str, Deck] = {TRIAGE["id"]: TRIAGE, TRADE["id"]: TRADE}


def list_decks() -> list[Deck]:
    return [DECKS["triage"], DECKS["trade"]]
