"""Rule-based toy tickets. Labels are deterministic so a smoke LoRA can overfit."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from pathlib import Path

from kev.train.dataset import DecisionDataset, DecisionRow, LabelSpec, QuestionSpec

TOPICS = ("billing", "tech", "sales")
TONES = ("calm", "frustrated", "angry")
URGENCIES = ("low", "mid", "high")

BILLING_INSTRUCTIONS = "Is this about billing?"
TONE_INSTRUCTIONS = "What is the customer's tone?"
URGENCY_INSTRUCTIONS = "How urgent is this?"
TONE_CRITERIA = {"calm": None, "frustrated": None, "angry": None}
URGENCY_LEVELS = ["can wait", "this week", "today"]

BILLING_TEMPLATES = (
    "I was charged twice for order {oid}.",
    "Please refund invoice {oid}.",
    "The invoice for order {oid} does not match.",
)
TECH_TEMPLATES = (
    "The dashboard for workspace {oid} crashed once.",
    "Login fails on account {oid} after the update.",
    "Export for workspace {oid} is missing a column.",
)
SALES_TEMPLATES = (
    "Can we upgrade workspace {oid} to the team plan?",
    "Need a quote for workspace {oid}.",
    "What is the price for adding seats on {oid}?",
)

TONE_TAILS = {
    "calm": "Thanks for looking when you can.",
    "frustrated": "This is annoying and keeps happening.",
    "angry": "I am furious. This is idiot-level service.",
}
URGENCY_TAILS = {
    "low": "No rush, it can wait.",
    "mid": "Please handle this week if possible.",
    "high": "This is urgent — the site is down ASAP.",
}


def is_billing_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("charged", "refund", "invoice"))


def is_high_urgency_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("asap", "urgent", "down"))


def angry_ish_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("furious", "idiot", "twice"))


def compose_ticket(
    index: int,
    topic: str,
    tone: str,
    urgency: str,
) -> dict[str, str]:
    oid = f"A-{1000 + index}"
    if topic == "billing":
        base = BILLING_TEMPLATES[index % len(BILLING_TEMPLATES)].format(oid=oid)
    elif topic == "tech":
        base = TECH_TEMPLATES[index % len(TECH_TEMPLATES)].format(oid=oid)
    else:
        base = SALES_TEMPLATES[index % len(SALES_TEMPLATES)].format(oid=oid)
    text = f"{base} {TONE_TAILS[tone]} {URGENCY_TAILS[urgency]}"
    return {
        "id": f"t{index:03d}",
        "topic": topic,
        "tone": tone,
        "urgency": urgency,
        "ticket": text,
    }


def generate_toy_tickets(n: int = 200, seed: int = 0) -> list[dict[str, str]]:
    del seed  # combinatorics are deterministic; seed kept for call-site stability
    combos = list(product(TOPICS, TONES, URGENCIES))
    tickets: list[dict[str, str]] = []
    index = 0
    while len(tickets) < n:
        topic, tone, urgency = combos[index % len(combos)]
        tickets.append(compose_ticket(index, topic, tone, urgency))
        index += 1
    return tickets


def _peaked(winner: str, options: Sequence[str], mass: float = 0.78) -> dict[str, float]:
    rest = (1.0 - mass) / (len(options) - 1)
    return {name: (mass if name == winner else rest) for name in options}


def billing_row(ticket: dict[str, str]) -> DecisionRow:
    text = ticket["ticket"]
    positive = is_billing_text(text)
    teacher_p = 0.93 if positive else 0.08
    return DecisionRow(
        id=f"{ticket['id']}-billing",
        state={"ticket": text, "topic": ticket["topic"]},
        question=QuestionSpec(
            id="billing",
            type="noul",
            instructions=BILLING_INSTRUCTIONS,
        ),
        label=LabelSpec(noul=1.0 if positive else 0.0),
        teacher_p=teacher_p,
        source="toy",
    )


def tone_row(ticket: dict[str, str]) -> DecisionRow:
    text = ticket["ticket"]
    if angry_ish_text(text):
        winner = "angry"
    elif "annoying" in text.lower() or ticket["tone"] == "frustrated":
        winner = "frustrated"
    else:
        winner = "calm"
    return DecisionRow(
        id=f"{ticket['id']}-tone",
        state={"ticket": text, "topic": ticket["topic"]},
        question=QuestionSpec(
            id="tone",
            type="choice",
            instructions=TONE_INSTRUCTIONS,
            criteria=dict(TONE_CRITERIA),
        ),
        label=LabelSpec(choice=winner),
        teacher_probs=_peaked(winner, list(TONE_CRITERIA)),
        source="toy",
    )


def urgency_row(ticket: dict[str, str]) -> DecisionRow:
    text = ticket["ticket"]
    if is_high_urgency_text(text):
        winner = "today"
        index = 2
    elif "this week" in text.lower() or ticket["urgency"] == "mid":
        winner = "this week"
        index = 1
    else:
        winner = "can wait"
        index = 0
    return DecisionRow(
        id=f"{ticket['id']}-urgency",
        state={"ticket": text, "topic": ticket["topic"]},
        question=QuestionSpec(
            id="urgency",
            type="score",
            instructions=URGENCY_INSTRUCTIONS,
            criteria=list(URGENCY_LEVELS),
        ),
        label=LabelSpec(level=winner, index=index),
        teacher_probs=_peaked(winner, URGENCY_LEVELS),
        source="toy",
    )


def tickets_to_rows(tickets: Sequence[dict[str, str]]) -> list[DecisionRow]:
    rows: list[DecisionRow] = []
    for ticket in tickets:
        rows.append(billing_row(ticket))
        rows.append(tone_row(ticket))
        rows.append(urgency_row(ticket))
    return rows


def write_toy_splits(
    train_path: str | Path = "data/sft/toy.jsonl",
    eval_path: str | Path = "data/eval/toy.jsonl",
    n_tickets: int = 200,
    eval_tickets: int = 40,
) -> tuple[DecisionDataset, DecisionDataset]:
    if eval_tickets >= n_tickets:
        raise ValueError("eval_tickets must be smaller than n_tickets")
    tickets = generate_toy_tickets(n_tickets)
    train_ds = DecisionDataset(tickets_to_rows(tickets[eval_tickets:]))
    eval_ds = DecisionDataset(tickets_to_rows(tickets[:eval_tickets]))
    train_ds.to_jsonl(train_path)
    eval_ds.to_jsonl(eval_path)
    return train_ds, eval_ds
