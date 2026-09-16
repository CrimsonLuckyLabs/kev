from __future__ import annotations

import pytest

from kev import Choice, KevClient, Noul, Score


@pytest.fixture
def ticket_state() -> dict[str, str]:
    return {
        "ticket": "I was charged twice for order A-104. Please refund the duplicate ASAP.",
        "policy": "Duplicate charges are eligible for a refund.",
    }


@pytest.fixture
def ticket_questions() -> dict[str, Noul | Choice | Score]:
    return {
        "refund_requested": Noul(instructions="Does the customer request a refund?"),
        "topic": Choice(
            instructions="What is this ticket about?",
            criteria={"billing": None, "technical": None, "other": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    }


@pytest.fixture
def client() -> KevClient:
    return KevClient(model="mock")
