from __future__ import annotations

import pytest

from kev.backends.mock import MockBackend, mock_noul_probability
from kev.client import KevClient
from kev.infer import system_one
from kev.primitives import Choice, Noul, Score
from kev.prompt import serialize_state
from kev.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, entropy_confidence


def test_mock_is_deterministic(ticket_state, ticket_questions) -> None:
    backend = MockBackend()
    first = system_one(ticket_state, ticket_questions, backend=backend)
    second = system_one(ticket_state, ticket_questions, backend=backend)
    assert first.answers == second.answers
    assert first.usage.output_tokens == 0
    assert second.usage.output_tokens == 0


def test_mock_hash_includes_question_id(ticket_state) -> None:
    question = Noul(instructions="Does the customer request a refund?")
    backend = MockBackend()
    left = system_one(ticket_state, {"alpha": question}, backend=backend)
    right = system_one(ticket_state, {"beta": question}, backend=backend)
    assert left.nouls["alpha"].noul != right.nouls["beta"].noul
    again = system_one(ticket_state, {"alpha": question}, backend=backend)
    assert left.nouls["alpha"] == again.nouls["alpha"]


def test_many_questions_share_one_state_encode(ticket_state, ticket_questions, monkeypatch) -> None:
    calls: list[object] = []
    original = serialize_state

    def wrapped(state: object) -> str:
        calls.append(state)
        return original(state)  # type: ignore[arg-type]

    monkeypatch.setattr("kev.infer.state_to_text", wrapped)
    system_one(ticket_state, ticket_questions, backend=MockBackend())
    assert len(calls) == 1


def test_noul_range_and_no_confidence_field(ticket_state) -> None:
    response = system_one(
        ticket_state,
        {"refund": Noul(instructions="Refund?")},
        backend=MockBackend(),
    )
    answer = response.nouls["refund"]
    assert isinstance(answer, NoulAnswer)
    assert 0.05 < answer.noul < 0.95
    assert "confidence" not in answer.model_dump()


def test_asap_boosts_urgency_like_noul() -> None:
    state = "I was charged twice. Please help ASAP."
    boosted = mock_noul_probability(state, "urgency", "Is this urgent?")
    ordinary = mock_noul_probability(state, "billing", "Is this about billing?")
    from_instructions = mock_noul_probability("calm note", "flag", "Handle this ASAP")
    assert boosted > 0.7
    assert from_instructions > 0.7
    assert 0.05 < ordinary < 0.95


def test_choice_cannot_invent_labels_and_is_peaked_not_one_hot(ticket_state) -> None:
    question = Choice(
        instructions="Topic?",
        criteria={"billing": None, "technical": None, "other": None},
    )
    response = system_one(ticket_state, {"topic": question}, backend=MockBackend())
    answer = response.choices["topic"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice in question.options
    assert set(answer.probabilities) == set(question.options)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert all(mass > 0.0 for mass in answer.probabilities.values())
    assert max(answer.probabilities.values()) < 1.0
    assert max(answer.probabilities.values()) > min(answer.probabilities.values())
    assert answer.confidence == pytest.approx(
        entropy_confidence(list(answer.probabilities.values()))
    )


def test_score_is_expected_index_and_entropy_confidence(ticket_state) -> None:
    question = Score(instructions="Urgency?", criteria=["can wait", "this week", "today"])
    response = system_one(ticket_state, {"urgency": question}, backend=MockBackend())
    answer = response.scores["urgency"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.levels == question.criteria
    expected = sum(
        index * answer.probabilities[level] for index, level in enumerate(answer.levels)
    )
    assert answer.score == pytest.approx(expected)
    assert 0.0 <= answer.score <= 2.0
    assert all(mass > 0.0 for mass in answer.probabilities.values())
    assert max(answer.probabilities.values()) < 1.0
    assert answer.confidence == pytest.approx(
        entropy_confidence(list(answer.probabilities[level] for level in answer.levels))
    )


def test_different_state_can_change_noul() -> None:
    question = {"q": Noul(instructions="Is this urgent?")}
    backend = MockBackend()
    a = system_one("calm request for a statement copy", question, backend=backend)
    b = system_one("URGENT outage, production is down", question, backend=backend)
    assert a.nouls["q"].noul != b.nouls["q"].noul


def test_seed_changes_mock_output(ticket_state) -> None:
    question = {"billing": Noul(instructions="Is this about billing?")}
    plain = system_one(ticket_state, question, backend=MockBackend())
    seeded = system_one(ticket_state, question, backend=MockBackend(seed=99))
    assert plain.nouls["billing"].noul != seeded.nouls["billing"].noul


def test_client_demo_shape(ticket_state, ticket_questions) -> None:
    with KevClient(model="mock") as client:
        response = client.system_one(ticket_state, ticket_questions)
    assert response.model == "mock"
    assert response.usage.output_tokens == 0
    assert response.usage.input_tokens > 0
    assert response.request_id
    assert response.id.startswith("kev_")
    assert set(response.answers) == set(ticket_questions)
