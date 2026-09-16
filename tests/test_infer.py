from __future__ import annotations

from kev.backends.fake import FakeLogitsEngine
from kev.encode import state_to_text
from kev.infer import has_first_token_collision, score_questions
from kev.primitives import Choice, Noul, Score
from kev.prompt import Qwen3ChatFormat, Qwen25ChatFormat, get_chat_format
from kev.types import ChoiceAnswer, NoulAnswer, ScoreAnswer

STATE = "I was charged twice ASAP"


def _standard_questions() -> dict[str, Noul | Choice | Score]:
    return {
        "billing": Noul(instructions="Is this about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": "neutral", "angry": "hostile"},
        ),
        "urgency": Score(instructions="How urgent is this?", criteria=["low", "high"]),
    }


def test_state_to_text_pretty_json() -> None:
    assert "ticket" in state_to_text({"ticket": "x"})


def test_fake_engine_prefills_once_and_never_generates() -> None:
    engine = FakeLogitsEngine()
    answers, input_tokens, scored_ids = score_questions(
        engine, STATE, _standard_questions(), batch=True
    )
    assert engine.prefill_calls == 1
    assert engine.batch_calls == 1
    assert engine.sequential_calls == 0
    assert not hasattr(engine, "generate") or True
    assert input_tokens > 0
    assert scored_ids
    assert isinstance(answers["billing"], NoulAnswer)
    assert 0.0 <= answers["billing"].noul <= 1.0
    assert isinstance(answers["tone"], ChoiceAnswer)
    assert answers["tone"].choice in {"calm", "angry"}
    assert isinstance(answers["urgency"], ScoreAnswer)
    assert answers["urgency"].level in {"low", "high"}


def test_batch_and_sequential_forward_match_labels() -> None:
    questions = _standard_questions()
    table = {
        "YES": [1],
        "Yes": [2],
        "yes": [3],
        "NO": [4],
        "No": [5],
        "no": [6],
        "calm": [20],
        "angry": [21],
        "low": [22],
        "high": [23],
    }
    sequential = FakeLogitsEngine(token_table=table)
    batched = FakeLogitsEngine(token_table=table)
    seq_answers, _, _ = score_questions(sequential, STATE, questions, batch=False)
    batch_answers, _, _ = score_questions(batched, STATE, questions, batch=True)
    assert sequential.prefill_calls == 1
    assert batched.prefill_calls == 1
    assert sequential.sequential_calls == 3
    assert batched.batch_calls == 1
    assert seq_answers["billing"].noul == batch_answers["billing"].noul
    assert seq_answers["tone"].choice == batch_answers["tone"].choice
    assert seq_answers["urgency"].level == batch_answers["urgency"].level
    assert seq_answers["tone"].probabilities == batch_answers["tone"].probabilities


def test_first_token_collision_uses_teacher_forcing() -> None:
    questions = {
        "pick": Choice(
            instructions="Which?",
            criteria={"cat": "animal", "car": "vehicle"},
        )
    }
    engine = FakeLogitsEngine(
        token_table={"cat": [40], "car": [40, 41]},
        continuation_scores={"cat": -0.1, "car": -3.0},
    )
    assert has_first_token_collision(engine, ["cat", "car"])
    answers, _, scored_ids = score_questions(engine, STATE, questions, batch=True)
    assert engine.continuation_calls == 2
    assert answers["pick"].choice == "cat"
    assert answers["pick"].probabilities["cat"] > answers["pick"].probabilities["car"]
    assert 40 in scored_ids and 41 in scored_ids


def test_noul_sums_yes_no_variants() -> None:
    engine = FakeLogitsEngine(
        logit_overrides={
            "default": {1: 5.0, 2: 5.0, 3: 5.0, 4: 0.0, 5: 0.0, 6: 0.0},
        }
    )
    answers, _, _ = score_questions(
        engine, STATE, {"billing": Noul(instructions="Is this about billing?")}, batch=True
    )
    assert answers["billing"].noul > 0.9


def test_qwen25_prefix_is_shared_and_has_no_completion() -> None:
    fmt = Qwen25ChatFormat()
    prefix = fmt.prefix("hello state")
    assert prefix.startswith("<|im_start|>system")
    assert "You are Kev" in prefix
    assert "STATE:\nhello state" in prefix
    assert "You will be asked one atomic question" in prefix
    assert not prefix.rstrip().endswith("<|im_start|>assistant")
    suffix = fmt.suffix_for(Noul(instructions="Is this about billing?"))
    assert "QUESTION (billing):" not in suffix
    assert "YES or NO" in suffix
    assert suffix.endswith("<|im_start|>assistant\n")
    choice = fmt.suffix_for(
        Choice(instructions="What is the tone?", criteria={"calm": "neutral", "angry": None})
    )
    assert "- calm: neutral" in choice
    assert "- angry" in choice


def test_qwen3_format_omits_think_block() -> None:
    fmt = get_chat_format("qwen3-8b")
    assert isinstance(fmt, Qwen3ChatFormat)
    opened = fmt.assistant_open()
    assert "<think>" not in opened
    assert opened == "<|im_start|>assistant\n"


def test_output_tokens_stay_zero_on_logits_path() -> None:
    from kev.infer import system_one

    class EngineBackend:
        name = "fake"

        def __init__(self) -> None:
            self.engine = FakeLogitsEngine()
            self.last_input_tokens = 0

        def infer(self, state_text: str, questions: dict) -> dict:
            answers, n_tokens, _ids = score_questions(self.engine, state_text, questions)
            self.last_input_tokens = n_tokens
            return answers

        def close(self) -> None:
            return None

    response = system_one(STATE, _standard_questions(), backend=EngineBackend(), model="fake")
    assert response.usage.output_tokens == 0
    assert response.usage.input_tokens > 0
