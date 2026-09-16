from __future__ import annotations

import asyncio

import pytest
from typer.testing import CliRunner

from kev import AsyncKevClient, Choice, KevClient, Noul, Score, __version__
from kev.cli import app
from kev.primitives import parse_questions

runner = CliRunner()

STATE = {
    "ticket": "I was charged twice. Please help ASAP.",
    "plan": "pro",
}
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


def test_convenience_views_match_public_api() -> None:
    client = KevClient(model="mock", device="auto", dtype="auto", max_state_chars=120_000)
    res = client.system_one(STATE, QUESTIONS)
    assert 0.05 < res.nouls["billing"].noul < 0.95
    assert res.choices["tone"].choice in QUESTIONS["tone"].options
    assert set(res.choices["tone"].probabilities) == set(QUESTIONS["tone"].options)
    assert all(mass > 0.0 for mass in res.choices["tone"].probabilities.values())
    assert 0.0 <= res.choices["tone"].confidence <= 1.0
    assert 0.0 <= res.scores["urgency"].score <= 2.0
    assert set(res.scores["urgency"].probabilities) == set(QUESTIONS["urgency"].criteria)
    assert res.usage.output_tokens == 0
    assert res.model == "mock"
    assert res.id.startswith("kev_")


def test_async_remote_client_asgi_transport() -> None:
    import httpx

    from kev.server import create_app

    async def _run() -> None:
        transport = httpx.ASGITransport(app=create_app(model="mock"))
        async with httpx.AsyncClient(transport=transport, base_url="http://kev.test") as http:
            async with AsyncKevClient(model="mock", http_client=http) as client:
                res = await client.system_one(STATE, QUESTIONS)
        assert res.id.startswith("kev_")
        assert "billing" in res.nouls

    asyncio.run(_run())


def test_async_client_system_one() -> None:
    async def _run() -> None:
        async with AsyncKevClient(model="mock") as client:
            res = await client.system_one(STATE, QUESTIONS)
        assert "billing" in res.nouls
        assert "tone" in res.choices
        assert "urgency" in res.scores

    asyncio.run(_run())


def test_client_context_manager(ticket_state) -> None:
    with KevClient(model="mock") as client:
        response = client.system_one(
            ticket_state,
            {"refund": Noul(instructions="Refund?")},
        )
    assert 0.05 < response.nouls["refund"].noul < 0.95
    assert response.usage.output_tokens == 0


def test_empty_instructions_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty instructions"):
        Noul(instructions="")
    with pytest.raises(ValueError, match="empty instructions"):
        Choice(instructions="  ", criteria={"a": None, "b": None})
    with pytest.raises(ValueError, match="empty instructions"):
        Score(instructions="", criteria=["low", "high"])


def test_choice_option_count_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 2 options"):
        Choice(instructions="Pick one", criteria={"only": None})
    too_many = {f"opt{i}": None for i in range(256)}
    with pytest.raises(ValueError, match="at most"):
        Choice(instructions="Pick one", criteria=too_many)


def test_score_level_count_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 2 levels"):
        Score(instructions="How urgent?", criteria=["today"])


def test_duplicate_question_ids_raise_value_error() -> None:
    with pytest.raises(ValueError, match="duplicate question id"):
        parse_questions(
            [
                ("billing", Noul(instructions="Is this about billing?")),
                ("billing", Noul(instructions="Is this about billing again?")),
            ]
        )


def test_max_state_chars_raises_value_error() -> None:
    client = KevClient(model="mock", max_state_chars=8)
    with pytest.raises(ValueError, match="max_state_chars"):
        client.system_one("this state is definitely too long", QUESTIONS)


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        KevClient(model="not-a-model")


def test_system_one_rejects_unknown_question_type() -> None:
    client = KevClient(model="mock")
    with pytest.raises(ValueError, match="unknown question type"):
        client.system_one("hello", {"bad": {"type": "chat", "instructions": "Say hi"}})


def test_qwen_alias_resolves_default_backend() -> None:
    from kev.backends import MLX_DEFAULT_REPO, display_model_id, on_darwin, resolve_model

    spec = resolve_model("kev-latest")
    spec15 = resolve_model("qwen2.5-1.5b")
    if on_darwin():
        assert spec.backend == "mlx_qwen"
        assert spec.repo == MLX_DEFAULT_REPO
        assert spec15.backend == "mlx_qwen"
    else:
        assert spec.backend in {"mlx_qwen", "hf_qwen"}
    assert spec15.repo is not None and "1.5B" in spec15.repo
    assert display_model_id("kev-latest") == "qwen2.5-1.5b"
    assert display_model_id("mock") == "mock"


def test_client_forwards_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_get_backend(name: str, **kwargs: object) -> object:
        seen["name"] = name
        seen.update(kwargs)
        from kev.backends.mock import MockBackend

        return MockBackend()

    monkeypatch.setattr("kev.client.get_backend", fake_get_backend)
    client = KevClient(model="kev-latest", adapter="artifacts/kev-toy-lora")
    assert seen["adapter"] == "artifacts/kev-toy-lora"
    assert client.adapter == "artifacts/kev-toy-lora"


def test_remote_client_does_not_load_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("remote KevClient must not load a local backend")

    monkeypatch.setattr("kev.client.get_backend", boom)
    with KevClient(base_url="http://127.0.0.1:8787", model="kev-latest") as client:
        assert client.backend is None
        assert client.base_url == "http://127.0.0.1:8787"


def test_qwen_backend_clear_error_without_runtime() -> None:
    import importlib.util

    from kev.backends.mlx_qwen import MLX_INSTALL

    if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
        with pytest.raises(RuntimeError, match="pip install mlx mlx-lm"):
            KevClient(model="qwen2.5-1.5b")
        assert MLX_INSTALL == "pip install mlx mlx-lm"
        return
    pytest.skip("mlx is installed; load coverage is @pytest.mark.gpu")


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_demo() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "billing" in result.stdout
    assert "tone" in result.stdout
    assert "urgency" in result.stdout
    assert "output_tokens=0" in result.stdout


def test_cli_demo_model_mock_offline() -> None:
    result = runner.invoke(app, ["demo", "--model", "mock"])
    assert result.exit_code == 0, result.output
    assert "model=mock" in result.stdout
    assert "output_tokens=0" in result.stdout


def test_cli_ask_unknown_type_fails_loud() -> None:
    result = runner.invoke(
        app,
        ["ask", "--state", "hello", "--question", "bad:chat:Say something nice"],
    )
    assert result.exit_code == 1
    assert "invalid question" in result.output
    assert "unknown question type" in result.output


def test_cli_ask_typed_flags() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "--model",
            "mock",
            "--state",
            "I was charged twice ASAP",
            "--noul",
            "billing=Is this about billing?",
            "--choice",
            "tone=calm,frustrated,angry",
            "--score",
            "urgency=can wait|this week|today",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "billing" in result.stdout
    assert "tone" in result.stdout
    assert "urgency" in result.stdout
    assert "output_tokens" in result.stdout


def test_cli_ask_duplicate_question_ids() -> None:
    result = runner.invoke(
        app,
        [
            "ask",
            "--state",
            "hello",
            "--question",
            "billing:noul:Is this billing?",
            "--question",
            "billing:noul:Is this billing again?",
        ],
    )
    assert result.exit_code == 1
    loud = f"{result.output}{result.exception}"
    assert "duplicate question id" in loud
    assert "invalid question" in result.output


def test_cli_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(asgi_app: object, host: str, port: int, reload: bool = False) -> None:
        called["host"] = host
        called["port"] = port
        called["reload"] = reload
        called["app"] = asgi_app

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["serve", "--model", "mock", "--port", "8787"])
    assert result.exit_code == 0
    assert called["port"] == 8787
    assert called["host"] == "0.0.0.0"


def test_cli_serve_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(asgi_app: object, host: str, port: int, reload: bool = False) -> None:
        called["host"] = host
        called["port"] = port
        called["app"] = asgi_app
        del reload

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(
        app, ["serve", "--model", "mock", "--host", "127.0.0.1", "--port", "8000"]
    )
    assert result.exit_code == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8000
