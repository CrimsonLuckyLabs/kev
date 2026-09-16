"""Kev CLI. Decisions only. It does not chat."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from kev import Choice, KevClient, Noul, Score, __version__
from kev.primitives import parse_questions

app = typer.Typer(
    name="kev",
    help="Kev: local System One engine. It does not chat.",
    no_args_is_help=True,
)
console = Console()

DEMO_STATE = {
    "ticket": "I was charged twice. Please help ASAP.",
    "plan": "pro",
}
DEMO_QUESTIONS: dict[str, Noul | Choice | Score] = {
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


def parse_id_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected id=value, got {raw!r}")
    question_id, _, value = raw.partition("=")
    question_id = question_id.strip()
    if not question_id:
        raise ValueError("empty question id")
    return question_id, value


def collect_ask_questions(
    question: list[str] | None,
    noul: list[str] | None,
    choice: list[str] | None,
    score: list[str] | None,
) -> dict[str, Noul | Choice | Score]:
    items: list[tuple[str, Noul | Choice | Score]] = []
    seen: set[str] = set()

    def _add(question_id: str, parsed: Noul | Choice | Score) -> None:
        if question_id in seen:
            raise ValueError(f"duplicate question id: {question_id}")
        seen.add(question_id)
        items.append((question_id, parsed))

    for spec in question or []:
        _add(*parse_ask_question(spec))
    for spec in noul or []:
        question_id, instructions = parse_id_value(spec)
        _add(question_id, Noul(instructions=instructions))
    for spec in choice or []:
        question_id, raw_options = parse_id_value(spec)
        labels = [item.strip() for item in raw_options.split(",") if item.strip()]
        _add(
            question_id,
            Choice(
                instructions=f"What is the {question_id}?",
                criteria={label: None for label in labels},
            ),
        )
    for spec in score or []:
        question_id, raw_levels = parse_id_value(spec)
        levels = [item.strip() for item in raw_levels.split("|") if item.strip()]
        _add(
            question_id,
            Score(instructions=f"How {question_id} is this?", criteria=levels),
        )
    if not items:
        raise ValueError("provide --noul, --choice, --score, and/or --question")
    return parse_questions(items)


def parse_ask_question(spec: str) -> tuple[str, Noul | Choice | Score]:
    try:
        question_id, question_type, rest = spec.split(":", 2)
    except ValueError as exc:
        raise ValueError("question must be id:type:instructions") from exc
    if not question_id.strip():
        raise ValueError("empty question id")
    kind = question_type.lower()
    if kind == "noul":
        return question_id, Noul(instructions=rest)
    if kind == "choice":
        instructions, separator, options = rest.partition("|")
        if not separator:
            raise ValueError("choice questions need id:choice:instructions|opt1,opt2")
        labels = [item.strip() for item in options.split(",") if item.strip()]
        return question_id, Choice(
            instructions=instructions,
            criteria={label: None for label in labels},
        )
    if kind == "score":
        instructions, separator, levels = rest.partition("|")
        if not separator:
            raise ValueError("score questions need id:score:instructions|level1,level2")
        criteria = [item.strip() for item in levels.split(",") if item.strip()]
        return question_id, Score(instructions=instructions, criteria=criteria)
    raise ValueError(f"unknown question type: {question_type}")


T = TypeVar("T")


def _fail_loud(exc: ValueError) -> NoReturn:
    console.print(f"[bold red]invalid question[/bold red] {exc}")
    raise typer.Exit(code=1) from exc


def _guarded(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except ValueError as exc:
        _fail_loud(exc)
    except RuntimeError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command()
def demo(
    model: str = typer.Option("mock", "--model"),
    adapter: str | None = typer.Option(None, "--adapter"),
) -> None:
    """Run the canonical ticket-triage System One call. Offline when --model mock."""

    def _run() -> object:
        with KevClient(model=model, adapter=adapter) as client:
            return client.system_one(DEMO_STATE, DEMO_QUESTIONS)

    response = _guarded(_run)
    _print_response(response, title=f"kev demo ({model})")


@app.command()
def ask(
    state: str = typer.Option(..., "--state"),
    model: str = typer.Option("mock", "--model"),
    question: list[str] | None = typer.Option(None, "--question"),
    noul: list[str] | None = typer.Option(None, "--noul"),
    choice: list[str] | None = typer.Option(None, "--choice"),
    score: list[str] | None = typer.Option(None, "--score"),
    adapter: str | None = typer.Option(None, "--adapter"),
) -> None:
    """Ask typed questions over a state. Does not chat."""

    def _run() -> object:
        questions = collect_ask_questions(question, noul, choice, score)
        with KevClient(model=model, adapter=adapter) as client:
            return client.system_one(state, questions)

    response = _guarded(_run)
    from kev.types import SystemOneResponse

    assert isinstance(response, SystemOneResponse)
    console.print_json(data=json.loads(response.model_dump_json()))


@app.command()
def serve(
    model: str = typer.Option("mock", "--model"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8787, "--port"),
    adapter: str | None = typer.Option(None, "--adapter"),
) -> None:
    """Serve the console at / and POST /v1/systemone."""
    import uvicorn

    from kev.access import api_key
    from kev.dash import dash_password
    from kev.limits import http_limits
    from kev.server import create_app

    try:
        application = create_app(model=model, adapter=adapter)
    except RuntimeError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"Kev console  http://{host}:{port}/")
    console.print("POST          /v1/systemone")
    if dash_password():
        console.print(f"desk          http://{host}:{port}/dash")
    else:
        console.print("desk          locked until KEV_DASH_PASSWORD is set")
    caps = http_limits()
    console.print(
        "limits        "
        f"judge {caps['judge_per_min']}/min/ip · "
        f"inflight {caps['max_inflight']} · "
        f"{caps['max_questions']} questions · "
        f"{caps['max_state_chars']} state chars"
    )
    console.print("auth          bearer" if api_key() else "auth          open")
    uvicorn.run(application, host=host, port=port, reload=False)


@app.command("eval")
def eval_command(
    cases: Path = typer.Option(Path("data/eval/smoke.jsonl"), "--cases"),
    model: str = typer.Option("mock", "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    teacher_openai: bool = typer.Option(False, "--teacher-openai"),
    adapter: str | None = typer.Option(None, "--adapter"),
    out_dir: Path = typer.Option(Path("artifacts/eval"), "--out-dir"),
) -> None:
    """Score gold cases. Reports accuracy, Brier, ECE, and latency."""
    if teacher_openai:
        console.print(
            "kev eval --teacher-openai is a stub. It does not call OpenAI. "
            "Use --model mock, --model qwen2.5-1.5b, or --base-url."
        )
        raise typer.Exit(code=1)
    from kev.eval.harness import evaluate_path

    def _run() -> tuple[dict[str, Any], Path]:
        return evaluate_path(
            cases,
            model=model,
            base_url=base_url,
            adapter=adapter,
            out_dir=out_dir,
        )

    report, path = _guarded(_run)
    table = Table(title="kev eval")
    table.add_column("metric")
    table.add_column("value")
    agreement = report["accuracy"]
    table.add_row("cases", str(report["n_cases"]))
    table.add_row("answers", str(report["n_answers"]))
    table.add_row("accuracy", "—" if agreement is None else f"{float(agreement):.3f}")
    brier = report.get("brier")
    table.add_row("brier", "—" if brier is None else f"{float(brier):.3f}")
    by_type = report["agreement"]
    table.add_row("noul", "—" if by_type["noul"] is None else f"{by_type['noul']:.3f}")
    table.add_row("choice", "—" if by_type["choice"] is None else f"{by_type['choice']:.3f}")
    table.add_row("score", "—" if by_type["score"] is None else f"{by_type['score']:.3f}")
    latency = report["latency_ms"]
    table.add_row("latency_p50_ms", f"{latency['p50']:.2f}")
    table.add_row("latency_p95_ms", f"{latency['p95']:.2f}")
    ece = report["calibration"]["ece"]
    table.add_row("ece", "—" if ece is None else f"{ece:.3f}")
    table.add_row("report", str(path))
    console.print(table)


def _print_response(response: object, title: str) -> None:
    table = Table(title=title)
    table.add_column("id")
    table.add_column("type")
    table.add_column("value")
    from kev.types import SystemOneResponse

    assert isinstance(response, SystemOneResponse)
    for name, answer in response.answers.items():
        if answer.type == "noul":
            table.add_row(name, "noul", f"{answer.noul:.4f}")
        elif answer.type == "choice":
            table.add_row(
                name,
                "choice",
                f"{answer.choice} conf={answer.confidence:.4f} {json.dumps(answer.probabilities)}",
            )
        else:
            table.add_row(
                name,
                "score",
                f"{answer.score:.4f} conf={answer.confidence:.4f} "
                f"{json.dumps(answer.probabilities)}",
            )
    console.print(table)
    console.print(
        f"id={response.id} "
        f"model={response.model} "
        f"input_tokens={response.usage.input_tokens} "
        f"output_tokens={response.usage.output_tokens} "
        f"latency_ms={response.usage.latency_ms:.2f}"
    )


if __name__ == "__main__":
    app()
