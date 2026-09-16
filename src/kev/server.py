"""HTTP surface for Kev. Same typed contract as the Python client. No chat endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from kev.backends import list_models, same_backend
from kev.client import KevClient
from kev.decks import list_decks, list_presets
from kev.infer import DEFAULT_MAX_STATE_CHARS
from kev.types import SystemOneRequest, SystemOneResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONSOLE_DIR = Path(__file__).resolve().parent / "console"
LEGACY_WEB_DIR = Path(__file__).resolve().parent / "web"
WEB_DIR = STATIC_DIR

FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Kev</title>
<style>
body{margin:0;background:#070807;color:#d7e0c8;font:13px/1.45 monospace;padding:1.2rem}
textarea{width:100%;min-height:8rem;background:#0b0d0b;color:#d7e0c8}
button{margin-top:.8rem;background:#c6f04a;border:0;padding:.5rem 1rem}
</style></head><body>
<h1>KEV SYSTEM ONE</h1>
<p>charged twice ASAP · checkout 500 · jailbreak</p>
<textarea id="state">I was charged twice. Refund now, ASAP.</textarea>
<p><button type="button" id="judge">Judge</button></p>
<pre id="out">Judge posts /v1/systemone</pre>
<script>
const Q = {
  billing: {type: "noul", instructions: "Is this about billing?"},
  tone: {
    type: "choice",
    instructions: "Tone?",
    criteria: {calm: null, frustrated: null, angry: null}
  },
  urgency: {
    type: "score",
    instructions: "Urgency?",
    criteria: ["can wait", "this week", "today"]
  }
};
document.getElementById("judge").onclick = async () => {
  const state = document.getElementById("state").value;
  const res = await fetch("/v1/systemone", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify({state: {ticket: state}, questions: Q})
  });
  const body = await res.json();
  document.getElementById("out").textContent = JSON.stringify(body, null, 2);
};
fetch("/v1/meta").then((r) => r.json()).then((m) => {
  document.title = "Kev · " + (m.model || "system one");
}).catch(() => {});
</script>
</body></html>
"""


def resolve_static_dir() -> Path | None:
    for path in (STATIC_DIR, CONSOLE_DIR, LEGACY_WEB_DIR):
        if (path / "index.html").is_file():
            return path
    return None


def load_console_html() -> str:
    directory = resolve_static_dir()
    if directory is not None:
        return (directory / "index.html").read_text(encoding="utf-8")
    try:
        from importlib.resources import files

        return files("kev").joinpath("static/index.html").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return FALLBACK_HTML


def create_app(
    model: str = "mock",
    device: str = "auto",
    dtype: str = "auto",
    max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
    seed: int | None = None,
    adapter: str | None = None,
) -> FastAPI:
    application = FastAPI(
        title="Kev",
        version="0.1.0",
        description="Local System One decision engine. It does not chat.",
    )
    application.state.client = KevClient(
        model=model,
        device=device,
        dtype=dtype,
        max_state_chars=max_state_chars,
        seed=seed,
        adapter=adapter,
    )

    @application.get("/healthz")
    @application.get("/health")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models")
    def models() -> dict[str, list[dict[str, object]]]:
        return {"models": list_models()}

    @application.get("/v1/meta")
    def meta() -> dict[str, str | None]:
        bound: KevClient = application.state.client
        return {"model": bound.model, "adapter": bound.adapter}

    @application.get("/v1/decks")
    def decks() -> dict[str, list[dict[str, object]]]:
        return {"decks": list_decks(), "presets": list_presets()}

    @application.get("/", response_class=HTMLResponse)
    @application.get("/index.html", response_class=HTMLResponse)
    def console() -> HTMLResponse:
        html = load_console_html()
        if not html.strip():
            html = FALLBACK_HTML
        return HTMLResponse(html, media_type="text/html; charset=utf-8")

    @application.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(payload: SystemOneRequest, http_request: Request) -> SystemOneResponse:
        client: KevClient = http_request.app.state.client
        requested = payload.model or client.model
        try:
            if not same_backend(requested, client.model):
                raise ValueError(
                    f"server is bound to model {client.model!r}; restart to use {requested!r}"
                )
            return client.system_one(payload.state, payload.questions, model=requested)
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    static = resolve_static_dir()
    if static is not None:
        application.mount("/static", StaticFiles(directory=static), name="static")
    return application


app = create_app()
