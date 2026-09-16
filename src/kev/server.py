"""HTTP surface for Kev. Same typed contract as the Python client. No chat endpoint."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from kev.access import GATE_COOKIE, api_key, bearer_ok, cors_origins
from kev.backends import list_models, same_backend
from kev.client import KevClient
from kev.dash import (
    DASH_COOKIE,
    LOGIN_HTML,
    TOKEN_TTL_S,
    VID_COOKIE,
    client_ip,
    dash_password,
    load_dash_html,
    password_ok,
    sign_token,
    token_ok,
)
from kev.decks import list_decks
from kev.limits import (
    DEFAULT_JUDGE_PER_MIN,
    DEFAULT_LOGIN_PER_MIN,
    DEFAULT_MAX_INFLIGHT,
    Inflight,
    WindowCounter,
    env_int,
    http_limits,
    http_state_chars,
    payload_error,
)
from kev.stats import StatsLog
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
<h1>KEV SYSTEM ZERO POINT ONE</h1>
<textarea id="state" placeholder="Paste unstructured state."></textarea>
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
</script>
</body></html>
"""


class DashLogin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str


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


def _secure_cookie(request: Request) -> bool:
    return request.headers.get("x-forwarded-proto", "").lower() == "https"


def _client_host(request: Request) -> str:
    return request.client.host if request.client is not None else ""


def _rate_limit(request: Request, bucket: str, limit: int) -> None:
    if limit <= 0:
        return
    rates: WindowCounter = request.app.state.rates
    ip = client_ip(request.headers, _client_host(request))
    allowed, retry = rates.allow(f"{bucket}:{ip}", limit)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit",
            headers={"Retry-After": str(retry)},
        )


def _set_gate_cookie(request: Request, response: HTMLResponse) -> None:
    key = api_key()
    if not key:
        return
    response.set_cookie(
        GATE_COOKIE,
        sign_token(key),
        httponly=True,
        samesite="lax",
        secure=_secure_cookie(request),
        max_age=TOKEN_TTL_S,
    )


def _require_judge_auth(request: Request) -> None:
    expected = api_key()
    if not expected:
        return
    header = request.headers.get("authorization")
    if header:
        if bearer_ok(header, expected):
            return
        raise HTTPException(status_code=401, detail="auth")
    if token_ok(expected, request.cookies.get(GATE_COOKIE)):
        return
    raise HTTPException(status_code=401, detail="auth")


def create_app(
    model: str = "mock",
    device: str = "auto",
    dtype: str = "auto",
    max_state_chars: int | None = None,
    seed: int | None = None,
    adapter: str | None = None,
) -> FastAPI:
    state_cap = http_state_chars() if max_state_chars is None else max_state_chars
    application = FastAPI(
        title="Kev",
        version="0.1.0",
        description="Local System One decision engine. It does not chat.",
    )
    application.state.client = KevClient(
        model=model,
        device=device,
        dtype=dtype,
        max_state_chars=state_cap,
        seed=seed,
        adapter=adapter,
    )
    application.state.stats = StatsLog()
    application.state.rates = WindowCounter()
    application.state.inflight = Inflight(env_int("KEV_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT))
    origins = cors_origins()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @application.middleware("http")
    async def track_visits(request: Request, call_next):  # type: ignore[no-untyped-def]
        vid = request.cookies.get(VID_COOKIE) or uuid.uuid4().hex
        response = await call_next(request)
        host = _client_host(request)
        application.state.stats.record(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ip=client_ip(request.headers, host),
            vid=vid,
            ua=request.headers.get("user-agent") or "",
        )
        if not request.cookies.get(VID_COOKIE):
            response.set_cookie(
                VID_COOKIE,
                vid,
                httponly=True,
                samesite="lax",
                secure=_secure_cookie(request),
                max_age=365 * 24 * 3600,
            )
        return response

    @application.get("/healthz")
    @application.get("/health")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/models")
    def models() -> dict[str, list[dict[str, object]]]:
        return {"models": list_models()}

    @application.get("/v1")
    def v1_index() -> dict[str, object]:
        return {
            "name": "KEV SYSTEM ZERO POINT ONE",
            "judge": {"method": "POST", "path": "/v1/systemone"},
            "auth": "bearer" if api_key() else "open",
            "endpoints": {
                "healthz": "/healthz",
                "meta": "/v1/meta",
                "models": "/v1/models",
                "decks": "/v1/decks",
                "docs": "/docs",
                "systemone": "/v1/systemone",
            },
            "limits": http_limits(),
        }

    @application.get("/v1/meta")
    def meta() -> dict[str, object]:
        bound: KevClient = application.state.client
        return {"model": bound.model, "adapter": bound.adapter, "limits": http_limits()}

    @application.get("/v1/decks")
    def decks() -> dict[str, list[dict[str, object]]]:
        return {"decks": list_decks()}

    @application.get("/", response_class=HTMLResponse)
    @application.get("/index.html", response_class=HTMLResponse)
    def console(request: Request) -> HTMLResponse:
        html = load_console_html()
        if not html.strip():
            html = FALLBACK_HTML
        response = HTMLResponse(html, media_type="text/html; charset=utf-8")
        _set_gate_cookie(request, response)
        return response

    @application.get("/dash", response_class=HTMLResponse)
    def dash(request: Request) -> HTMLResponse:
        if token_ok(dash_password(), request.cookies.get(DASH_COOKIE)):
            return HTMLResponse(load_dash_html(), media_type="text/html; charset=utf-8")
        return HTMLResponse(LOGIN_HTML, media_type="text/html; charset=utf-8")

    @application.post("/dash/login")
    def dash_login(payload: DashLogin, request: Request) -> JSONResponse:
        _rate_limit(request, "login", env_int("KEV_LOGIN_PER_MIN", DEFAULT_LOGIN_PER_MIN))
        expected = dash_password()
        if not expected:
            raise HTTPException(status_code=503, detail="dash unset")
        if not password_ok(payload.password, expected):
            raise HTTPException(status_code=403, detail="invalid")
        response = JSONResponse({"ok": True})
        response.set_cookie(
            DASH_COOKIE,
            sign_token(dash_password()),
            httponly=True,
            samesite="lax",
            secure=_secure_cookie(request),
            max_age=TOKEN_TTL_S,
        )
        return response

    @application.post("/dash/logout")
    def dash_logout() -> JSONResponse:
        response = JSONResponse({"ok": True})
        response.delete_cookie(DASH_COOKIE)
        return response

    @application.get("/dash/stats")
    def dash_stats(request: Request) -> dict[str, object]:
        if not token_ok(dash_password(), request.cookies.get(DASH_COOKIE)):
            raise HTTPException(status_code=401, detail="auth")
        stats: StatsLog = application.state.stats
        return stats.snapshot()

    @application.post("/v1/systemone", response_model=SystemOneResponse)
    def systemone(payload: SystemOneRequest, http_request: Request) -> SystemOneResponse:
        _require_judge_auth(http_request)
        heavy = payload_error(payload.questions)
        if heavy:
            raise HTTPException(status_code=400, detail=heavy)
        _rate_limit(
            http_request, "judge", env_int("KEV_JUDGE_PER_MIN", DEFAULT_JUDGE_PER_MIN)
        )
        gate: Inflight = http_request.app.state.inflight
        if not gate.acquire():
            raise HTTPException(
                status_code=429, detail="busy", headers={"Retry-After": "1"}
            )
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
        finally:
            gate.release()

    static = resolve_static_dir()
    if static is not None:
        application.mount("/static", StaticFiles(directory=static), name="static")
    return application


app = create_app()
