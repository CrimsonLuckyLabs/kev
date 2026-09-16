"""HTTP surface for Kev. Same typed contract as the Python client. No chat endpoint."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from kev.backends import list_models, same_backend
from kev.client import KevClient
from kev.infer import DEFAULT_MAX_STATE_CHARS
from kev.types import SystemOneRequest, SystemOneResponse


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

    return application


app = create_app()
