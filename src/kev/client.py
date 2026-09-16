"""Public System One client. In-process backend, or HTTP against a Kev server."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from kev.backends import get_backend
from kev.backends.base import Backend
from kev.infer import DEFAULT_MAX_STATE_CHARS, system_one
from kev.primitives import Question, dump_questions
from kev.types import State, SystemOneResponse

DEFAULT_MODEL = "kev-latest"
DEFAULT_TIMEOUT = 120.0
RETRY_429 = 3
RETRY_AFTER_CAP_S = 30.0


def _http_error(response: httpx.Response) -> ValueError:
    try:
        payload: Any = response.json()
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    except Exception:
        detail = response.text
    return ValueError(f"Kev HTTP {response.status_code}: {detail}")


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit is None:
        found = os.environ.get("KEV_API_KEY", "").strip()
        return found or None
    stripped = explicit.strip()
    return stripped or None


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _retry_after_s(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After") or "1"
    try:
        wait = float(raw)
    except ValueError:
        wait = 1.0
    return min(RETRY_AFTER_CAP_S, max(0.0, wait))


def _systemone_payload(
    model: str,
    state: State,
    questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
) -> dict[str, Any]:
    return {"model": model, "state": state, "questions": dump_questions(questions)}


def _post_systemone(
    http: httpx.Client, payload: dict[str, Any], headers: dict[str, str]
) -> httpx.Response:
    response: httpx.Response | None = None
    for attempt in range(RETRY_429 + 1):
        response = http.post("/v1/systemone", json=payload, headers=headers)
        if response.status_code != 429 or attempt >= RETRY_429:
            return response
        time.sleep(_retry_after_s(response))
    assert response is not None
    return response


async def _apost_systemone(
    http: httpx.AsyncClient, payload: dict[str, Any], headers: dict[str, str]
) -> httpx.Response:
    response: httpx.Response | None = None
    for attempt in range(RETRY_429 + 1):
        response = await http.post("/v1/systemone", json=payload, headers=headers)
        if response.status_code != 429 or attempt >= RETRY_429:
            return response
        await asyncio.sleep(_retry_after_s(response))
    assert response is not None
    return response


class KevClient:
    """System One client.

    `base_url=None` runs an in-process backend.
    `base_url="http://127.0.0.1:8787"` calls POST /v1/systemone on that server.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        backend: Backend | None = None,
        seed: int | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        adapter: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_state_chars = max_state_chars
        self.seed = seed
        self.adapter = adapter
        self.api_key = _resolve_api_key(api_key)
        self.base_url = base_url.rstrip("/") if base_url else None
        self._http = http_client
        self._owns_http = False
        if http_client is not None or self.base_url:
            if self._http is None:
                assert self.base_url is not None
                self._http = httpx.Client(base_url=self.base_url, timeout=timeout)
                self._owns_http = True
            self.backend = None
        else:
            self.backend = backend or get_backend(
                model, seed=seed, device=device, dtype=dtype, adapter=adapter
            )

    def system_one(
        self,
        state: State,
        questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
        model: str | None = None,
    ) -> SystemOneResponse:
        requested = model or self.model
        if self._http is not None:
            return self._system_one_remote(state, questions, requested)
        if self.backend is None:
            raise RuntimeError("KevClient has no backend and no HTTP client")
        return system_one(
            state,
            questions,
            backend=self.backend,
            model=requested,
            max_state_chars=self.max_state_chars,
        )

    def _system_one_remote(
        self,
        state: State,
        questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
        model: str,
    ) -> SystemOneResponse:
        assert self._http is not None
        response = _post_systemone(
            self._http,
            _systemone_payload(model, state, questions),
            _auth_headers(self.api_key),
        )
        if response.status_code >= 400:
            raise _http_error(response)
        return SystemOneResponse.model_validate(response.json())

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            self._http.close()
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> KevClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncKevClient:
    """Async client. Remote uses httpx.AsyncClient; in-process uses to_thread."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        max_state_chars: int = DEFAULT_MAX_STATE_CHARS,
        backend: Backend | None = None,
        seed: int | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        adapter: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = _resolve_api_key(api_key)
        self._http = http_client
        self._owns_http = False
        self._sync: KevClient | None
        if http_client is not None or self.base_url:
            if self._http is None:
                assert self.base_url is not None
                self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
                self._owns_http = True
            self._sync = None
        else:
            self._sync = KevClient(
                model=model,
                device=device,
                dtype=dtype,
                max_state_chars=max_state_chars,
                backend=backend,
                seed=seed,
                adapter=adapter,
            )

    async def system_one(
        self,
        state: State,
        questions: Mapping[str, Question | Mapping[str, Any]] | Sequence[Any],
        model: str | None = None,
    ) -> SystemOneResponse:
        requested = model or self.model
        if self._http is not None:
            response = await _apost_systemone(
                self._http,
                _systemone_payload(requested, state, questions),
                _auth_headers(self.api_key),
            )
            if response.status_code >= 400:
                raise _http_error(response)
            return SystemOneResponse.model_validate(response.json())
        if self._sync is None:
            raise RuntimeError("AsyncKevClient has no backend and no HTTP client")
        return await asyncio.to_thread(self._sync.system_one, state, questions, requested)

    async def aclose(self) -> None:
        if self._owns_http and self._http is not None:
            await self._http.aclose()
        if self._sync is not None:
            self._sync.close()

    async def __aenter__(self) -> AsyncKevClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
