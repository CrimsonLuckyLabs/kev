"""Public System One client. In-process backend, or HTTP against a Kev server."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from kev.backends import get_backend
from kev.backends.base import Backend
from kev.infer import DEFAULT_MAX_STATE_CHARS, system_one
from kev.primitives import Question, dump_questions
from kev.types import State, SystemOneResponse

DEFAULT_MODEL = "kev-latest"


def _http_error(response: httpx.Response) -> ValueError:
    try:
        payload: Any = response.json()
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    except Exception:
        detail = response.text
    return ValueError(f"Kev HTTP {response.status_code}: {detail}")


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
        timeout: float = 60.0,
        adapter: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_state_chars = max_state_chars
        self.seed = seed
        self.adapter = adapter
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
        response = self._http.post(
            "/v1/systemone",
            json={
                "model": model,
                "state": state,
                "questions": dump_questions(questions),
            },
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
        timeout: float = 60.0,
        adapter: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
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
            response = await self._http.post(
                "/v1/systemone",
                json={
                    "model": requested,
                    "state": state,
                    "questions": dump_questions(questions),
                },
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
