"""HTTP access: optional bearer key, console gate cookie, CORS allowlist."""

from __future__ import annotations

import os

from kev.dash import password_ok

GATE_COOKIE = "kev_gate"


def api_key() -> str:
    return os.environ.get("KEV_API_KEY", "").strip()


def cors_origins() -> list[str]:
    raw = os.environ.get("KEV_CORS_ORIGINS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def bearer_ok(authorization: str | None, expected: str) -> bool:
    if not expected or not authorization:
        return False
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return False
    return password_ok(token.strip(), expected)
