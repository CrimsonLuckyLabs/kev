"""Password gate for the operator dashboard. Secret is KEV_DASH_PASSWORD only."""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections.abc import Mapping
from pathlib import Path

DASH_COOKIE = "kev_dash"
VID_COOKIE = "kev_vid"
TOKEN_TTL_S = 60 * 60 * 24 * 14

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>KEV SYSTEM ZERO POINT ONE</title>
<style>
body{margin:0;background:#080908;color:#e4ecd4;font:16px/1.45 monospace;
min-height:100vh;display:grid;place-items:center;padding:1rem}
form{border:1px solid #2c3326;padding:1.4rem 1.5rem;background:#101210;width:min(22rem,100%)}
.mark{color:#c6f04a;letter-spacing:.2em}
.sub{color:#7d8a6c;letter-spacing:.06em;font-size:.8rem;margin-top:.35rem}
label{display:block;color:#7d8a6c;letter-spacing:.12em;font-size:.7rem;
text-transform:uppercase;margin:1rem 0 .35rem}
input{width:100%;background:#0b0d0b;color:#e4ecd4;border:1px solid #2c3326;
padding:.7rem;font:inherit}
button{margin-top:1rem;width:100%;min-height:2.8rem;background:#c6f04a;border:0;
padding:.5rem .9rem;letter-spacing:.16em;text-transform:uppercase;font:inherit;cursor:pointer}
.err{color:#ff5a3c;min-height:1.2rem;margin:.6rem 0 0}
</style></head><body>
<form id="f" method="post" action="/dash/login">
  <div class="mark">KEV</div>
  <div class="sub">SYSTEM ZERO POINT ONE</div>
  <label for="password">password</label>
  <input id="password" name="password" type="password" autocomplete="current-password" autofocus />
  <button type="submit">enter</button>
  <p class="err" id="err"></p>
</form>
<script>
document.getElementById("f").addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = document.getElementById("password").value;
  const res = await fetch("/dash/login", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify({password: password})
  });
  if (res.ok) { window.location.href = "/dash"; return; }
  document.getElementById("err").textContent = "no";
});
</script>
</body></html>
"""


def dash_password() -> str:
    return os.environ.get("KEV_DASH_PASSWORD", "").strip()


def password_ok(given: str, expected: str) -> bool:
    if not expected:
        return False
    left = hashlib.sha256(given.encode()).digest()
    right = hashlib.sha256(expected.encode()).digest()
    return hmac.compare_digest(left, right)


def client_ip(headers: Mapping[str, str], fallback: str) -> str:
    lowered = {key.lower(): value for key, value in headers.items()}
    cf = lowered.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    forwarded = lowered.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback


def sign_token(password: str, now: float | None = None) -> str:
    issued = str(int(now if now is not None else time.time()))
    digest = hmac.new(password.encode(), issued.encode(), hashlib.sha256).hexdigest()
    return f"{issued}.{digest}"


def token_ok(password: str, token: str | None) -> bool:
    if not password or not token or "." not in token:
        return False
    issued, digest = token.split(".", 1)
    try:
        ts = int(issued)
    except ValueError:
        return False
    if abs(time.time() - ts) > TOKEN_TTL_S:
        return False
    expected = hmac.new(password.encode(), issued.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, expected)


def load_dash_html() -> str:
    path = Path(__file__).resolve().parent / "static" / "dash.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    try:
        from importlib.resources import files

        return files("kev").joinpath("static/dash.html").read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return LOGIN_HTML
