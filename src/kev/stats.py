"""In-process visit log. Unique visitors are cookie ids, not accounts."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

MAX_EVENTS = 10_000
SKIP_PREFIXES = ("/dash", "/static", "/docs", "/redoc", "/openapi")
SKIP_PATHS = {"/healthz", "/health", "/favicon.ico"}


@dataclass
class Event:
    ts: float
    method: str
    path: str
    status: int
    ip: str
    vid: str
    ua: str


@dataclass
class StatsLog:
    events: list[Event] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self,
        *,
        method: str,
        path: str,
        status: int,
        ip: str,
        vid: str,
        ua: str,
    ) -> None:
        route = path.split("?", 1)[0]
        if route in SKIP_PATHS or route.startswith(SKIP_PREFIXES):
            return
        event = Event(
            ts=time.time(),
            method=method.upper(),
            path=route,
            status=int(status),
            ip=ip[:80],
            vid=vid[:64],
            ua=ua[:180],
        )
        with self._lock:
            self.events.append(event)
            if len(self.events) > MAX_EVENTS:
                self.events = self.events[-MAX_EVENTS:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self.events)
        now = time.time()
        day = now - 86400
        pageviews = [e for e in rows if e.method == "GET" and e.path in {"/", "/index.html"}]
        judges = [e for e in rows if e.method == "POST" and e.path == "/v1/systemone"]
        day_rows = [e for e in rows if e.ts >= day]
        users: dict[str, dict[str, Any]] = {}
        for event in rows:
            slot = users.setdefault(
                event.vid or event.ip,
                {
                    "id": event.vid or event.ip,
                    "ip": event.ip,
                    "ua": event.ua,
                    "visits": 0,
                    "judges": 0,
                    "first_ts": event.ts,
                    "last_ts": event.ts,
                    "last_path": event.path,
                },
            )
            slot["visits"] += 1
            if event.method == "POST" and event.path == "/v1/systemone":
                slot["judges"] += 1
            slot["ip"] = event.ip
            slot["ua"] = event.ua
            slot["first_ts"] = min(float(slot["first_ts"]), event.ts)
            slot["last_ts"] = max(float(slot["last_ts"]), event.ts)
            slot["last_path"] = event.path
        ranked = sorted(users.values(), key=lambda item: float(item["last_ts"]), reverse=True)
        paths: dict[str, int] = defaultdict(int)
        for event in rows:
            paths[f"{event.method} {event.path}"] += 1
        recent = [
            {
                "ts": event.ts,
                "method": event.method,
                "path": event.path,
                "status": event.status,
                "ip": event.ip,
                "vid": event.vid,
            }
            for event in reversed(rows[-80:])
        ]
        return {
            "totals": {
                "events": len(rows),
                "pageviews": len(pageviews),
                "judges": len(judges),
                "users": len(users),
                "last_24h": len(day_rows),
            },
            "users": ranked[:100],
            "paths": dict(sorted(paths.items(), key=lambda item: item[1], reverse=True)[:20]),
            "recent": recent,
        }
