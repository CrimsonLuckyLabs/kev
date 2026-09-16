"""In-process visit log. Unique visitors are cookie ids, not accounts."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MAX_EVENTS = 10_000
MAX_JUDGES = 80
STATE_CHARS = 2_000
SKIP_PREFIXES = ("/dash", "/static", "/docs", "/redoc", "/openapi")
SKIP_PATHS = {"/healthz", "/health", "/favicon.ico"}


def clip_state(state: object, limit: int = STATE_CHARS) -> str:
    if isinstance(state, str):
        text = state
    else:
        try:
            text = json.dumps(state, default=str)
        except TypeError:
            text = str(state)
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def clip_questions(questions: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, question in questions.items():
        if hasattr(question, "model_dump"):
            data = question.model_dump(mode="json", exclude_none=True)
        elif isinstance(question, Mapping):
            data = dict(question)
        else:
            data = {"instructions": str(question)}
        row: dict[str, Any] = {
            "id": str(qid),
            "type": data.get("type"),
            "instructions": str(data.get("instructions") or "")[:240],
        }
        criteria = data.get("criteria")
        if isinstance(criteria, dict):
            row["options"] = [str(key) for key in list(criteria)[:32]]
        elif isinstance(criteria, list):
            row["options"] = [str(item) for item in criteria[:32]]
        rows.append(row)
    return rows


def clip_answers(answers: Mapping[str, Any]) -> dict[str, Any]:
    dumped: dict[str, Any] = {}
    for name, answer in answers.items():
        if hasattr(answer, "model_dump"):
            dumped[str(name)] = answer.model_dump(mode="json")
        elif isinstance(answer, Mapping):
            dumped[str(name)] = dict(answer)
        else:
            dumped[str(name)] = {"value": str(answer)}
    return dumped


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
class JudgeRecord:
    ts: float
    vid: str
    ip: str
    request_id: str
    model: str
    latency_ms: float
    status: int
    state: str
    questions: list[dict[str, Any]]
    answers: dict[str, Any]


@dataclass
class StatsLog:
    events: list[Event] = field(default_factory=list)
    judges: list[JudgeRecord] = field(default_factory=list)
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

    def record_judge(
        self,
        *,
        vid: str,
        ip: str,
        state: object,
        questions: Mapping[str, Any],
        answers: Mapping[str, Any],
        model: str,
        request_id: str,
        latency_ms: float,
        status: int = 200,
    ) -> None:
        row = JudgeRecord(
            ts=time.time(),
            vid=vid[:64],
            ip=ip[:80],
            request_id=request_id[:80],
            model=model[:120],
            latency_ms=float(latency_ms),
            status=int(status),
            state=clip_state(state),
            questions=clip_questions(questions),
            answers=clip_answers(answers),
        )
        with self._lock:
            self.judges.append(row)
            if len(self.judges) > MAX_JUDGES:
                self.judges = self.judges[-MAX_JUDGES:]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self.events)
            judge_rows = list(self.judges)
        now = time.time()
        day = now - 86400
        pageviews = [e for e in rows if e.method == "GET" and e.path in {"/", "/index.html"}]
        judge_hits = [e for e in rows if e.method == "POST" and e.path == "/v1/systemone"]
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
                "judges": len(judge_hits),
                "users": len(users),
                "last_24h": len(day_rows),
            },
            "users": ranked[:100],
            "judges": [
                {
                    "ts": item.ts,
                    "vid": item.vid,
                    "ip": item.ip,
                    "id": item.request_id,
                    "model": item.model,
                    "latency_ms": item.latency_ms,
                    "status": item.status,
                    "state": item.state,
                    "questions": item.questions,
                    "answers": item.answers,
                }
                for item in reversed(judge_rows[-MAX_JUDGES:])
            ],
            "paths": dict(sorted(paths.items(), key=lambda item: item[1], reverse=True)[:20]),
            "recent": recent,
        }
