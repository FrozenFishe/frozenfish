"""SQLite-backed audit log with query/export helpers."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from .security import make_event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  outcome TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(ts);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit(actor);
"""


def _parse_since(since: str | None) -> float | None:
    if not since:
        return None
    s = since.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if s.endswith(tuple(units)):
        return time.time() - float(s[:-1]) * units[s[-1]]
    # ISO date
    return datetime.fromisoformat(s).timestamp()


class AuditLog:
    """Append-only audit trail.

    Events are PII-redacted at construction time.  ``tail`` returns the most
    recent events; ``export`` dumps a full copy as JSON.
    """

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        if enabled:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        else:
            self._conn = None

    # -- write --------------------------------------------------------------
    def record(
        self,
        actor: str,
        action: str,
        target: str,
        outcome: str = "ok",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ev = make_event(actor, action, target, outcome, detail)
        ev["ts"] = time.time()
        if not self.enabled:
            return None
        cur = self._conn.execute(
            "INSERT INTO audit (ts, actor, action, target, outcome, detail) "
            "VALUES (?,?,?,?,?,?)",
            (
                ev["ts"],
                ev["actor"],
                ev["action"],
                ev["target"],
                ev["outcome"],
                json.dumps(ev["detail"], ensure_ascii=False),
            ),
        )
        ev["id"] = cur.lastrowid
        self._conn.commit()
        return ev

    # -- read -----------------------------------------------------------------
    def tail(
        self,
        since: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        where: list[str] = []
        params: list[Any] = []
        ts = _parse_since(since)
        if ts is not None:
            where.append("ts >= ?")
            params.append(ts)
        if actor:
            where.append("actor = ?")
            params.append(actor)
        if action:
            where.append("action = ?")
            params.append(action)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(
            f"SELECT id, ts, actor, action, target, outcome, detail FROM audit"
            f"{clause} ORDER BY ts DESC, id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "iso": datetime.fromtimestamp(r[1], timezone.utc).isoformat(),
                "actor": r[2],
                "action": r[3],
                "target": r[4],
                "outcome": r[5],
                "detail": json.loads(r[6]),
            }
            for r in rows
        ]

    def export(self, fmt: str = "json") -> str:
        """Export the whole log.  ``fmt`` is ``json`` (array of events)."""
        events = self.tail(limit=10**9)
        if fmt == "json":
            return json.dumps(events, indent=2, ensure_ascii=False)
        raise ValueError(f"unsupported export format: {fmt!r}")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.tail(limit=10**9))


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts for `ff audit tail --summary` style output."""
    by_actor: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for ev in events:
        by_actor[ev["actor"]] = by_actor.get(ev["actor"], 0) + 1
        by_outcome[ev["outcome"]] = by_outcome.get(ev["outcome"], 0) + 1
    return {"count": len(events), "by_actor": by_actor, "by_outcome": by_outcome}
