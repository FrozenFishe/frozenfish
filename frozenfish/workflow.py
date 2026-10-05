"""Workflow engine: triggers, step execution, retry, dead-letter.

A workflow is a YAML document (parsed by :func:`frozenfish.config.miniyaml_load`):

.. code-block:: yaml

    name: email_summarize
    trigger: {type: cron, expr: "0 8 * * *"}      # cron | interval | event
    steps:
      - name: fetch
        tool: email.inbox
      - name: summarize
        tool: rag.search
        input: {query: "{{steps.fetch.result}}"}

Triggers: ``cron`` (5-field, stdlib-evaluated), ``interval`` (seconds),
``event`` (named queue, e.g. ``file:///.../.ff/events``).  Steps reference
prior outputs with ``{{steps.<name>.result}}``.  Each step retries
``workflows.retry.max_attempts`` with exponential backoff; after exhaustion
the run goes to the dead-letter queue.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from .config import miniyaml_load

# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


class _CronField:
    """One cron field: supports *, */n, a, a-b, a-b/n, comma lists."""

    def __init__(self, spec: str, lo: int, hi: int):
        self.values: set[int] = set()
        # "*" (bare star, no step) is unrestricted; e.g. "*/15" IS restricted.
        self.starred = spec.strip() == "*"
        for part in spec.split(","):
            part = part.strip()
            step = 1
            if "/" in part:
                part, step_s = part.split("/", 1)
                step = int(step_s)
            if part == "*":
                rng = range(lo, hi + 1, step)
            elif "-" in part:
                a, b = part.split("-", 1)
                rng = range(int(a), int(b) + 1, step)
            else:
                # Bare single value: exactly that value (Vixie).  "N/step"
                # keeps the standard N..hi/step semantics.
                if step == 1:
                    rng = range(int(part), int(part) + 1)
                else:
                    rng = range(int(part), hi + 1, step)
            self.values.update(rng)

    def matches(self, v: int) -> bool:
        return v in self.values


class CronTrigger:
    FIELDS = [
        ("minute", 0, 59), ("hour", 0, 23), ("dom", 1, 31),
        ("month", 1, 12), ("dow", 0, 7),
    ]

    def __init__(self, expr: str):
        parts = expr.split()
        if len(parts) != 5:
            raise ValueError(f"cron expr must have 5 fields: {expr!r}")
        self.fields = [
            _CronField(p, lo, hi) for p, (name, lo, hi) in zip(parts, self.FIELDS)
        ]

    def matches(self, dt: datetime | None = None) -> bool:
        dt = dt or datetime.now(timezone.utc)
        minute, hour, dom, month, dow = (
            dt.minute, dt.hour, dt.day, dt.month, dt.weekday()
        )
        # dow: cron uses 0/7=Sunday; python weekday() is 0=Monday.
        cron_dow = (dow + 1) % 7
        dom_ok = self.fields[2].matches(dom)
        dow_ok = self.fields[4].matches(cron_dow) or (
            cron_dow == 0 and self.fields[4].matches(7)
        )
        # Vixie day rule: if BOTH dom and dow are restricted (not "*"),
        # the day matches when EITHER matches; otherwise both must match.
        if self.fields[2].starred and self.fields[4].starred:
            day_ok = dom_ok and dow_ok
        elif self.fields[2].starred:
            day_ok = dow_ok
        elif self.fields[4].starred:
            day_ok = dom_ok
        else:
            day_ok = dom_ok or dow_ok
        return (
            self.fields[0].matches(minute)
            and self.fields[1].matches(hour)
            and self.fields[3].matches(month)
            and day_ok
        )

    def next_fire(self, after: datetime | None = None) -> datetime:
        """Brute-force minute scan (<= 366*24*60 iters) - good enough locally."""
        t = (after or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
        for _ in range(366 * 24 * 60):
            t = t.replace(minute=t.minute)
            if self.matches(t):
                return t
            # advance one minute
            t = (
                datetime.fromtimestamp(t.timestamp() + 60, tz=timezone.utc)
            )
        raise RuntimeError("cron expression never fires")


class IntervalTrigger:
    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self._next: float = time.time() + self.seconds

    def due(self, now: float | None = None) -> bool:
        now = now or time.time()
        return now >= self._next

    def fire(self) -> None:
        self._next = time.time() + self.seconds


class EventTrigger:
    """Polls a directory of drop files (one JSON file per event)."""

    def __init__(self, dir: str):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)

    def poll(self) -> list[dict[str, Any]]:
        events = []
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.dir, fn)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    events.append(json.load(fh))
                os.remove(path)
            except (json.JSONDecodeError, OSError):
                continue
        return events


class Trigger:
    """Wraps one of the concrete trigger types."""

    def __init__(self, spec: dict[str, Any]):
        kind = spec.get("type", "interval")
        if kind == "cron":
            self.kind = "cron"
            self.cron = CronTrigger(spec["expr"])
            self._last_min: float = 0.0
        elif kind == "interval":
            self.kind = "interval"
            self.interval = IntervalTrigger(spec.get("seconds", 3600))
        elif kind == "event":
            self.kind = "event"
            self.event = EventTrigger(spec.get("dir", ".ff/events"))
        else:
            raise ValueError(f"unknown trigger type {kind!r}")

    def check(self) -> list[dict[str, Any]]:
        """Return a list of due events (0 or more)."""
        if self.kind == "cron":
            now = time.time()
            minute_epoch = int(now // 60)
            if minute_epoch != int(self._last_min // 60):
                self._last_min = now
                if self.cron.matches():
                    return [{"type": "cron", "ts": now}]
            return []
        if self.kind == "interval":
            if self.interval.due():
                self.interval.fire()
                return [{"type": "interval", "ts": time.time()}]
            return []
        return self.event.poll()


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

_VAR_RE = re.compile(r"\{\{\s*([a-z_.\d]+)\s*\}\}")


def render(template: str, context: dict[str, Any]) -> str:
    def sub(m: re.Match) -> str:
        parts = m.group(1).split(".")
        cur: Any = context
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return m.group(0)
        return str(cur)

    return _VAR_RE.sub(sub, template)


def render_args(args: Any, context: dict[str, Any]) -> Any:
    if isinstance(args, str):
        return render(args, context)
    if isinstance(args, dict):
        return {k: render_args(v, context) for k, v in args.items()}
    if isinstance(args, list):
        return [render_args(v, context) for v in args]
    return args


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    workflow TEXT,
    status TEXT,
    started_at REAL,
    ended_at REAL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS dead_letter (
    id TEXT PRIMARY KEY,
    workflow TEXT,
    step TEXT,
    attempts INTEGER,
    error TEXT,
    ts REAL
);
CREATE TABLE IF NOT EXISTS workflow_state (
    name TEXT PRIMARY KEY,
    enabled INTEGER,
    last_run REAL,
    last_status TEXT
);
"""


class WorkflowEngine:
    """Holds workflow definitions, executes runs, tracks state in SQLite."""

    def __init__(
        self,
        mcp_registry,
        dir: str = "workflows",
        state_db: str = ".ff/workflows.db",
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
        audit=None,
    ):
        self.mcp = mcp_registry
        self.dir = dir
        self.max_attempts = max_attempts
        self.backoff = backoff_seconds
        self.audit = audit
        os.makedirs(os.path.dirname(state_db) or ".", exist_ok=True)
        self.db = sqlite3.connect(state_db, check_same_thread=False)
        self._db_lock = threading.Lock()
        self.db.executescript(STATE_SCHEMA)
        self.db.commit()
        self.workflows: dict[str, dict[str, Any]] = {}
        self.triggers: dict[str, Trigger] = {}
        self.running: set[str] = set()
        self.history: list[dict[str, Any]] = []

    # -- loading ---------------------------------------------------------------
    def load_dir(self) -> int:
        if not os.path.isdir(self.dir):
            return 0
        n = 0
        for fn in sorted(os.listdir(self.dir)):
            if fn.endswith((".yaml", ".yml")):
                with open(os.path.join(self.dir, fn), "r", encoding="utf-8") as fh:
                    doc = miniyaml_load(fh.read())
                if isinstance(doc, dict) and doc.get("name"):
                    self.load_workflow(doc)
                    n += 1
        return n

    def load_workflow(self, doc: dict[str, Any]) -> None:
        name = doc["name"]
        self.workflows[name] = doc
        self.triggers[name] = Trigger(doc.get("trigger", {"type": "interval", "seconds": 86400}))
        with self._db_lock:
            cur = self.db.execute(
                "INSERT INTO workflow_state(name, enabled, last_run, last_status) "
                "VALUES(?,?,NULL,NULL) ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled",
                (name, 1 if doc.get("enabled", True) else 0),
            )
            self.db.commit()
            del cur

    # -- execution ---------------------------------------------------------------
    def _execute_step(self, wf: dict, step: dict, context: dict[str, Any]) -> Any:
        tool = step["tool"]
        server, _, tool_name = tool.partition(".")
        args = render_args(step.get("input", {}), context)
        last_err: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.mcp.call(server, tool_name, args)
                return result
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.max_attempts:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(
            f"step {step.get('name')!r} failed after {self.max_attempts} attempts: {last_err}"
        )

    def run_workflow(self, name: str, event: dict[str, Any] | None = None) -> dict[str, Any]:
        wf = self.workflows.get(name)
        if wf is None:
            raise KeyError(f"unknown workflow {name!r}")
        run_id = str(uuid.uuid4())
        started = time.time()
        context: dict[str, Any] = {"event": event or {}, "steps": {}}
        self.running.add(name)
        status, detail = "succeeded", ""
        try:
            for step in wf.get("steps", []):
                result = self._execute_step(wf, step, context)
                context["steps"][step.get("name", step["tool"])] = {"result": result}
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            detail = str(exc)
            self._dead_letter(name, step.get("name", "?"), self.max_attempts, str(exc))
        finally:
            self.running.discard(name)
        ended = time.time()
        record = {
            "id": run_id, "workflow": name, "status": status,
            "started_at": started, "ended_at": ended, "detail": detail,
        }
        with self._db_lock:
            self.db.execute(
                "INSERT INTO runs(id, workflow, status, started_at, ended_at, detail) "
                "VALUES(?,?,?,?,?,?)",
                (run_id, name, status, started, ended, detail),
            )
            self.db.execute(
                "UPDATE workflow_state SET last_run=?, last_status=? WHERE name=?",
                (ended, status, name),
            )
            self.db.commit()
        self.history.append(record)
        if self.audit is not None:
            self.audit.record("system", "workflow:run", name, status, detail)
        return record

    def _dead_letter(self, workflow: str, step: str, attempts: int, error: str) -> None:
        with self._db_lock:
            self.db.execute(
                "INSERT INTO dead_letter(id, workflow, step, attempts, error, ts) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), workflow, step, attempts, error, time.time()),
            )
            self.db.commit()

    # -- control plane ---------------------------------------------------------------
    def pause(self, name: str) -> None:
        self._set_enabled(name, 0)

    def resume(self, name: str) -> None:
        self._set_enabled(name, 1)

    def _set_enabled(self, name: str, v: int) -> None:
        with self._db_lock:
            self.db.execute(
                "UPDATE workflow_state SET enabled=? WHERE name=?", (v, name)
            )
            self.db.commit()

    def delete(self, name: str) -> bool:
        self.workflows.pop(name, None)
        self.triggers.pop(name, None)
        with self._db_lock:
            self.db.execute("DELETE FROM workflow_state WHERE name=?", (name,))
            self.db.commit()

    def is_enabled(self, name: str) -> bool:
        with self._db_lock:
            row = self.db.execute(
                "SELECT enabled FROM workflow_state WHERE name=?", (name,)
            ).fetchone()
        return bool(row and row[0]) if row is not None else False

    def logs(self, name: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        with self._db_lock:
            if name:
                rows = self.db.execute(
                    "SELECT id, workflow, status, started_at, ended_at, detail FROM runs "
                    "WHERE workflow=? ORDER BY started_at DESC LIMIT ?", (name, limit),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT id, workflow, status, started_at, ended_at, detail FROM runs "
                    "ORDER BY started_at DESC LIMIT ?", (limit,),
                ).fetchall()
        return [
            {"id": r[0], "workflow": r[1], "status": r[2], "started_at": r[3],
             "ended_at": r[4], "detail": r[5]}
            for r in rows
        ]

    def dead_letters(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = self.db.execute(
                "SELECT workflow, step, attempts, error, ts FROM dead_letter "
                "ORDER BY ts DESC LIMIT ?", (limit,),
            ).fetchall()
        return [
            {"workflow": r[0], "step": r[1], "attempts": r[2], "error": r[3], "ts": r[4]}
            for r in rows
        ]

    # -- scheduler loop -----------------------------------------------------------
    def tick(self) -> list[dict[str, Any]]:
        """One scheduler iteration: fire due triggers for enabled workflows."""
        fired: list[dict[str, Any]] = []
        for name in list(self.workflows):
            if name in self.running or not self.is_enabled(name):
                continue
            for event in self.triggers[name].check():
                rec = self.run_workflow(name, event)
                fired.append(rec)
        return fired

    def close(self) -> None:
        with self._db_lock:
            self.db.close()


def engine_from_config(cfg, mcp_registry, audit=None, base_dir: str = ".") -> WorkflowEngine:
    import os

    return WorkflowEngine(
        mcp_registry,
        dir=os.path.join(base_dir, cfg.get("orchestrator.workflows.dir", "workflows")),
        state_db=os.path.join(base_dir, cfg.get("orchestrator.workflows.state_db", ".ff/workflows.db")),
        max_attempts=int(cfg.get("orchestrator.workflows.retry.max_attempts", 3)),
        backoff_seconds=float(cfg.get("orchestrator.workflows.retry.backoff_seconds", 2.0)),
        audit=audit,
    )
