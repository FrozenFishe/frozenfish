"""Orchestrator: wires config, security, MCP, GPU, router, context, workflows.

A running :class:`Orchestrator` is the single object the CLI and dashboard
talk to.  ``ff orchestrator start`` constructs one and (optionally) serves
the dashboard; ``stop`` shuts down clients and the scheduler.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from .audit import AuditLog
from .config import Config
from .context import ContextRouter, context_from_config
from .gpu import GpuRegistry, backend_from_config
from .mcp import McpRegistry, build_registry
from .router import Router, router_from_config
from .security import EgressFilter, RBAC, egress_from_config, rbac_from_config
from .workflow import WorkflowEngine, engine_from_config


class Orchestrator:
    def __init__(self, cfg: Config, base_dir: str = "."):
        self.cfg = cfg
        self.base_dir = os.path.abspath(base_dir)
        cfg.validate()

        # security & audit
        self.rbac: RBAC = rbac_from_config(cfg)
        self.egress: EgressFilter = egress_from_config(cfg)
        self.audit: AuditLog = AuditLog(
            os.path.join(self.base_dir, cfg.get("orchestrator.security.audit.path", ".ff/audit.db")),
            enabled=bool(cfg.get("orchestrator.security.audit.enabled", True)),
        )

        # mcp + sandbox (egress allowlist enforced on outbound tool calls)
        self.mcp: McpRegistry = build_registry(cfg, base_dir=self.base_dir, egress=self.egress)

        # gpu + router
        self.gpus: GpuRegistry = backend_from_config(cfg)
        if not self.gpus.names():
            self.gpus.discover(self.base_dir)
        self.router: Router = router_from_config(cfg, self.gpus)

        # context
        self.context: ContextRouter = context_from_config(cfg)

        # workflows
        self.workflows: WorkflowEngine = engine_from_config(
            cfg, self.mcp, audit=self.audit, base_dir=self.base_dir
        )
        self.workflows.load_dir()

        self.started = False
        self._stop_evt = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self.started_at: float | None = None

    # -- lifecycle -------------------------------------------------------------
    def start(self, serve_dashboard: bool = False, tick_interval: float = 30.0) -> None:
        if self.started:
            return
        self.started = True
        self.started_at = time.time()
        self.audit.record("system", "orchestrator:start", "orchestrator", "ok")
        if self.cfg.get("orchestrator.context.cag.warm_on_boot", False):
            self._warm_cag()
        self._stop_evt.clear()
        self._tick_thread = threading.Thread(
            target=self._tick_loop, args=(tick_interval,), daemon=True
        )
        self._tick_thread.start()

    def _warm_cag(self) -> None:
        """Build CAG entries for any workflow preface found under .ff/prefaces."""
        d = os.path.join(self.base_dir, ".ff", "prefaces")
        if not os.path.isdir(d):
            return
        for fn in os.listdir(d):
            try:
                with open(os.path.join(d, fn), "r", encoding="utf-8") as fh:
                    self.context.cag.build(fh.read(), fingerprint=fn)
            except (OSError, ValueError):
                continue

    def _tick_loop(self, interval: float) -> None:
        while not self._stop_evt.wait(interval):
            try:
                self.workflows.tick()
            except Exception:  # noqa: BLE001
                pass

    def stop(self) -> None:
        if not self.started:
            return
        self._stop_evt.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=2.0)
        self.workflows.close()
        self.mcp.shutdown()
        self.audit.close()
        self.started = False
        self.audit_closed = True

    # -- request path -----------------------------------------------------------
    def handle_request(
        self,
        user: str,
        permission: str,
        model: str,
        session: str | None = None,
        preface: str | None = None,
        tools: list[str] | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Full local request path: auth -> route -> context -> placement."""
        if not self.rbac.can(user, permission):
            self.audit.record(user, permission, model, "denied")
            raise PermissionError(f"user {user!r} lacks {permission!r}")
        decision = self.router.route(model, session=session)
        bundle = self.context.resolve(
            query=query or model, preface=preface, tools=tools
        )
        self.audit.record(user, permission, model, "ok",
                          f"backend={decision.backend} ctx={bundle['source']}")
        return {
            "backend": decision.backend,
            "score": decision.score,
            "context": bundle,
            "decision": decision.to_dict(),
        }

    # -- status -------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "version": "0.9.4",
            "backends": [b.snapshot() for b in self.gpus.all()],
            "router": self.router.stats(),
            "mcp": {
                "servers": [e.info() for e in self.mcp._entries.values()],
                "sandbox": {
                    "enabled": self.mcp.sandbox.enabled,
                    "workdir": self.mcp.sandbox.workdir,
                },
            },
            "context": self.context.stats(),
            "workflows": {
                "loaded": sorted(self.workflows.workflows),
                "enabled": [n for n in self.workflows.workflows if self.workflows.is_enabled(n)],
            },
            "security": {
                "users": {u: self.rbac.role_of(u) for u in self.rbac.users},
                "egress_entries": len(self.egress.entries),
            },
        }

    def stats(self, metrics: list[str] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        metrics = set(metrics or [])
        if "gpu" in metrics or not metrics:
            out["gpu"] = [b.snapshot() for b in self.gpus.all()]
        if "context" in metrics or not metrics:
            out["context"] = self.context.stats()
        if "cost" in metrics or not metrics:
            out["cost"] = {
                "runs": len(self.workflows.history),
                "succeeded": sum(1 for r in self.workflows.history if r["status"] == "succeeded"),
                "failed": sum(1 for r in self.workflows.history if r["status"] == "failed"),
            }
        return out

    def as_json(self) -> str:
        return json.dumps(self.status(), indent=2)
