"""Scoring router for inference placement.

The router scores every healthy backend for a (model, session) request:

    score = w_util * (1 - util) + w_warm * warm_bonus + w_queue * (1 - queue_norm)

with weights from config (default ``util=0.5, warm=0.3, queue=0.2``).
``warm_bonus`` is 1.0 when the model is warm on that backend, else 0.0.
``queue_norm`` clamps ``queue_depth / max_queue`` to [0, 1].

Sticky sessions: when ``sticky`` is enabled and a session was previously
routed to a backend, that backend receives a +0.25 score bonus (capped at
1.5) so short conversations stay on one device without thrashing KV state.

Policies
--------
``gpu_aware``   - full scoring above (default).
``round_robin`` - next backend in stable name order.
``least_queue`` - minimal queue depth, ties broken by name.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .gpu import GpuRegistry

STICKY_BONUS = 0.25
MAX_QUEUE = 64  # queue depth at which queue pressure = 1.0


class RoutingDecision:
    def __init__(self, backend: str, score: float, breakdown: dict[str, float],
                 sticky: bool, policy: str, ts: float):
        self.backend = backend
        self.score = score
        self.breakdown = breakdown
        self.sticky = sticky
        self.policy = policy
        self.ts = ts

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "score": round(self.score, 4),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "sticky": self.sticky,
            "policy": self.policy,
            "ts": self.ts,
        }


class Router:
    def __init__(self, registry: GpuRegistry,
                 weights: dict[str, float] | None = None,
                 sticky: bool = True,
                 policy: str = "gpu_aware"):
        self.registry = registry
        self.weights = dict(weights or {"util": 0.5, "warm": 0.3, "queue": 0.2})
        self.sticky = sticky
        self.policy = policy
        self._sessions: dict[str, str] = {}      # session -> backend
        self._rr_index = 0
        self._lock = threading.Lock()

    # -- configuration ---------------------------------------------------------
    def set_policy(self, policy: str) -> None:
        if policy not in ("gpu_aware", "round_robin", "least_queue"):
            raise ValueError(f"unknown policy {policy!r}")
        self.policy = policy

    def get_policy(self) -> str:
        return self.policy

    # -- core ---------------------------------------------------------------------
    def score_backend(self, backend_name: str, model: str, session: str | None = None) -> float:
        b = self.registry.get(backend_name)
        if not b.healthy:
            return -1.0
        w = self.weights
        util_term = (1.0 - max(0.0, min(1.0, b.utilization))) * w["util"]
        warm_term = (1.0 if b.is_warm(model) else 0.0) * w["warm"]
        q = max(0, b.queue_depth) / MAX_QUEUE
        queue_term = (1.0 - min(1.0, q)) * w["queue"]
        total = util_term + warm_term + queue_term
        if self.sticky and session and self._sessions.get(session) == backend_name:
            total += STICKY_BONUS
        return total

    def rank(self, model: str, session: str | None = None) -> list[tuple[str, float]]:
        return sorted(
            ((n, self.score_backend(n, model, session)) for n in self.registry.names()),
            key=lambda t: (-t[1], t[0]),
        )

    def route(self, model: str, session: str | None = None) -> RoutingDecision:
        with self._lock:
            names = self.registry.names()
            if not names:
                raise RuntimeError("no backends registered")
            if self.policy == "round_robin":
                healthy = [n for n in names if self.registry.get(n).healthy] or names
                name = healthy[self._rr_index % len(healthy)]
                self._rr_index += 1
                self._sessions[session or name] = name
                return RoutingDecision(name, self.score_backend(name, model, session),
                                       {}, False, self.policy, time.time())
            if self.policy == "least_queue":
                name = min(names, key=lambda n: (self.registry.get(n).queue_depth, n))
                self._sessions[session or name] = name
                return RoutingDecision(name, self.score_backend(name, model, session),
                                       {}, False, self.policy, time.time())
            scored = self.rank(model, session)
            name, score = scored[0]
            if score < 0:
                raise RuntimeError("no healthy backends available")
            sticky = bool(session) and self._sessions.get(session) == name
            b = self.registry.get(name)
            breakdown = {
                "util": (1.0 - b.utilization) * self.weights["util"],
                "warm": 1.0 if b.is_warm(model) else 0.0,
                "queue": (1.0 - min(1.0, max(0, b.queue_depth) / MAX_QUEUE)) * self.weights["queue"],
                "sticky": STICKY_BONUS if sticky else 0.0,
            }
            if session:
                self._sessions[session] = name
            return RoutingDecision(name, score, breakdown, sticky, self.policy, time.time())

    def release(self, backend: str) -> None:
        b = self.registry.get(backend)
        b.in_flight = max(0, b.in_flight - 1)
        b.queue_depth = max(0, b.queue_depth - 1)

    def acquire(self, backend: str) -> None:
        b = self.registry.get(backend)
        b.in_flight += 1
        b.queue_depth += 1

    def session_backend(self, session: str) -> str | None:
        return self._sessions.get(session)

    def stats(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "weights": self.weights,
            "sticky": self.sticky,
            "active_sessions": len(self._sessions),
        }


def router_from_config(cfg, registry: GpuRegistry) -> Router:
    return Router(
        registry,
        weights=cfg.get("orchestrator.routing.weights"),
        sticky=bool(cfg.get("orchestrator.routing.sticky", True)),
        policy=str(cfg.get("orchestrator.routing.strategy", "backends")).replace("backends", "gpu_aware"),
    )
