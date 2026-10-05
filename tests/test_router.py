"""Tests for the GPU-aware scoring router."""

from __future__ import annotations

import pytest

from frozenfish.gpu import Backend, GpuRegistry
from frozenfish.router import MAX_QUEUE, Router, STICKY_BONUS, router_from_config


def _reg():
    reg = GpuRegistry()
    a = reg.add(Backend("gpu-a", accelerator="cuda"))
    a.utilization = 0.0
    b = reg.add(Backend("gpu-b", accelerator="cuda"))
    b.utilization = 0.9
    return reg


class TestScoring:
    def test_constants(self):
        assert STICKY_BONUS == 0.25
        assert MAX_QUEUE == 64

    def test_score_formula(self):
        reg = _reg()
        r = Router(reg)
        # idle, cold, empty queue: 0.5*1.0 + 0.3*0 + 0.2*1.0 = 0.7
        assert r.score_backend("gpu-a", "model-x") == pytest.approx(0.7)
        # warm adds 0.3
        reg.get("gpu-a").mark_warm("model-x")
        assert r.score_backend("gpu-a", "model-x") == pytest.approx(1.0)
        # busy util 0.9: 0.5*0.1 + 0 + 0.2 = 0.25
        assert r.score_backend("gpu-b", "model-x") == pytest.approx(0.25)

    def test_queue_pressure(self):
        reg = _reg()
        r = Router(reg)
        b = reg.get("gpu-a")
        b.queue_depth = MAX_QUEUE
        # 0.5*1.0 + 0 + 0.2*0 = 0.5
        assert r.score_backend("gpu-a", "m") == pytest.approx(0.5)

    def test_unhealthy_scores_negative_one(self):
        reg = _reg()
        reg.get("gpu-b").healthy = False
        r = Router(reg)
        assert r.score_backend("gpu-b", "m") == -1.0

    def test_sticky_bonus(self):
        reg = _reg()
        r = Router(reg, sticky=True)
        r.route("m", session="s1")  # binds s1 -> gpu-a (idle wins)
        assert r.session_backend("s1") == "gpu-a"
        base = r.score_backend("gpu-a", "m", session="s1")
        assert base == pytest.approx(0.7 + STICKY_BONUS)

    def test_rank_orders_by_score(self):
        reg = _reg()
        r = Router(reg)
        ranked = r.rank("m")
        assert [n for n, _ in ranked] == ["gpu-a", "gpu-b"]


class TestRoute:
    def test_route_returns_decision(self):
        reg = _reg()
        r = Router(reg)
        d = r.route("m")
        assert d.backend == "gpu-a"
        assert d.policy == "gpu_aware"
        assert d.sticky is False
        assert set(d.breakdown) == {"util", "warm", "queue", "sticky"}
        dd = d.to_dict()
        assert dd["backend"] == "gpu-a"
        assert isinstance(dd["score"], float)

    def test_route_picks_warm(self):
        reg = _reg()
        reg.get("gpu-b").mark_warm("hot-model")
        r = Router(reg)
        # gpu-a: 0.7  gpu-b: 0.5*0.1 + 0.3 + 0.2 = 0.55  -> gpu-a still wins
        assert r.route("hot-model").backend == "gpu-a"

    def test_empty_registry_raises(self):
        r = Router(GpuRegistry())
        with pytest.raises(RuntimeError, match="no backends registered"):
            r.route("m")

    def test_all_unhealthy_raises(self):
        reg = _reg()
        for n in reg.names():
            reg.get(n).healthy = False
        r = Router(reg)
        with pytest.raises(RuntimeError, match="no healthy backends available"):
            r.route("m")

    def test_session_binding(self):
        reg = _reg()
        r = Router(reg)
        r.route("m", session="conv-1")
        assert r.session_backend("conv-1") == "gpu-a"
        assert r.stats()["active_sessions"] == 1

    def test_acquire_release(self):
        reg = _reg()
        r = Router(reg)
        b = reg.get("gpu-a")
        r.acquire(b.name)
        assert b.in_flight == 1 and b.queue_depth == 1
        r.release(b.name)
        assert b.in_flight == 0 and b.queue_depth == 0
        r.release(b.name)  # clamps at zero, no underflow
        assert b.queue_depth == 0 and b.in_flight == 0


class TestPolicies:
    def test_unknown_policy_rejected(self):
        r = Router(_reg())
        with pytest.raises(ValueError, match="unknown policy"):
            r.set_policy("greedy")
        assert r.get_policy() == "gpu_aware"

    def test_round_robin_cycles(self):
        reg = GpuRegistry()
        reg.add(Backend("b-1"))
        reg.add(Backend("b-2"))
        r = Router(reg, policy="round_robin")
        picks = [r.route("m").backend for _ in range(4)]
        assert picks == ["b-1", "b-2", "b-1", "b-2"]

    def test_least_queue(self):
        reg = GpuRegistry()
        x = reg.add(Backend("zeta"))
        y = reg.add(Backend("alpha"))
        y.queue_depth = 5
        r = Router(reg, policy="least_queue")
        assert r.route("m").backend == "zeta"
        # tie broken by name
        x.queue_depth = 5
        assert r.route("m").backend == "alpha"


class TestFromConfig:
    def test_router_from_config_maps_backends(self):
        from frozenfish.config import Config

        cfg = Config.load(None)
        reg = _reg()
        r = router_from_config(cfg, reg)
        assert r.policy == "gpu_aware"  # "backends" strategy maps to gpu_aware
        assert r.sticky is True
        assert r.weights == {"util": 0.5, "warm": 0.3, "queue": 0.2}
