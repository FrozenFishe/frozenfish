"""Tests for frozenfish.orchestrator: wiring, RBAC gate, lifecycle, stats."""
import copy
import json
import time

import pytest

from frozenfish.audit import AuditLog
from frozenfish.config import DEFAULT_CONFIG, Config
from frozenfish.orchestrator import Orchestrator


def make_cfg():
    data = copy.deepcopy(DEFAULT_CONFIG)
    data["orchestrator"]["backends"] = [
        {
            "name": "test-a",
            "kind": "openai",
            "endpoint": "http://127.0.0.1:9999",
            "accelerator": "cpu",
        }
    ]
    data["orchestrator"]["security"]["rbac"]["users"] = {
        "alice": "admin",
        "bob": "operator",
        "carol": "viewer",
    }
    # keep the boot path quiet + fast
    data["orchestrator"]["context"]["cag"]["warm_on_boot"] = False
    data["orchestrator"]["workflows"]["retry"]["backoff_seconds"] = 0.01
    cfg = Config(data)
    cfg.validate()
    return cfg


@pytest.fixture
def orch(tmp_path):
    return Orchestrator(make_cfg(), base_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Construction / status shape
# ---------------------------------------------------------------------------

class TestStatus:
    def test_basic_shape(self, orch):
        s = orch.status()
        assert s["started"] is False
        assert s["uptime_s"] == 0.0
        assert s["version"] == "0.9.4"

        # deterministic single backend from config (no auto-discover)
        assert len(s["backends"]) == 1
        assert s["backends"][0]["name"] == "test-a"
        assert s["backends"][0]["accelerator"] == "cpu"

        # default strategy "backends" maps to gpu_aware policy
        assert s["router"]["policy"] == "gpu_aware"
        assert s["router"]["sticky"] is True

        # 10 built-in MCP servers, sandbox on
        assert len(s["mcp"]["servers"]) == 10
        assert s["mcp"]["sandbox"]["enabled"] is True

        # rbac users + empty egress allowlist (local-first default)
        assert s["security"]["users"] == {
            "alice": "admin",
            "bob": "operator",
            "carol": "viewer",
        }
        assert s["security"]["egress_entries"] == 0

        # the egress filter is wired into the MCP registry (enforced on calls)
        assert orch.mcp.egress is orch.egress

        # no workflows on disk in a fresh dir
        assert s["workflows"]["loaded"] == []
        assert s["workflows"]["enabled"] == []

    def test_context_stats_present(self, orch):
        ctx = orch.status()["context"]
        assert "cag" in ctx and "rag" in ctx and "car" in ctx


# ---------------------------------------------------------------------------
# Request path (RBAC gate -> route -> context)
# ---------------------------------------------------------------------------

class TestHandleRequest:
    def test_admin_allowed(self, orch):
        res = orch.handle_request("alice", "anything:at:all", "test-model")
        assert res["backend"] == "test-a"
        assert res["score"] > 0.0
        assert res["context"]["source"] == "none"  # no corpus loaded
        assert res["decision"]["policy"] == "gpu_aware"
        assert res["decision"]["backend"] == "test-a"

    def test_operator_allowed_and_denied(self, orch):
        # operator has mcp:call
        res = orch.handle_request("bob", "mcp:call", "test-model")
        assert res["backend"] == "test-a"
        # operator lacks mcp:delete
        with pytest.raises(PermissionError, match="user 'bob' lacks 'mcp:delete'"):
            orch.handle_request("bob", "mcp:delete", "test-model")

    def test_viewer_read_only(self, orch):
        assert orch.handle_request(
            "carol", "orchestrator:read", "test-model"
        )["backend"] == "test-a"
        with pytest.raises(PermissionError, match="user 'carol' lacks 'mcp:call'"):
            orch.handle_request("carol", "mcp:call", "test-model")

    def test_unknown_user_denied(self, orch):
        with pytest.raises(PermissionError, match="user 'mallory' lacks 'orchestrator:read'"):
            orch.handle_request("mallory", "orchestrator:read", "test-model")

    def test_denies_are_audited(self, orch):
        with pytest.raises(PermissionError):
            orch.handle_request("carol", "mcp:call", "test-model")
        rows = orch.audit.tail(action="mcp:call", actor="carol")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "denied"

    def test_success_is_audited_with_backend(self, orch):
        orch.handle_request("alice", "orchestrator:read", "test-model")
        rows = orch.audit.tail(action="orchestrator:read", actor="alice")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "ok"
        assert "backend=test-a" in rows[0]["detail"]  # stored as a JSON string


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_start_stop(self, orch):
        orch.start(tick_interval=0.05)
        assert orch.started is True
        assert orch.started_at is not None
        s = orch.status()
        assert s["started"] is True
        assert s["uptime_s"] >= 0.0
        time.sleep(0.1)

        orch.stop()
        assert orch.started is False
        assert orch.audit_closed is True

        # audit log is closed by stop(); reopen on the same file to read
        rows = AuditLog(orch.audit.path).tail(action="orchestrator:start")
        assert len(rows) == 1
        assert rows[0]["actor"] == "system"
        assert rows[0]["outcome"] == "ok"

    def test_start_is_idempotent(self, orch):
        orch.start(tick_interval=30.0)
        t1 = orch._tick_thread
        orch.start(tick_interval=30.0)  # no second thread
        assert orch._tick_thread is t1
        orch.stop()

    def test_stop_without_start_is_noop(self, orch):
        orch.stop()  # must not raise
        assert not hasattr(orch, "audit_closed")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_all_groups(self, orch):
        s = orch.stats(None)
        assert set(s) == {"gpu", "context", "cost"}
        assert len(s["gpu"]) == 1
        assert s["cost"] == {"runs": 0, "succeeded": 0, "failed": 0}

    def test_filtered_groups(self, orch):
        assert set(orch.stats(["gpu"])) == {"gpu"}
        assert set(orch.stats(["cost"])) == {"cost"}
        assert orch.stats(["gpu", "context"])["gpu"][0]["name"] == "test-a"

    def test_as_json(self, orch):
        j = json.loads(orch.as_json())
        assert j["version"] == "0.9.4"
        assert [b["name"] for b in j["backends"]] == ["test-a"]
