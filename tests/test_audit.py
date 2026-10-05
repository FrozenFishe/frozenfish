"""Tests for frozenfish.audit."""

from __future__ import annotations

import json

import pytest

from frozenfish.audit import AuditLog, make_event, summarize


def _log(tmp_path, enabled=True):
    return AuditLog(str(tmp_path / "audit.db"), enabled=enabled)


class TestRecord:
    def test_disabled_record_returns_none(self, tmp_path):
        log = _log(tmp_path, enabled=False)
        assert log.record("alice", "mcp:call", "sqlite", "ok") is None
        assert log.tail() == []

    def test_enabled_record_returns_event_with_id(self, tmp_path):
        log = _log(tmp_path)
        ev = log.record("alice", "mcp:call", "sqlite", "ok", {"k": 1})
        assert ev is not None
        assert ev["id"] >= 1
        assert ev["ts"] > 0
        assert ev["detail"] == {"k": 1}
        log.close()


class TestTail:
    @pytest.fixture()
    def log(self, tmp_path):
        log = _log(tmp_path)
        log.record("alice", "mcp:call", "sqlite", "ok")
        log.record("bob", "workflow:run", "smoke", "failed")
        log.record("alice", "mcp:call", "filesystem", "ok")
        return log

    def test_rows_have_iso_and_detail(self, log):
        rows = log.tail()
        assert len(rows) == 3
        row = rows[0]
        for key in ("id", "ts", "iso", "actor", "action", "target", "outcome", "detail"):
            assert key in row
        assert "T" in row["iso"]  # ISO-8601 with time separator
        assert isinstance(row["detail"], dict)

    def test_filter_actor(self, log):
        rows = log.tail(actor="alice")
        assert {r["actor"] for r in rows} == {"alice"}
        assert len(rows) == 2

    def test_filter_action(self, log):
        rows = log.tail(action="workflow:run")
        assert len(rows) == 1
        assert rows[0]["actor"] == "bob"

    def test_since_window(self, log):
        # everything recorded within the last hour
        assert len(log.tail(since="1h")) == 3
        # a since-bound far in the future contains nothing
        assert log.tail(since="2999-01-01") == []

    def test_limit(self, log):
        assert len(log.tail(limit=2)) == 2

    def test_order_newest_first(self, log):
        rows = log.tail()
        assert rows[0]["id"] > rows[-1]["id"]


class TestExport:
    def test_export_json(self, tmp_path):
        log = _log(tmp_path)
        log.record("alice", "mcp:call", "sqlite", "ok")
        data = json.loads(log.export("json"))
        assert isinstance(data, list) and len(data) == 1
        log.close()

    def test_export_bad_format(self, tmp_path):
        log = _log(tmp_path)
        with pytest.raises(ValueError, match="unsupported export format"):
            log.export("csv")
        log.close()


class TestSummarize:
    def test_summarize_keys_and_counts(self, tmp_path):
        log = _log(tmp_path)
        log.record("alice", "a", "t", "ok")
        log.record("alice", "b", "t", "denied")
        log.record("bob", "c", "t", "ok")
        s = summarize(log.tail())
        assert s["count"] == 3
        assert s["by_actor"] == {"alice": 2, "bob": 1}
        assert s["by_outcome"] == {"ok": 2, "denied": 1}
        log.close()

    def test_summarize_empty(self):
        assert summarize([]) == {"count": 0, "by_actor": {}, "by_outcome": {}}


class TestMakeEvent:
    def test_redacts_actor_and_target(self):
        ev = make_event("api_key: abcdef12345678", "act", "user bob@example.com")
        assert "abcdef12345678" not in ev["actor"]
        assert "bob@example.com" not in ev["target"]
        assert "[REDACTED:EMAIL]" in ev["target"]

    def test_clean_values_unchanged(self):
        ev = make_event("alice", "mcp:call", "sqlite", "ok", {"x": 1})
        assert ev["actor"] == "alice"
        assert ev["target"] == "sqlite"
        assert ev["outcome"] == "ok"
        assert ev["detail"] == {"x": 1}
