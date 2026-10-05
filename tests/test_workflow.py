"""Tests for frozenfish.workflow: triggers, templating, engine, control plane."""
import json
import time
from datetime import datetime, timezone

import pytest

from frozenfish.audit import AuditLog
from frozenfish.mcp import McpRegistry, VectorServer
from frozenfish.workflow import (
    CronTrigger,
    EventTrigger,
    IntervalTrigger,
    Trigger,
    WorkflowEngine,
    render,
)


# ---------------------------------------------------------------------------
# Cron
# ---------------------------------------------------------------------------

class TestCron:
    def test_bad_field_count(self):
        with pytest.raises(ValueError, match="cron expr must have 5 fields"):
            CronTrigger("a b c d")

    def test_matches_explicit_datetimes(self):
        # 2026-01-05 is a Monday: python weekday 0 -> cron dow 1
        mon = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
        tue = datetime(2026, 1, 6, 0, 0, tzinfo=timezone.utc)
        trig = CronTrigger("0 0 5 1 1")  # minute 0, hour 0, dom 5, month 1, dow Mon
        assert trig.matches(mon) is True
        assert trig.matches(tue) is False

    def test_wildcard_and_step(self):
        # */15 in minutes matches 0, 15, 30, 45
        trig = CronTrigger("*/15 * * * *")
        assert trig.matches(datetime(2026, 3, 2, 1, 30)) is True
        assert trig.matches(datetime(2026, 3, 2, 1, 16)) is False

    def test_range(self):
        trig = CronTrigger("0 9-17 * * *")
        assert trig.matches(datetime(2026, 5, 1, 9, 0)) is True
        assert trig.matches(datetime(2026, 5, 1, 18, 0)) is False

    def test_next_fire(self):
        after = datetime(2026, 1, 1, 12, 33, tzinfo=timezone.utc)
        nxt = CronTrigger("0 13 * * *").next_fire(after=after)
        assert nxt == datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)

    def test_never_fires(self):
        with pytest.raises(RuntimeError, match="cron expression never fires"):
            CronTrigger("0 0 31 2 *").next_fire()  # Feb 31 does not exist


# ---------------------------------------------------------------------------
# Interval / Event
# ---------------------------------------------------------------------------

class TestInterval:
    def test_not_due_immediately(self):
        t = IntervalTrigger(3600)
        assert t.due() is False

    def test_due_after_window_and_rearm(self):
        t = IntervalTrigger(0.02)
        time.sleep(0.03)
        assert t.due() is True
        t.fire()
        assert t.due() is False


class TestEvent:
    def test_poll_consumes_and_skips_bad_json(self, tmp_path):
        d = tmp_path / "events"
        trig = EventTrigger(str(d))
        (d / "a.json").write_text(json.dumps({"kind": "x"}))
        (d / "b.json").write_text("{not json")
        (d / "ignore.txt").write_text("nope")
        events = trig.poll()
        assert events == [{"kind": "x"}]
        assert (d / "a.json").exists() is False
        assert (d / "b.json").exists() is True  # bad JSON kept
        assert trig.poll() == []  # consumed, idempotent


# ---------------------------------------------------------------------------
# Trigger wrapper
# ---------------------------------------------------------------------------

class TestTrigger:
    def test_unknown_type(self):
        with pytest.raises(ValueError, match="unknown trigger type"):
            Trigger({"type": "warp"})

    def test_default_is_interval(self):
        t = Trigger({})
        assert t.kind == "interval"
        assert t.interval.seconds == 3600.0

    def test_cron_check_shape(self):
        t = Trigger({"type": "cron", "expr": "* * * * *"})
        events = t.check()  # first call always evaluates (minute-epoch dedup)
        assert isinstance(events, list)
        for e in events:
            assert e["type"] == "cron" and "ts" in e
        # same minute again: deduped, no re-fire
        assert t.check() == []

    def test_interval_fires_at_most_once_per_window(self):
        t = Trigger({"type": "interval", "seconds": 0.01})
        time.sleep(0.02)
        assert len(t.check()) == 1
        assert t.check() == []  # re-armed after fire()


# ---------------------------------------------------------------------------
# Templating
# ---------------------------------------------------------------------------

class TestRender:
    def test_resolves_and_leaves_unresolved(self):
        out = render(
            "id={{steps.a.result.id}} missing={{nope}}",
            {"steps": {"a": {"result": {"id": 42}}}},
        )
        assert out == "id=42 missing={{nope}}"

    def test_renders_nested_structures(self):
        from frozenfish.workflow import render_args

        args = render_args(
            {"x": {"y": "{{event.k}}"}, "z": ["{{event.k}}", "lit"]},
            {"event": {"k": 7}},
        )
        assert args == {"x": {"y": "7"}, "z": ["7", "lit"]}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _engine(tmp_path, audit=None, max_attempts=3, backoff=0.01):
    reg = McpRegistry()
    reg.register_builtin(VectorServer())
    return WorkflowEngine(
        reg,
        dir=str(tmp_path / "workflows"),
        state_db=str(tmp_path / ".ff" / "wf.db"),
        max_attempts=max_attempts,
        backoff_seconds=backoff,
        audit=audit,
    ), reg


def _write_wf(path, name, steps_yaml, trigger_yaml="trigger:\n  type: interval\n  seconds: 3600\n"):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{name}.yaml").write_text(
        f"name: {name}\n{trigger_yaml}steps:\n{steps_yaml}"
    )


class TestLoadDir:
    def test_skips_nameless_docs(self, tmp_path):
        eng, _ = _engine(tmp_path)
        d = tmp_path / "workflows"
        _write_wf(d, "good", "  - name: s1\n    tool: vector_db.add\n    input:\n      text: hi\n")
        (d / "bad.yaml").write_text("trigger:\n  type: interval\n  seconds: 1\n")  # no name
        assert eng.load_dir() == 1
        assert "good" in eng.workflows
        assert "bad" not in eng.workflows

    def test_missing_dir(self, tmp_path):
        eng, _ = _engine(tmp_path)
        assert eng.load_dir() == 0


class TestRunWorkflow:
    def test_success_with_templating(self, tmp_path):
        audit = AuditLog(str(tmp_path / ".ff" / "audit.db"))
        eng, reg = _engine(tmp_path, audit=audit)
        _write_wf(
            tmp_path / "workflows",
            "seed_ref",
            "  - name: seed\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      text: hello world\n"
            "  - name: ref\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      id: \"{{steps.seed.result.id}}\"\n"
            "      text: duplicate\n",
        )
        eng.load_dir()

        # spy on registry.call to inspect rendered arguments
        calls = []
        orig = reg.call

        def spy(name, tool, arguments, session=None):
            r = orig(name, tool, arguments, session=session)
            calls.append((name, tool, dict(arguments), r))
            return r

        reg.call = spy
        record = eng.run_workflow("seed_ref")
        assert record["status"] == "succeeded"
        assert record["detail"] == ""
        assert record["workflow"] == "seed_ref"
        assert set(record) == {"id", "workflow", "status", "started_at", "ended_at", "detail"}

        # second step must receive the id rendered from the first step's result
        assert calls[0][3]["id"] is not None
        assert calls[1][2]["id"] == calls[0][3]["id"]

        logs = eng.logs()
        assert len(logs) == 1
        assert set(logs[0]) == {"id", "workflow", "status", "started_at", "ended_at", "detail"}

        # audit got a succeeded run
        rows = audit.tail(action="workflow:run")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "succeeded"
        assert rows[0]["actor"] == "system"

    def test_failure_dead_letters_and_retries(self, tmp_path):
        audit = AuditLog(str(tmp_path / ".ff" / "audit.db"))
        eng, reg = _engine(tmp_path, audit=audit, max_attempts=3, backoff=0.01)
        _write_wf(
            tmp_path / "workflows",
            "broken",
            "  - name: boom\n"
            "    tool: vector_db.nope\n"
            "    input:\n"
            "      text: x\n",
        )
        eng.load_dir()

        attempts = []
        orig = reg.call

        def spy(name, tool, arguments, session=None):
            attempts.append(1)
            return orig(name, tool, arguments, session=session)

        reg.call = spy
        record = eng.run_workflow("broken")
        assert record["status"] == "failed"
        assert "failed after 3 attempts" in record["detail"]
        assert len(attempts) == 3  # retried max_attempts times

        dl = eng.dead_letters()
        assert len(dl) == 1
        assert set(dl[0]) == {"workflow", "step", "attempts", "error", "ts"}
        assert dl[0]["workflow"] == "broken"
        assert dl[0]["step"] == "boom"
        assert dl[0]["attempts"] == 3

        rows = audit.tail(action="workflow:run")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "failed"

        # not left "running" after failure
        assert "broken" not in eng.running

    def test_unknown_workflow(self, tmp_path):
        eng, _ = _engine(tmp_path)
        with pytest.raises(KeyError, match="unknown workflow"):
            eng.run_workflow("nope")

    def test_event_is_passed_through(self, tmp_path):
        eng, reg = _engine(tmp_path)
        _write_wf(
            tmp_path / "workflows",
            "echo_evt",
            "  - name: s\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      text: \"{{event.msg}}\"\n",
        )
        eng.load_dir()
        orig = reg.call
        seen = {}

        def spy(name, tool, arguments, session=None):
            seen.update(arguments)
            return orig(name, tool, arguments, session=session)

        reg.call = spy
        eng.run_workflow("echo_evt", event={"msg": "from event"})
        assert seen["text"] == "from event"


class TestControlPlane:
    def _loaded(self, tmp_path):
        eng, _ = _engine(tmp_path)
        _write_wf(
            tmp_path / "workflows",
            "wf1",
            "  - name: s\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      text: x\n",
        )
        eng.load_dir()
        return eng

    def test_pause_resume_delete(self, tmp_path):
        eng = self._loaded(tmp_path)
        assert eng.is_enabled("wf1") is True
        eng.pause("wf1")
        assert eng.is_enabled("wf1") is False
        eng.resume("wf1")
        assert eng.is_enabled("wf1") is True
        eng.delete("wf1")
        assert "wf1" not in eng.workflows
        assert eng.is_enabled("wf1") is False

    def test_tick_fires_due_interval_and_skips_paused(self, tmp_path):
        eng, _ = _engine(tmp_path)
        _write_wf(
            tmp_path / "workflows",
            "fast",
            "  - name: s\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      text: x\n",
            trigger_yaml="trigger:\n  type: interval\n  seconds: 0.01\n",
        )
        _write_wf(
            tmp_path / "workflows",
            "slow",
            "  - name: s\n"
            "    tool: vector_db.add\n"
            "    input:\n"
            "      text: y\n",
        )
        eng.load_dir()
        time.sleep(0.02)  # make "fast" due
        eng.pause("slow")
        records = eng.tick()
        names = {r["workflow"] for r in records}
        assert names == {"fast"}
        # paused workflow is skipped, due one consumed (re-armed)
        assert eng.tick() == []
