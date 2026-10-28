"""Tests for frozenfish.mcp: Sandbox, servers, registry, external entries."""

from __future__ import annotations

import json

import pytest

from frozenfish.mcp import (
    ADVERTISED_NAMES,
    EmailServer,
    FilesystemServer,
    HttpClientServer,
    MarkdownServer,
    McpEntry,
    McpRegistry,
    Sandbox,
    SandboxError,
    ShellServer,
    SqliteServer,
    VectorServer,
    build_registry,
)
from frozenfish.config import Config
from frozenfish.security import EgressFilter

# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_run_basic(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        out = sb.run("echo hello")
        assert out["exit"] == 0
        assert "hello" in out["stdout"]

    def test_run_missing_command(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        out = sb.run("definitely-not-a-real-cmd-xyz")
        assert out["exit"] == 127
        assert "command not found" in out["stderr"]

    def test_run_timeout(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        out = sb.run("sleep 5", timeout=0.3)
        assert out["exit"] == 124
        assert out.get("timeout") is True

    def test_resolve_denied_path(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        with pytest.raises(SandboxError, match="denied by sandbox denylist"):
            sb.resolve("/etc/passwd")

    def test_resolve_escape(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        with pytest.raises(SandboxError, match="escapes sandbox workdir"):
            sb.resolve("../outside")

    def test_resolve_joins_relative(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        p = sb.resolve("sub/file.txt")
        assert p.startswith(str(tmp_path))

    def test_disabled_no_checks(self, tmp_path):
        sb = Sandbox(str(tmp_path), enabled=False)
        assert sb.resolve("/etc/passwd") == "/etc/passwd"


# ---------------------------------------------------------------------------
# Built-in servers (called directly with a minimal ctx)
# ---------------------------------------------------------------------------

CTX = {"session": "t1", "sandbox": None}


class TestFilesystemServer:
    def test_write_read_list(self, tmp_path):
        fs = FilesystemServer(str(tmp_path))
        r = fs.call("write_file", {"path": "a.txt", "content": "hi"}, {"sandbox": Sandbox(str(tmp_path))})
        assert r["bytes"] == 2
        r = fs.call("read_file", {"path": "a.txt"}, {"sandbox": Sandbox(str(tmp_path))})
        assert r["content"] == "hi"
        r = fs.call("list_dir", {"path": "."}, {"sandbox": Sandbox(str(tmp_path))})
        assert "a.txt" in r["entries"]

    def test_search(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        fs = FilesystemServer(str(tmp_path))
        fs.call("write_file", {"path": "notes/README.md", "content": "needle-haystack"}, {"sandbox": sb})
        fs.call("write_file", {"path": "notes/other.txt", "content": "nothing here"}, {"sandbox": sb})
        r = fs.call("search", {"path": ".", "pattern": "needle"}, {"sandbox": sb})
        assert len(r["matches"]) == 1
        assert r["matches"][0].endswith("README.md")

    def test_unknown_tool(self, tmp_path):
        fs = FilesystemServer(str(tmp_path))
        with pytest.raises(ValueError, match="unknown tool"):
            fs.call("nope", {}, CTX)

    def test_tools_shape(self, tmp_path):
        names = {t["name"] for t in FilesystemServer(str(tmp_path)).tools()}
        assert names == {"read_file", "write_file", "list_dir", "search"}


class TestSqliteServer:
    def test_query_and_update(self, tmp_path):
        db = str(tmp_path / "t.db")
        srv = SqliteServer(db)
        r = srv.call("query", {"db": db, "sql": "CREATE TABLE t(a)"} , CTX)
        assert "rowcount" in r
        srv.call("query", {"db": db, "sql": "INSERT INTO t VALUES (1), (2)"}, CTX)
        r = srv.call("query", {"db": db, "sql": "SELECT a FROM t ORDER BY a"}, CTX)
        assert r["columns"] == ["a"]
        assert r["rows"] == [[1], [2]]

    def test_unknown_tool(self, tmp_path):
        with pytest.raises(ValueError, match="unknown tool"):
            SqliteServer(str(tmp_path / "x.db")).call("nope", {}, CTX)


class TestShellServer:
    def test_run_via_registry_ctx(self, tmp_path):
        sb = Sandbox(str(tmp_path))
        r = ShellServer().call("run", {"cmd": "echo ok"}, {"sandbox": sb})
        assert r["exit"] == 0 and "ok" in r["stdout"]

    def test_requires_sandbox(self):
        with pytest.raises(SandboxError):
            ShellServer().call("run", {"cmd": "echo"}, {"sandbox": None})


class TestVectorServer:
    def test_add_search_count(self):
        v = VectorServer()
        assert v.call("count", {}, CTX)["count"] == 0
        v.call("add", {"id": "a", "text": "arctic ice fish"}, CTX)
        v.call("add", {"text": "completely unrelated banana"}, CTX)
        assert v.call("count", {}, CTX)["count"] == 2
        r = v.call("search", {"query": "arctic ice fish", "k": 1}, CTX)
        assert r["results"][0]["id"] == "a"
        assert r["results"][0]["score"] > 0.5


class TestMarkdownServer:
    def test_outline_and_stats(self):
        text = "# Title\n\npara\n\n## Sub\n"
        r = MarkdownServer().call("outline", {"text": text}, CTX)
        assert [h["level"] for h in r["outline"]] == [1, 2]
        r = MarkdownServer().call("stats", {"text": text}, CTX)
        assert r["chars"] == len(text) and r["lines"] == 4


class TestEmailServer:
    def test_send_and_inbox(self, tmp_path):
        srv = EmailServer(str(tmp_path / "mail"))
        r = srv.call("send", {"to": "a@b.c", "subject": "s", "body": "b"}, CTX)
        assert r["id"]
        r = srv.call("inbox", {}, CTX)
        assert len(r["messages"]) == 1


class TestRegistry:
    def test_builtin_round_trip(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        reg.register_builtin(FilesystemServer(str(tmp_path)))
        assert reg.names() == ["filesystem"]
        entry = reg.get("filesystem")
        assert isinstance(entry, McpEntry)
        r = reg.call("filesystem", "write_file", {"path": "x.txt", "content": "y"})
        assert r["bytes"] == 1

    def test_get_unknown(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        with pytest.raises(KeyError, match="unknown mcp server"):
            reg.get("nope")

    def test_inspect_and_info(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        reg.register_builtin(MarkdownServer())
        info = reg.inspect("markdown")
        assert info["name"] == "markdown"
        assert info["kind"] == "builtin"
        assert any(t["name"] == "outline" for t in info["tools"])

    def test_unregister(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        reg.register_builtin(MarkdownServer())
        assert reg.unregister("markdown") is True
        assert reg.unregister("markdown") is False

    def test_call_returns_dict(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        reg.register_builtin(VectorServer())
        r = reg.call("vector_db", "count", {})
        assert isinstance(r, dict) and r["count"] == 0


class TestEgressWiring:
    def test_registry_denies_unlisted_host(self):
        reg = McpRegistry()
        reg.egress = EgressFilter(["allowed.example.com"])
        reg.register_builtin(HttpClientServer())
        with pytest.raises(PermissionError, match="egress denied"):
            reg.call("http_client", "get", {"url": "http://blocked.example.com/x"})

    def test_empty_allowlist_denies_all(self):
        reg = McpRegistry(egress=EgressFilter([]))
        reg.register_builtin(HttpClientServer())
        with pytest.raises(PermissionError, match="egress denied"):
            reg.call("http_client", "get", {"url": "http://anything.example.com/"})

    def test_build_registry_wires_egress_from_config(self, tmp_path):
        cfg = Config.load(None)
        reg = build_registry(cfg, base_dir=str(tmp_path))
        assert reg.egress is not None
        assert reg.egress.entries == []  # local-first default: deny all
        reg.register_builtin(HttpClientServer())
        with pytest.raises(PermissionError, match="egress denied"):
            reg.call("http_client", "get", {"url": "http://anything.example.com/"})

    def test_no_config_means_no_egress_filter(self, tmp_path):
        reg = build_registry(base_dir=str(tmp_path))
        assert reg.egress is None


class TestExternalEntry:
    def test_external_info_and_stop(self, tmp_path):
        reg = McpRegistry(Sandbox(str(tmp_path)))
        reg.register_external("echo-srv", "python3 -c 'pass'")
        entry = reg.get("echo-srv")
        assert entry.is_external()
        assert entry.info()["kind"] == "external"
        assert entry.info()["cmd"] == "python3 -c 'pass'"
        entry.stop()  # must not raise
        assert reg.names() == ["echo-srv"]


class TestBuildRegistry:
    def test_builtin_count_and_advertised(self, tmp_path):
        reg = build_registry(base_dir=str(tmp_path))
        assert len(reg.names()) == 10
        for name in ADVERTISED_NAMES:
            if name not in ("postgres", "browser"):
                assert name in reg.names()

    def test_external_from_config(self, tmp_path):
        cfg = Config.load(None)
        cfg._data["orchestrator"]["mcp"]["servers"] = [
            {"name": "custom", "cmd": "echo hi"},
            {"name": "bad", "cmd": ""},  # skipped: no cmd
        ]
        reg = build_registry(cfg, base_dir=str(tmp_path))
        assert "custom" in reg.names()
        assert "bad" not in reg.names()
