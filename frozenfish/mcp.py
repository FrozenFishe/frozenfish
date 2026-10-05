"""MCP layer: registry, in-process servers, stdio JSON-RPC client, sandbox.

Frozen Fish speaks JSON-RPC 2.0 with the MCP method surface:
``initialize``, ``tools/list``, ``tools/call``.  Servers may run in-process
(the 12 built-ins) or as child processes over stdio (user-registered).
"""

from __future__ import annotations

import abc
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Server interface
# ---------------------------------------------------------------------------

Tool = dict  # {"name": str, "description": str, "input": {schema-ish}}


class McpServer(abc.ABC):
    """Base class for in-process MCP servers."""

    name: str = "abstract"
    description: str = ""

    @abc.abstractmethod
    def tools(self) -> list[Tool]:
        """Return the advertised tool list."""

    @abc.abstractmethod
    def call(self, tool: str, arguments: dict[str, Any], ctx: dict[str, Any]) -> Any:
        """Execute a tool.  ``ctx`` carries sandbox + session metadata."""

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "server": self.name}


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """Filesystem sandbox for tool execution.

    Tools that receive a path resolve it inside ``workdir``; absolute paths
    outside the workdir (or under a deny path) raise :class:`SandboxError`.
    """

    def __init__(
        self,
        workdir: str,
        enabled: bool = True,
        deny_paths: list[str] | None = None,
        max_output_bytes: int = 1_048_576,
    ):
        self.enabled = enabled
        self.workdir = os.path.abspath(workdir)
        if deny_paths is None:
            deny_paths = ["/etc", "/usr", "/bin", "/sbin", "/System", "/Library"]
        self.deny_paths = [os.path.abspath(p) for p in deny_paths]
        self.max_output_bytes = max_output_bytes
        if enabled:
            os.makedirs(self.workdir, exist_ok=True)

    def resolve(self, path: str) -> str:
        if not self.enabled:
            return os.path.abspath(path)
        target = os.path.abspath(os.path.join(self.workdir, path) if not os.path.isabs(path) else path)
        for deny in self.deny_paths:
            if target == deny or target.startswith(deny + os.sep):
                raise SandboxError(f"path {path!r} denied by sandbox denylist")
        if not (target == self.workdir or target.startswith(self.workdir + os.sep)):
            raise SandboxError(f"path {path!r} escapes sandbox workdir {self.workdir!r}")
        return target

    def run(self, cmd: list[str] | str, timeout: float = 30.0, cwd: str | None = None) -> dict[str, Any]:
        """Run a command inside the sandbox.  Returns {exit, stdout, stderr}."""
        argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
        workdir = self.resolve(cwd or ".") if self.enabled else (cwd or os.getcwd())
        try:
            proc = subprocess.run(
                argv,
                cwd=workdir,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"exit": 124, "stdout": "", "stderr": f"timeout after {timeout}s", "timeout": True}
        except FileNotFoundError:
            return {"exit": 127, "stdout": "", "stderr": f"command not found: {argv[0]!r}"}
        return {
            "exit": proc.returncode,
            "stdout": proc.stdout[: self.max_output_bytes].decode("utf-8", "replace"),
            "stderr": proc.stderr[: self.max_output_bytes].decode("utf-8", "replace"),
        }


# ---------------------------------------------------------------------------
# JSON-RPC over stdio (client side)
# ---------------------------------------------------------------------------


class StdioMcpClient:
    """Talk JSON-RPC 2.0 to a child-process MCP server over stdio."""

    def __init__(self, name: str, cmd: str, cwd: str | None = None,
                 env: dict[str, str] | None = None, timeout_ms: int = 30_000):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env = {**os.environ, **(env or {})}
        self.timeout_s = timeout_ms / 1000.0
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._id = 0

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            shlex.split(self.cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
        )

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.start()
        assert self._proc and self._proc.stdin and self._proc.stdout
        with self._lock:
            self._id += 1
            rid = self._id
            payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
            line = (json.dumps(payload) + "\n").encode()
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
                raw = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise ConnectionError(f"mcp server {self.name!r} died: {exc}") from exc
            if not raw:
                raise ConnectionError(f"mcp server {self.name!r} closed stdout")
            msg = json.loads(raw.decode())
        if "error" in msg:
            raise RuntimeError(f"mcp error from {self.name}: {msg['error']}")
        return msg.get("result")

    def tools(self) -> list[Tool]:
        res = self.request("tools/list")
        return res.get("tools", res if isinstance(res, list) else [])

    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        return self.request("tools/call", {"name": tool, "arguments": arguments})

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        finally:
            self._proc = None

    def health(self) -> dict[str, Any]:
        try:
            self.request("ping")
            return {"status": "ok", "server": self.name}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "server": self.name, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class McpEntry:
    def __init__(self, name: str, server: McpServer | None, spec: dict[str, Any] | None = None):
        self.name = name
        self.server = server          # in-process server
        self.spec = spec or {}       # {cmd, cwd, env, timeout_ms} for external
        self._client: StdioMcpClient | None = None
        self.registered_at = time.time()

    # -- in-process ---------------------------------------------------------
    def is_external(self) -> bool:
        return self.server is None and bool(self.spec.get("cmd"))

    def _ensure_client(self) -> StdioMcpClient:
        if not self.is_external():
            raise ValueError(f"{self.name!r} is not an external server")
        if self._client is None:
            self._client = StdioMcpClient(
                self.name,
                self.spec["cmd"],
                cwd=self.spec.get("cwd"),
                env=self.spec.get("env"),
                timeout_ms=int(self.spec.get("timeout_ms", 30_000)),
            )
        return self._client

    # -- unified facade -------------------------------------------------------
    def tools(self) -> list[Tool]:
        if self.server is not None:
            return self.server.tools()
        return self._ensure_client().tools()

    def call(self, tool: str, arguments: dict[str, Any], ctx: dict[str, Any] | None = None) -> Any:
        ctx = ctx or {}
        if self.server is not None:
            return self.server.call(tool, arguments, ctx)
        return self._ensure_client().call(tool, arguments)

    def health(self) -> dict[str, Any]:
        if self.server is not None:
            try:
                return self.server.health()
            except Exception as exc:  # noqa: BLE001
                return {"status": "error", "server": self.name, "error": str(exc)}
        try:
            return self._ensure_client().health()
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "server": self.name, "error": str(exc)}

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
            self._client = None

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "external" if self.is_external() else "builtin",
            "cmd": self.spec.get("cmd"),
            "description": self.server.description if self.server else (self.spec.get("description") or ""),
        }


class McpRegistry:
    """Holds every registered MCP server and dispatches tool calls."""

    def __init__(self, sandbox: Sandbox | None = None):
        self._entries: dict[str, McpEntry] = {}
        self.sandbox = sandbox
        self._lock = threading.RLock()

    def register_builtin(self, server: McpServer) -> McpEntry:
        with self._lock:
            entry = McpEntry(server.name, server)
            self._entries[server.name] = entry
            return entry

    def register_external(
        self,
        name: str,
        cmd: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_ms: int = 30_000,
        description: str = "",
    ) -> McpEntry:
        spec = {"cmd": cmd, "cwd": cwd, "env": env, "timeout_ms": timeout_ms, "description": description}
        with self._lock:
            entry = McpEntry(name, None, spec)
            self._entries[name] = entry
            return entry

    def unregister(self, name: str) -> bool:
        with self._lock:
            entry = self._entries.pop(name, None)
            if entry is None:
                return False
            entry.stop()
            return True

    def get(self, name: str) -> McpEntry:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"unknown mcp server: {name!r}")
        return entry

    def names(self) -> list[str]:
        return sorted(self._entries)

    def inspect(self, name: str) -> dict[str, Any]:
        entry = self.get(name)
        info = entry.info()
        info["tools"] = entry.tools()
        return info

    def call(self, name: str, tool: str, arguments: dict[str, Any],
             session: str | None = None) -> Any:
        entry = self.get(name)
        ctx = {
            "session": session or str(uuid.uuid4()),
            "sandbox": self.sandbox,
        }
        result = entry.call(tool, arguments, ctx)
        if self.sandbox is not None and isinstance(result, dict) and "stdout" in result:
            result = dict(result)
        return result

    def shutdown(self) -> None:
        for entry in self._entries.values():
            entry.stop()


# ---------------------------------------------------------------------------
# Built-in servers
# ---------------------------------------------------------------------------


class FilesystemServer(McpServer):
    name = "filesystem"
    description = "Sandboxed file read/write/search."

    def __init__(self, root: str = "."):
        # Informational default; per-call paths resolve via ctx["sandbox"].
        self.root = os.path.abspath(root)

    def tools(self) -> list[Tool]:
        return [
            {"name": "read_file", "description": "Read a UTF-8 file inside the sandbox.",
             "input": {"path": "str"}},
            {"name": "write_file", "description": "Write a UTF-8 file inside the sandbox.",
             "input": {"path": "str", "content": "str"}},
            {"name": "list_dir", "description": "List a directory inside the sandbox.",
             "input": {"path": "str"}},
            {"name": "search", "description": "Grep files under the sandbox for a substring.",
             "input": {"path": "str", "pattern": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        sb: Sandbox | None = ctx.get("sandbox")
        if tool == "read_file":
            p = sb.resolve(a["path"]) if sb else a["path"]
            with open(p, "r", encoding="utf-8") as fh:
                return {"content": fh.read()}
        if tool == "write_file":
            p = sb.resolve(a["path"]) if sb else a["path"]
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(a["content"])
            return {"bytes": len(a["content"].encode())}
        if tool == "list_dir":
            p = sb.resolve(a["path"]) if sb else a["path"]
            return {"entries": sorted(os.listdir(p))}
        if tool == "search":
            root = sb.resolve(a["path"]) if sb else a["path"]
            hits = []
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    fp = os.path.join(dirpath, fn)
                    try:
                        with open(fp, "r", encoding="utf-8") as fh:
                            text = fh.read()
                    except (UnicodeDecodeError, OSError):
                        continue
                    if a["pattern"] in text:
                        hits.append(fp)
            return {"matches": hits}
        raise ValueError(f"unknown tool {tool!r}")


class SqliteServer(McpServer):
    name = "sqlite"
    description = "SQLite query + DDL over sandboxed DB files."

    def __init__(self, default_db: str = ".ff/db.sqlite3"):
        import sqlite3

        self._sqlite3 = sqlite3
        self.default_db = default_db

    def tools(self) -> list[Tool]:
        return [
            {"name": "query", "description": "Run a SQL statement; returns rows or rowcount.",
             "input": {"db": "str", "sql": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        if tool != "query":
            raise ValueError(f"unknown tool {tool!r}")
        sb: Sandbox | None = ctx.get("sandbox")
        db = a.get("db") or self.default_db
        db_path = sb.resolve(db) if sb else db
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = self._sqlite3.connect(db_path)
        try:
            cur = conn.execute(a["sql"])
            if a["sql"].strip().lower().startswith("select"):
                cols = [d[0] for d in cur.description or []]
                return {"columns": cols, "rows": [list(r) for r in cur.fetchall()]}
            conn.commit()
            return {"rowcount": cur.rowcount}
        finally:
            conn.close()


class ShellServer(McpServer):
    name = "shell"
    description = "Run shell commands inside the sandbox."

    def tools(self) -> list[Tool]:
        return [
            {"name": "run", "description": "Run a command inside the sandbox; returns exit/stdout/stderr.",
             "input": {"cmd": "str", "timeout_s": "float?"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        if tool != "run":
            raise ValueError(f"unknown tool {tool!r}")
        sb: Sandbox | None = ctx.get("sandbox")
        if sb is None:
            raise SandboxError("shell server requires a sandbox")
        return sb.run(a["cmd"], timeout=float(a.get("timeout_s", 30.0)))


class HttpClientServer(McpServer):
    name = "http_client"
    description = "Outbound HTTP requests (egress-filtered). GET/POST JSON."

    def tools(self) -> list[Tool]:
        return [
            {"name": "get", "description": "HTTP GET a URL and return status + body.",
             "input": {"url": "str"}},
            {"name": "post_json", "description": "HTTP POST JSON to a URL.",
             "input": {"url": "str", "body": "obj"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        import urllib.parse
        import urllib.request

        url = a["url"]
        egress: Any = ctx.get("egress")
        host = urllib.parse.urlsplit(url).netloc.split(":")[0]
        if egress is not None and not egress.allows(host):
            raise PermissionError(f"egress denied: {host!r} not in allowlist")
        if tool == "get":
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
        if tool == "post_json":
            data = json.dumps(a.get("body", {})).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return {"status": resp.status, "body": resp.read().decode("utf-8", "replace")}
        raise ValueError(f"unknown tool {tool!r}")


class VectorServer(McpServer):
    name = "vector_db"
    description = "In-memory vector store: add + cosine search (sha256-derived embeddings)."

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._items: dict[str, dict[str, Any]] = {}

    def _embed(self, text: str) -> list[float]:
        import hashlib

        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = hashlib.sha256(token.encode()).digest()
            for i in range(0, 16, 1):
                vec[i % self.dim] += (h[i] / 255.0) * 2.0 - 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def tools(self) -> list[Tool]:
        return [
            {"name": "add", "description": "Add a document {id?, text, meta?}.",
             "input": {"text": "str", "id": "str?", "meta": "obj?"}},
            {"name": "search", "description": "Cosine top-k search.",
             "input": {"query": "str", "k": "int?"}},
            {"name": "count", "description": "Number of stored documents.", "input": {}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        if tool == "add":
            did = a.get("id") or str(uuid.uuid4())
            self._items[did] = {"text": a["text"], "meta": a.get("meta") or {}, "vec": self._embed(a["text"])}
            return {"id": did, "count": len(self._items)}
        if tool == "search":
            qv = self._embed(a["query"])
            scored = sorted(
                ((self._cos(qv, it["vec"]), did, it) for did, it in self._items.items()),
                key=lambda t: -t[0],
            )
            k = int(a.get("k", 5))
            return {"results": [
                {"id": did, "score": round(s, 4), "text": it["text"], "meta": it["meta"]}
                for s, did, it in scored[:k]
            ]}
        if tool == "count":
            return {"count": len(self._items)}
        raise ValueError(f"unknown tool {tool!r}")


class MarkdownServer(McpServer):
    name = "markdown"
    description = "Markdown utilities: headings, word count, html-like outline."

    def tools(self) -> list[Tool]:
        return [
            {"name": "outline", "description": "Extract the heading outline of a markdown string.",
             "input": {"text": "str"}},
            {"name": "stats", "description": "Word/char/line counts for a markdown string.",
             "input": {"text": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        import re

        text = a["text"]
        if tool == "outline":
            out = []
            for line in text.splitlines():
                m = re.match(r"^(#{1,6})\s+(.*)$", line)
                if m:
                    out.append({"level": len(m.group(1)), "text": m.group(2).strip()})
            return {"outline": out}
        if tool == "stats":
            return {
                "words": len(text.split()),
                "chars": len(text),
                "lines": text.count("\n") - (1 if text.endswith("\n") else 0),
            }
        raise ValueError(f"unknown tool {tool!r}")


class GitServer(McpServer):
    name = "git"
    description = "Git repo operations via the shell sandbox (log, status, diff)."

    def tools(self) -> list[Tool]:
        return [
            {"name": "log", "description": "Last N commit summaries.", "input": {"repo": "str", "n": "int?"}},
            {"name": "status", "description": "git status --porcelain.", "input": {"repo": "str"}},
            {"name": "diff", "description": "git diff (unstaged).", "input": {"repo": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        sb: Sandbox | None = ctx.get("sandbox")
        if sb is None:
            raise SandboxError("git server requires a sandbox")
        n = int(a.get("n", 10))
        if tool == "log":
            return sb.run(f"git -C {a['repo']} log --oneline -n {n}")
        if tool == "status":
            return sb.run(f"git -C {a['repo']} status --porcelain")
        if tool == "diff":
            return sb.run(f"git -C {a['repo']} diff")
        raise ValueError(f"unknown tool {tool!r}")


class EmailServer(McpServer):
    name = "email"
    description = "Local mock mailbox: send/peek (air-gapped friendly; no SMTP)."

    def __init__(self, dir: str = ".ff/mail"):
        self.dir = dir

    def tools(self) -> list[Tool]:
        return [
            {"name": "send", "description": "Write an email file to the mock outbox.",
             "input": {"to": "str", "subject": "str", "body": "str"}},
            {"name": "inbox", "description": "List messages in the mock inbox.", "input": {}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        sb: Sandbox | None = ctx.get("sandbox")
        base = self.dir
        if tool == "send":
            os.makedirs(base, exist_ok=True)
            mid = str(uuid.uuid4())
            path = os.path.join(base, mid + ".eml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"To: {a['to']}\nSubject: {a['subject']}\n\n{a['body']}\n")
            return {"id": mid, "path": path}
        if tool == "inbox":
            if not os.path.isdir(base):
                return {"messages": []}
            return {"messages": sorted(os.listdir(base))}
        raise ValueError(f"unknown tool {tool!r}")


class SlackServer(McpServer):
    name = "slack"
    description = "Slack webhook notifications (egress-filtered)."

    def __init__(self, webhook: str | None = None):
        self.webhook = webhook

    def tools(self) -> list[Tool]:
        return [
            {"name": "notify", "description": "Post a message to the configured webhook.",
             "input": {"text": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        if tool != "notify":
            raise ValueError(f"unknown tool {tool!r}")
        if not self.webhook:
            return {"delivered": False, "reason": "no webhook configured (offline stub)"}
        import urllib.request

        egress = ctx.get("egress")
        if egress is not None:
            import urllib.parse

            host = urllib.parse.urlsplit(self.webhook).netloc.split(":")[0]
            if not egress.allows(host):
                raise PermissionError(f"egress denied: {host!r}")
        data = json.dumps({"text": a["text"]}).encode()
        req = urllib.request.Request(self.webhook, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"delivered": True, "status": resp.status}


class WebScraperServer(McpServer):
    name = "web_scraper"
    description = "Fetch a URL and extract visible text (no JS)."

    def tools(self) -> list[Tool]:
        return [
            {"name": "fetch", "description": "Fetch a URL and return stripped text.",
             "input": {"url": "str"}},
        ]

    def call(self, tool: str, a: dict[str, Any], ctx: dict[str, Any]) -> Any:
        import re
        import urllib.parse
        import urllib.request

        url = a["url"]
        egress = ctx.get("egress")
        if egress is not None:
            host = urllib.parse.urlsplit(url).netloc.split(":")[0]
            if not egress.allows(host):
                raise PermissionError(f"egress denied: {host!r}")
        with urllib.request.urlopen(url, timeout=15) as resp:
            html = resp.read().decode("utf-8", "replace")
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return {"url": url, "text": text[:50_000]}


BUILTIN_SERVERS: list[Callable[[], McpServer]] = [
    lambda: FilesystemServer(),
    lambda: SqliteServer(),
    lambda: ShellServer(),
    lambda: HttpClientServer(),
    lambda: VectorServer(),
    lambda: MarkdownServer(),
    lambda: GitServer(),
    lambda: EmailServer(),
    lambda: SlackServer(),
    lambda: WebScraperServer(),
]

#: Names advertised on the landing page (postgres & browser are adapter stubs
#: in 0.9.x that require optional runtime deps; they register when available).
ADVERTISED_NAMES = [
    "filesystem", "sqlite", "postgres", "web_scraper", "git", "email",
    "shell", "browser", "slack", "vector_db", "http_client", "markdown",
]


def build_registry(config=None, base_dir: str = ".") -> McpRegistry:
    """Create a registry pre-loaded with every available built-in server."""
    workdir = os.path.join(base_dir, ".ff", "sandbox")
    sandbox_cfg = config.get("orchestrator.mcp.sandbox") if config is not None else None
    sb = Sandbox(
        workdir=workdir,
        enabled=bool(sandbox_cfg.get("enabled", True)) if sandbox_cfg else True,
        deny_paths=(sandbox_cfg.get("deny_paths") if sandbox_cfg else None) or ["/etc", "/usr", "/boot", "/var"],
        max_output_bytes=int(sandbox_cfg.get("max_output_bytes", 1_048_576)) if sandbox_cfg else 1_048_576,
    )
    reg = McpRegistry(sandbox=sb)
    for factory in BUILTIN_SERVERS:
        reg.register_builtin(factory())
    if config is not None:
        for spec in config.get("orchestrator.mcp.servers", []) or []:
            if not isinstance(spec, dict) or not spec.get("name") or not spec.get("cmd"):
                continue
            reg.register_external(
                spec["name"],
                spec["cmd"],
                cwd=spec.get("cwd"),
                env=spec.get("env"),
                timeout_ms=int(spec.get("timeout_ms", 30_000)),
                description=spec.get("description", ""),
            )
    return reg
