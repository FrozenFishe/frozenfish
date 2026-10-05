"""``ff`` command-line interface (pure stdlib argparse).

Groups:
  ff orchestrator start|stop|status|reload
  ff mcp register|list|inspect|call|unregister
  ff workflow deploy|list|logs|pause|resume|delete
  ff context cag build|invalidate
  ff car partition --by tool
  ff rag reindex
  ff stats
  ff gpu list|route
  ff router policy set|get
  ff dashboard --metrics gpu,context,cost
  ff user add|remove|roles
  ff audit tail|export
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Config

# ---------------------------------------------------------------------------
# Persistent state: a tiny "orchestrator handle" file so that ``stop`` /
# ``status`` in later shell invocations know which process owns the daemon.
# The daemon itself is a backgrounded python subprocess; we record pid here.
# ---------------------------------------------------------------------------


def _ff_dir(base: str = ".") -> str:
    return os.path.join(base, ".ff")


def _pid_path(base: str = ".") -> str:
    return os.path.join(_ff_dir(base), "orchestrator.pid")


def _read_pid(base: str) -> int | None:
    try:
        with open(_pid_path(base)) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _write_pid(base: str, pid: int) -> None:
    os.makedirs(_ff_dir(base), exist_ok=True)
    with open(_pid_path(base), "w", encoding="utf-8") as fh:
        fh.write(str(pid))


def _clear_pid(base: str) -> None:
    try:
        os.remove(_pid_path(base))
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_config(args) -> Config:
    path = getattr(args, "config", None) or os.environ.get("FF_CONFIG")
    return Config.load(path)


def _emit(data) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def cmd_orchestrator(args) -> int:
    base = args.base
    if args.action == "start":
        cfg = _load_config(args)
        pid = _read_pid(base)
        if pid and _pid_alive(pid):
            print(f"already running (pid {pid})")
            return 0
        # Spawn the daemon as a detached subprocess.
        import subprocess

        os.makedirs(_ff_dir(base), exist_ok=True)
        cmd = [sys.executable, "-m", "frozenfish._daemon", "--base", base]
        if args.config:
            cmd += ["--config", args.config]
        proc = subprocess.Popen(
            cmd,
            stdout=open(os.path.join(_ff_dir(base), "orchestrator.log"), "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _write_pid(base, proc.pid)
        print(f"started (pid {proc.pid}); dashboard at "
              f"http://{cfg.get('orchestrator.web.host')}:{cfg.get('orchestrator.web.port')}")
        return 0
    if args.action == "stop":
        pid = _read_pid(base)
        if not pid:
            print("not running")
            return 0
        import signal

        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        _clear_pid(base)
        print(f"stopped (pid {pid})")
        return 0
    if args.action == "status":
        pid = _read_pid(base)
        print(json.dumps({
            "running": bool(pid and _pid_alive(pid)),
            "pid": pid,
        }, indent=2))
        return 0
    if args.action == "reload":
        pid = _read_pid(base)
        if not pid or not _pid_alive(pid):
            print("not running; nothing to reload")
            return 1
        import signal

        os.kill(pid, signal.SIGHUP)
        print("reloaded")
        return 0
    return 2


def _daemon_main(argv) -> int:
    """Entry used by ``python -m frozenfish._daemon``."""
    import signal
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--config", default=None)
    a = parser.parse_args(argv)

    from .orchestrator import Orchestrator
    from .dashboard import Dashboard

    cfg = _load_config(a)
    orch = Orchestrator(cfg, base_dir=a.base)
    orch.start(serve_dashboard=False)
    dash = Dashboard(
        orch,
        host=cfg.get("orchestrator.web.host", "127.0.0.1"),
        port=int(cfg.get("orchestrator.web.port", 8321)),
        auth_token=cfg.get("orchestrator.web.auth_token"),
    )
    port = dash.start(blocking=False)

    stop_evt = []

    def _term(signum, frame):
        dash.stop()
        orch.stop()
        _clear_pid(a.base)
        os._exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        _term(signal.SIGTERM, None)
    return 0


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


def cmd_mcp(args) -> int:
    cfg = _load_config(args)
    from .mcp import build_registry

    reg = build_registry(cfg, base_dir=args.base)
    try:
        if args.action == "list":
            _emit([e.info() for e in reg._entries.values()])
        elif args.action == "inspect":
            entry = reg.get(args.name)
            _emit(entry.info())
        elif args.action == "register":
            spec = json.loads(args.spec)
            reg.register_external(
                spec["name"], cmd=spec["cmd"], cwd=spec.get("cwd"),
                env=spec.get("env"), timeout_ms=spec.get("timeout_ms", 30_000),
            )
            print(f"registered {spec['name']}")
        elif args.action == "unregister":
            reg.unregister(args.name)
            print(f"unregistered {args.name}")
        elif args.action == "call":
            entry = reg.get(args.name)
            argv = json.loads(args.arguments) if args.arguments else {}
            result = reg.call(args.name, args.tool, argv)
            _emit(result)
    finally:
        reg.shutdown()
    return 0


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------


def cmd_workflow(args) -> int:
    cfg = _load_config(args)
    from .mcp import build_registry
    from .workflow import engine_from_config

    reg = build_registry(cfg, base_dir=args.base)
    try:
        eng = engine_from_config(cfg, reg, base_dir=args.base)
        if args.action == "deploy":
            with open(args.path, "r", encoding="utf-8") as fh:
                doc = __import__("frozenfish.config", fromlist=["miniyaml_load"]).miniyaml_load(fh.read())
            if not isinstance(doc, dict) or not doc.get("name"):
                print("workflow yaml must be a mapping with a 'name' key", file=sys.stderr)
                return 1
            eng.load_workflow(doc)
            print(f"deployed {doc['name']}")
        elif args.action == "list":
            eng.load_dir()
            out = []
            for name in sorted(eng.workflows):
                out.append({
                    "name": name,
                    "enabled": eng.is_enabled(name),
                    "trigger": eng.workflows[name].get("trigger"),
                })
            _emit(out)
        elif args.action == "logs":
            _emit(eng.logs(args.workflow, limit=args.limit))
        elif args.action == "pause":
            eng.pause(args.name)
            print(f"paused {args.name}")
        elif args.action == "resume":
            eng.resume(args.name)
            print(f"resumed {args.name}")
        elif args.action == "delete":
            eng.delete(args.name)
            print(f"deleted {args.name}")
        elif args.action == "run":
            rec = eng.run_workflow(args.name)
            _emit(rec)
    finally:
        reg.shutdown()
    return 0


# ---------------------------------------------------------------------------
# context / car / rag
# ---------------------------------------------------------------------------


def cmd_context(args) -> int:
    cfg = _load_config(args)
    from .context import CagCache, CarIndex, RagIndex

    if args.sub == "cag":
        cag = CagCache(
            path=os.path.join(args.base, ".ff", "cag.sqlite3"),
            max_tokens=int(cfg.get("orchestrator.context.cag.max_tokens", 32_768)),
            enabled=bool(cfg.get("orchestrator.context.cag.enabled", True)),
        )
        if args.action == "build":
            with open(args.preface, "r", encoding="utf-8") as fh:
                key = cag.build(fh.read(), fingerprint=args.fingerprint or "")
            print(f"built {key[:16]}…")
        elif args.action == "invalidate":
            n = cag.invalidate()
            print(f"invalidated {n} entries")
    elif args.sub == "car":
        car = CarIndex(partition_by="tool",
                       enabled=bool(cfg.get("orchestrator.context.car.enabled", True)))
        if args.action == "partition":
            car._partitions.setdefault("tool", CarIndex(partition_by="tool")._partitions.get("tool"))
            print("partitions:", car.partitions())
    elif args.sub == "rag":
        rag = RagIndex(
            top_k=int(cfg.get("orchestrator.context.rag.top_k", 5)),
            threshold=float(cfg.get("orchestrator.context.rag.similarity_threshold", 0.35)),
            mmr=bool(cfg.get("orchestrator.context.mmr.enabled", True)),
            mmr_lambda=float(cfg.get("orchestrator.context.mmr.lambda", 0.7)),
            enabled=bool(cfg.get("orchestrator.context.rag.enabled", True)),
        )
        if args.action == "reindex":
            texts = []
            with open(args.corpus, "r", encoding="utf-8") as fh:
                for chunk in fh.read().split("\n\n"):
                    if chunk.strip():
                        texts.append(chunk.strip())
            n = rag.reindex(texts)
            print(f"indexed {n} chunks")
    return 0


# ---------------------------------------------------------------------------
# gpu / router / stats
# ---------------------------------------------------------------------------


def cmd_gpu(args) -> int:
    cfg = _load_config(args)
    from .gpu import backend_from_config
    from .router import Router, router_from_config

    reg = backend_from_config(cfg)
    if not reg.names():
        reg.discover(args.base)
    if args.action == "list":
        _emit([b.snapshot() for b in reg.all()])
    elif args.action == "route":
        r = router_from_config(cfg, reg)
        d = r.route(args.model, session=args.session)
        _emit(d.to_dict())
    return 0


def cmd_router(args) -> int:
    cfg = _load_config(args)
    from .gpu import backend_from_config
    from .router import Router

    reg = backend_from_config(cfg)
    if not reg.names():
        reg.discover(args.base)
    r = router_from_config(cfg, reg)
    if args.action == "set":
        r.set_policy(args.policy)
        print(f"policy={r.get_policy()}")
    elif args.action == "get":
        print(r.get_policy())
    return 0


def cmd_stats(args) -> int:
    cfg = _load_config(args)
    from .gpu import backend_from_config
    from .router import Router, router_from_config
    from .context import CagCache, CarIndex, RagIndex, ContextRouter

    reg = backend_from_config(cfg)
    if not reg.names():
        reg.discover(args.base)
    cag = CagCache(".ff/cag.sqlite3",
                   max_tokens=int(cfg.get("orchestrator.context.cag.max_tokens", 32_768)),
                   enabled=bool(cfg.get("orchestrator.context.cag.enabled", True)))
    car = CarIndex(enabled=bool(cfg.get("orchestrator.context.car.enabled", True)))
    rag = RagIndex(top_k=int(cfg.get("orchestrator.context.rag.top_k", 5)),
                   threshold=float(cfg.get("orchestrator.context.rag.similarity_threshold", 0.35)),
                   mmr=bool(cfg.get("orchestrator.context.mmr.enabled", True)),
                   mmr_lambda=float(cfg.get("orchestrator.context.mmr.lambda", 0.7)),
                   enabled=bool(cfg.get("orchestrator.context.rag.enabled", True)))
    ctx = ContextRouter(cag, car, rag)
    _emit(ctx.stats())
    return 0


def cmd_dashboard(args) -> int:
    cfg = _load_config(args)
    metrics = [m.strip() for m in args.metrics.split(",")] if args.metrics else []
    print(json.dumps({"metrics": metrics,
                      "url": f"http://{cfg.get('orchestrator.web.host')}:"
                             f"{cfg.get('orchestrator.web.port')}"}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# user / audit
# ---------------------------------------------------------------------------


def cmd_user(args) -> int:
    cfg = _load_config(args)
    from .security import rbac_from_config

    rbac = rbac_from_config(cfg)
    if args.action == "add":
        role = args.role or "viewer"
        if role not in rbac.roles:
            print(f"unknown role {role!r}; available: {sorted(rbac.roles)}", file=sys.stderr)
            return 1
        rbac.add_user(args.name, role)
        print(f"added {args.name} as {role}")
    elif args.action == "remove":
        rbac.remove_user(args.name)
        print(f"removed {args.name}")
    elif args.action == "roles":
        _emit({u: rbac.role_of(u) for u in rbac.users})
    return 0


def cmd_audit(args) -> int:
    cfg = _load_config(args)
    from .audit import AuditLog

    log = AuditLog(os.path.join(args.base, ".ff", "audit.db"),
                   enabled=bool(cfg.get("orchestrator.security.audit.enabled", True)))
    try:
        if args.action == "tail":
            events = log.tail(since=args.since, limit=args.limit)
            _emit(events)
        elif args.action == "export":
            _emit(log.export(fmt=args.format))
    finally:
        log.close()
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ff", description="Frozen Fish CLI")
    p.add_argument("--base", default=".", help="project base directory")
    p.add_argument("--config", default=None, help="path to ff.yaml")
    sub = p.add_subparsers(dest="group", required=True)

    o = sub.add_parser("orchestrator", help="start/stop/status/reload the daemon")
    o.add_argument("action", choices=["start", "stop", "status", "reload"])
    o.set_defaults(func=cmd_orchestrator)

    m = sub.add_parser("mcp", help="MCP server management")
    m.add_argument("action",
                   choices=["register", "list", "inspect", "call", "unregister"])
    m.add_argument("name", nargs="?")
    m.add_argument("--spec", help="JSON spec for register")
    m.add_argument("--tool", help="tool name for call")
    m.add_argument("--arguments", help="JSON arguments for call")
    m.set_defaults(func=cmd_mcp)

    w = sub.add_parser("workflow", help="workflow lifecycle")
    w.add_argument("action",
                   choices=["deploy", "list", "logs", "pause", "resume", "delete", "run"])
    w.add_argument("name", nargs="?")
    w.add_argument("--path", help="yaml path for deploy")
    w.add_argument("--workflow", help="filter logs by workflow")
    w.add_argument("--limit", type=int, default=20)
    w.set_defaults(func=cmd_workflow)

    c = sub.add_parser("context", help="context layer ops (cag/car/rag)")
    c.add_argument("sub", choices=["cag", "car", "rag"])
    c.add_argument("action", choices=["build", "invalidate", "partition", "reindex"])
    c.add_argument("--preface", help="preface file for cag build")
    c.add_argument("--fingerprint", help="fingerprint for cag build")
    c.add_argument("--by", default="tool")
    c.add_argument("--corpus", help="corpus file for rag reindex")
    c.set_defaults(func=cmd_context)

    g = sub.add_parser("gpu", help="backend registry")
    g.add_argument("action", choices=["list", "route"])
    g.add_argument("--model", default="default")
    g.add_argument("--session", default=None)
    g.set_defaults(func=cmd_gpu)

    r = sub.add_parser("router", help="router policy")
    r.add_argument("action", choices=["set", "get"])
    r.add_argument("policy", nargs="?")
    r.set_defaults(func=cmd_router)

    s = sub.add_parser("stats", help="context stats")
    s.set_defaults(func=cmd_stats)

    d = sub.add_parser("dashboard", help="dashboard info")
    d.add_argument("--metrics", default="")
    d.set_defaults(func=cmd_dashboard)

    u = sub.add_parser("user", help="RBAC users")
    u.add_argument("action", choices=["add", "remove", "roles"])
    u.add_argument("name", nargs="?")
    u.add_argument("--role", default="viewer")
    u.set_defaults(func=cmd_user)

    a = sub.add_parser("audit", help="audit log")
    a.add_argument("action", choices=["tail", "export"])
    a.add_argument("--since", default="1h")
    a.add_argument("--format", default="json")
    a.add_argument("--limit", type=int, default=100)
    a.set_defaults(func=cmd_audit)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
