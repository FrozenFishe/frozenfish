"""Configuration schema + loader for Frozen Fish.

Frozen Fish ships a stdlib-only YAML subset parser (``miniyaml``) so the
package works on air-gapped machines without PyYAML.  When PyYAML *is*
installed it is used transparently for full YAML support.

Schema version: "0.9"
"""

from __future__ import annotations

import copy
import os
import re
from typing import Any

SCHEMA_VERSION = "0.9"

#: Canonical schema defaults.  A user config file only needs to override what
#: it changes; every key below exists in the final resolved config.
DEFAULT_CONFIG: dict[str, Any] = {
    "version": SCHEMA_VERSION,
    "orchestrator": {
        "mcp": {
            "servers": [],          # list of {name, cmd, cwd?, env?, timeout_ms?}
            "sandbox": {
                "enabled": True,
                "workdir": ".ff/sandbox",
                "deny_paths": ["/etc", "/usr", "/boot", "/var"],
                "max_output_bytes": 1_048_576,
            },
            "timeout_ms": 30_000,
        },
        "routing": {
            "strategy": "backends",
            "fallback": "local",
            "weights": {"util": 0.5, "warm": 0.3, "queue": 0.2},
            "sticky": True,
        },
        "context": {
            "cag": {"enabled": True, "max_tokens": 32_768, "warm_on_boot": True},
            "car": {"enabled": True, "partition_by": "tool"},
            "rag": {"enabled": True, "top_k": 5, "similarity_threshold": 0.35},
            "mmr": {"enabled": True, "lambda": 0.7},
        },
        "observability": {
            "metrics": True,
            "logs": True,
            "otlp_endpoint": None,
        },
        "security": {
            "rbac": {
                "roles": {
                    "admin": {"permissions": ["*"]},
                    "operator": {
                        "permissions": ["orchestrator:read", "workflow:*", "mcp:call"]
                    },
                    "viewer": {
                        "permissions": ["orchestrator:read", "stats:read", "audit:read"]
                    },
                },
                "users": {},  # name -> role
            },
            "audit": {"enabled": True, "path": ".ff/audit.db"},
            "egress_allowlist": [],
        },
        "web": {
            "host": "127.0.0.1",
            "port": 8321,
            "auth_token": None,
        },
        "workflows": {
            "dir": "workflows",
            "state_db": ".ff/workflows.db",
            "retry": {"max_attempts": 3, "backoff_seconds": 2.0},
        },
    },
}

# ---------------------------------------------------------------------------
# miniyaml - a tiny, dependency-free YAML subset
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _scalar(raw: str) -> Any:
    """Parse a single YAML scalar."""
    s = raw.strip()
    if s == "" or s in ("~", "null", "Null", "NULL"):
        return None
    if s in ("true", "True", "TRUE", "yes", "on"):
        return True
    if s in ("false", "False", "FALSE", "no", "off"):
        return False
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _strip_comment(line: str) -> str:
    """Remove a trailing ``# comment`` that is not inside quotes."""
    out = []
    quote = None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _split_inline_list(s: str) -> list[Any]:
    body = s.strip()[1:-1].strip()
    if not body:
        return []
    parts: list[str] = []
    depth = 0
    cur = ""
    for ch in body:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [_scalar(p) for p in parts if p.strip()]


def miniyaml_load(text: str) -> Any:
    """Parse a YAML subset into Python objects.

    Supports: nested mappings by 2-space indent, ``- item`` sequences,
    inline ``[a, b]`` lists, scalars (int/float/bool/null/str), quoted
    strings and ``#`` comments.  Multi-line strings and anchors are not
    supported - keep workflow files within the subset.
    """
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.strip()))

    pos = 0

    def parse_block(indent: int) -> Any:
        nonlocal pos
        if pos >= len(lines):
            return None
        cur_indent, cur = lines[pos]
        if cur_indent < indent:
            return None
        if cur.startswith("- ") or cur == "-":
            return parse_sequence(cur_indent)
        return parse_mapping(cur_indent)

    def parse_sequence(indent: int) -> list[Any]:
        nonlocal pos
        out: list[Any] = []
        while pos < len(lines):
            ci, cur = lines[pos]
            if ci != indent or not (cur.startswith("- ") or cur == "-"):
                break
            item = cur[1:].strip()
            pos += 1
            if not item:
                # nested block belongs to this list item
                if pos < len(lines) and lines[pos][0] > indent:
                    out.append(parse_block(lines[pos][0]))
                else:
                    out.append(None)
            elif ":" in item and not item.startswith("["):
                # inline mapping start, e.g. "- name: foo"
                key, val = item.split(":", 1)
                d = {key.strip(): _scalar(val) if val.strip() else None}
                if pos < len(lines) and lines[pos][0] > indent:
                    nested = parse_block(lines[pos][0])
                    if isinstance(nested, dict):
                        d.update(nested)
                out.append(d)
            else:
                out.append(_scalar(item))
        return out

    def parse_mapping(indent: int) -> dict[str, Any]:
        nonlocal pos
        out: dict[str, Any] = {}
        while pos < len(lines):
            ci, cur = lines[pos]
            if ci != indent or cur.startswith("- "):
                break
            if ":" not in cur:
                raise ValueError(f"miniyaml: bad mapping line: {cur!r}")
            key, val = cur.split(":", 1)
            key = key.strip().strip('"').strip("'")
            val = val.strip()
            pos += 1
            if val == "":
                if pos < len(lines) and lines[pos][0] > indent:
                    out[key] = parse_block(lines[pos][0])
                else:
                    out[key] = None
            elif val.startswith("["):
                out[key] = _split_inline_list(val)
            else:
                out[key] = _scalar(val)
        return out

    result = parse_block(0)
    return result if result is not None else {}


def load_yaml(text: str) -> Any:
    """Load YAML using PyYAML when available, else the mini parser."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return miniyaml_load(text)


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------


class Config:
    """Resolved, validated configuration."""

    def __init__(self, data: dict[str, Any], source: str | None = None):
        self._data = data
        self.source = source

    # -- access -----------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def to_dict(self) -> dict[str, Any]:
        return self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    # -- construction -----------------------------------------------------
    @classmethod
    def default(cls) -> "Config":
        return cls(copy.deepcopy(DEFAULT_CONFIG), source="<defaults>")

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Load config: defaults <- file <- $FF_CONFIG file."""
        data = copy.deepcopy(DEFAULT_CONFIG)
        candidate = path or os.environ.get("FF_CONFIG")
        if candidate and os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as fh:
                user = load_yaml(fh.read())
            if not isinstance(user, dict):
                raise ValueError(f"config root must be a mapping: {candidate}")
            _deep_merge(data, user)
            cfg = cls(data, source=candidate)
            cfg.validate()
            return cfg
        if candidate:
            raise FileNotFoundError(f"config file not found: {candidate}")
        return cls(data, source="<defaults>")

    def validate(self) -> None:
        data = self._data
        ver = data.get("version")
        if ver != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported config version {ver!r} (expected {SCHEMA_VERSION!r})"
            )
        weights = data["orchestrator"]["routing"]["weights"]
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"routing weights must sum to 1.0, got {total:.3f}: {weights}"
            )
        lam = data["orchestrator"]["context"]["mmr"]["lambda"]
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"mmr.lambda must be in [0, 1], got {lam}")
        cag = data["orchestrator"]["context"]["cag"]
        if cag["max_tokens"] < 1024:
            raise ValueError(f"cag.max_tokens must be >= 1024, got {cag['max_tokens']}")
        users = data["orchestrator"]["security"]["rbac"]["users"]
        roles = set(data["orchestrator"]["security"]["rbac"]["roles"])
        for name, role in users.items():
            if role not in roles:
                raise ValueError(f"user {name!r} has unknown role {role!r}")


def _deep_merge(base: dict, extra: dict) -> dict:
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def dump_defaults() -> str:
    """Serialize the canonical defaults (used by `ff config show` and tests)."""
    return _dump(DEFAULT_CONFIG)


def _dump(node: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, dict):
        if not node:
            return pad + "{}"
        out = []
        for k, v in node.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:\n{_dump(v, indent + 1)}")
            else:
                out.append(f"{pad}{k}: {_fmt(v)}")
        return "\n".join(out)
    if isinstance(node, list):
        if not node:
            return pad + "[]"
        return "\n".join(
            f"{pad}- {_fmt(v) if not isinstance(v, (dict, list)) else _dump(v, indent + 1)}"
            for v in node
        )
    return pad + _fmt(node)


def _fmt(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v if re.match(r"^[A-Za-z0-9_./-]+$", v) else f'"{v}"'
    return str(v)
