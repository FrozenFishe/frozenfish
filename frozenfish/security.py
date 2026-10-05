"""Security primitives: RBAC, PII redaction, egress allowlist."""

from __future__ import annotations

import fnmatch
import ipaddress
import re
from typing import Any

# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

_PERMISSION_RE = re.compile(r"^(\*|[a-z_]+)(?::(\*|[a-z_]+))?(?::[a-z_*]+)?$")


def parse_permission(perm: str) -> tuple[str, str, str]:
    """Split ``area:resource:action`` into (area, resource, action)."""
    m = _PERMISSION_RE.match(perm)
    if not m:
        raise ValueError(f"malformed permission: {perm!r}")
    parts = perm.split(":")
    return parts[0], parts[1] if len(parts) > 1 else "*", parts[2] if len(parts) > 2 else "*"


class RBAC:
    """Role-based access control over permission strings ``area:resource:action``.

    Matching rules (most specific wins):
      * ``*`` in any segment matches everything in that position.
      * ``workflow:*`` matches ``workflow:deploy:run``.
      * ``workflow:*:read`` matches ``workflow:deploy:read``.
      * exact strings always match.
    """

    def __init__(self, roles: dict[str, set[str]], users: dict[str, str] | None = None):
        self.roles = {name: set(perms) for name, perms in roles.items()}
        self.users = dict(users or {})

    def add_user(self, name: str, role: str) -> None:
        if role not in self.roles:
            raise ValueError(f"unknown role: {role!r}")
        self.users[name] = role

    def remove_user(self, name: str) -> bool:
        return self.users.pop(name, None) is not None

    def role_of(self, user: str) -> str | None:
        return self.users.get(user)

    def _perm_matches(self, granted: str, requested: str) -> bool:
        try:
            g_area, g_res, g_act = parse_permission(granted)
        except ValueError:
            return False
        requested_parts = requested.split(":")
        r_area = requested_parts[0]
        r_res = requested_parts[1] if len(requested_parts) > 1 else "*"
        r_act = requested_parts[2] if len(requested_parts) > 2 else "*"
        return (
            (g_area == "*" or g_area == r_area)
            and (g_res == "*" or g_res == r_res)
            and (g_act == "*" or g_act == r_act)
        )

    def can(self, user: str, permission: str) -> bool:
        role = self.users.get(user)
        if role is None:
            return False
        return any(self._perm_matches(g, permission) for g in self.roles[role])

    def grant(self, user: str, permission: str) -> bool:
        """Grant a permission at the *user's role* level (persistent)."""
        role = self.users.get(user)
        if role is None:
            return False
        if not _PERMISSION_RE.match(permission):
            raise ValueError(f"malformed permission: {permission!r}")
        if permission in self.roles[role]:
            return False
        self.roles[role].add(permission)
        return True


def rbac_from_config(cfg) -> RBAC:
    roles = {
        name: set(perms.get("permissions", []))
        for name, perms in (cfg.get("orchestrator.security.rbac.roles", {}) or {}).items()
    }
    users = dict(cfg.get("orchestrator.security.rbac.users", {}) or {})
    return RBAC(roles, users)


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# 11-digit Chinese mobile numbers, 10-digit US, SSN-style; keep conservative.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?86)?1[3-9]\d{9}(?!\d)")
_US_PHONE_RE = re.compile(r"(?<!\d)\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_API_KEY_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[:=]\s*)(\"[^\"]{8,}\"|\'[^\']*\'|[A-Za-z0-9_\-./]{8,})"
)


def redact_pii(text: str) -> str:
    """Return ``text`` with common PII replaced by typed redaction markers."""
    text = _API_KEY_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    text = _EMAIL_RE.sub("[REDACTED:EMAIL]", text)
    text = _SSN_RE.sub("[REDACTED:SSN]", text)
    text = _PHONE_RE.sub("[REDACTED:PHONE]", text)
    text = _US_PHONE_RE.sub("[REDACTED:PHONE]", text)
    return text


# ---------------------------------------------------------------------------
# Egress allowlist
# ---------------------------------------------------------------------------


class EgressFilter:
    """Allow outbound network targets only when they match the allowlist.

    Entries may be:
      * hostnames:   ``api.example.com``
      * wildcard:    ``*.example.com``
      * CIDR:        ``10.0.0.0/8``
      * bare IP:     ``10.1.2.3``
    An empty allowlist means **no egress** (local-first default).
    """

    def __init__(self, entries: list[str] | None):
        self.entries = list(entries or [])
        self._hosts: list[str] = []
        self._cidrs: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for entry in entries or []:
            entry = entry.strip()
            if not entry:
                continue
            if "/" in entry and re.match(r"^\d", entry):
                self._cidrs.append(ipaddress.ip_network(entry, strict=False))
            else:
                self._hosts.append(entry.lower())

    def allows(self, host: str) -> bool:
        host = host.lower()
        for pattern in self._hosts:
            if fnmatch.fnmatch(host, pattern):
                return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(ip in net for net in self._cidrs)


def egress_from_config(cfg) -> EgressFilter:
    return EgressFilter(cfg.get("orchestrator.security.egress_allowlist") or [])


# ---------------------------------------------------------------------------
# Audit event model
# ---------------------------------------------------------------------------


def make_event(
    actor: str,
    action: str,
    target: str,
    outcome: str = "ok",
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a normalized audit event (redacted before storage)."""
    ev: dict[str, Any] = {
        "actor": actor,
        "action": action,
        "target": target,
        "outcome": outcome,
        "detail": detail or {},
    }
    for key in ("actor", "target"):
        ev[key] = redact_pii(str(ev[key]))
    return ev
