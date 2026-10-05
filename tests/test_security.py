"""Tests for RBAC, PII redaction, and the egress allowlist filter."""

from __future__ import annotations

import pytest

from frozenfish.security import (
    EgressFilter,
    RBAC,
    egress_from_config,
    parse_permission,
    rbac_from_config,
    redact_pii,
)

ROLES = {
    "admin": {"permissions": ["*"]},
    "operator": {"permissions": ["orchestrator:read", "workflow:*", "mcp:call"]},
    "viewer": {"permissions": ["orchestrator:read", "stats:read", "audit:read"]},
}


class TestParsePermission:
    def test_full(self):
        assert parse_permission("workflow:run:x") == ("workflow", "run", "x")

    def test_wildcards_fill_in(self):
        assert parse_permission("workflow:*") == ("workflow", "*", "*")
        assert parse_permission("*") == ("*", "*", "*")

    def test_malformed(self):
        with pytest.raises(ValueError, match="malformed permission"):
            parse_permission("not a permission")


class TestRBAC:
    def _rbac(self):
        return RBAC(
            roles={r: set(v["permissions"]) for r, v in ROLES.items()},
            users={"alice": "admin", "bob": "operator", "carol": "viewer"},
        )

    def test_admin_wildcard(self):
        r = self._rbac()
        assert r.can("alice", "anything:at:all") is True

    def test_operator_scope(self):
        r = self._rbac()
        assert r.can("bob", "workflow:run") is True
        assert r.can("bob", "orchestrator:read") is True
        assert r.can("bob", "audit:read") is False
        assert r.can("bob", "user:add") is False

    def test_viewer_read_only(self):
        r = self._rbac()
        assert r.can("carol", "stats:read") is True
        assert r.can("carol", "workflow:run") is False

    def test_unknown_user_denied(self):
        r = self._rbac()
        assert r.can("eve", "orchestrator:read") is False

    def test_role_of(self):
        r = self._rbac()
        assert r.role_of("bob") == "operator"

    def test_add_remove_user(self):
        r = self._rbac()
        r.add_user("dave", "viewer")
        assert r.can("dave", "audit:read") is True
        with pytest.raises(ValueError):
            r.add_user("dave", "not-a-role")
        assert r.remove_user("dave") is True
        assert r.remove_user("dave") is False

    def test_grant_persists_to_role(self):
        r = self._rbac()
        assert r.grant("bob", "audit:read") is True
        # granted to the whole role, not just bob
        assert r.grant("bob", "audit:read") is False  # already present
        r.add_user("frank", "operator")
        assert r.can("frank", "audit:read") is True

    def test_grant_validates_format(self):
        r = self._rbac()
        with pytest.raises(ValueError, match="malformed permission"):
            r.grant("bob", "no-colon-here")

    def test_rbac_from_config(self):
        from frozenfish.config import Config

        cfg = Config.load(None)
        rbac = rbac_from_config(cfg)
        assert rbac.can("nobody", "*") is False  # no users configured


class TestRedactPII:
    def test_email(self):
        assert "a@b.co" not in redact_pii("mail me at a@b.co ok")
        assert "[REDACTED:EMAIL]" in redact_pii("mail me at a@b.co ok")

    def test_ssn(self):
        out = redact_pii("ssn 123-45-6789")
        assert "[REDACTED:SSN]" in out and "123-45-6789" not in out

    def test_phone_cn(self):
        out = redact_pii("call 13812345678")
        assert "[REDACTED:PHONE]" in out and "13812345678" not in out

    def test_api_key(self):
        out = redact_pii('api_key = "supersecretvalue"')
        assert "supersecretvalue" not in out
        assert "REDACTED" in out

    def test_clean_text_unchanged(self):
        text = "the quick brown fox"
        assert redact_pii(text) == text


class TestEgressFilter:
    def test_empty_allows_nothing(self):
        f = EgressFilter([])
        assert f.allows("example.com") is False

    def test_hostname(self):
        f = EgressFilter(["example.com"])
        assert f.allows("example.com") is True
        assert f.allows("other.com") is False

    def test_wildcard(self):
        f = EgressFilter(["*.example.com"])
        assert f.allows("api.example.com") is True
        assert f.allows("example.com") is False  # wildcard excludes apex

    def test_cidr(self):
        f = EgressFilter(["10.0.0.0/8"])
        assert f.allows("10.1.2.3") is True
        assert f.allows("192.168.0.1") is False

    def test_bare_ip(self):
        f = EgressFilter(["192.168.1.5"])
        assert f.allows("192.168.1.5") is True
        assert f.allows("192.168.1.6") is False

    def test_egress_from_config(self):
        from frozenfish.config import Config

        cfg = Config.load(None)
        f = egress_from_config(cfg)
        assert f.allows("example.com") is False  # default allowlist is empty
