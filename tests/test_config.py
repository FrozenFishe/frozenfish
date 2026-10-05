"""Tests for config loading, validation, and the miniyaml parser."""

from __future__ import annotations

import pytest

from frozenfish.config import (
    Config,
    DEFAULT_CONFIG,
    SCHEMA_VERSION,
    miniyaml_load,
)


class TestSchema:
    def test_schema_version(self):
        assert SCHEMA_VERSION == "0.9"

    def test_default_shape(self):
        cfg = Config.load(None)
        assert cfg.get("version") == SCHEMA_VERSION
        assert cfg.get("orchestrator.routing.strategy") == "backends"
        assert cfg.get("orchestrator.routing.sticky") is True
        assert cfg.get("orchestrator.context.cag.max_tokens") == 32768
        assert cfg.get("orchestrator.context.rag.top_k") == 5
        assert cfg.get("orchestrator.context.rag.similarity_threshold") == 0.35
        assert cfg.get("orchestrator.context.mmr.lambda") == 0.7
        assert cfg.get("orchestrator.web.host") == "127.0.0.1"
        assert cfg.get("orchestrator.web.port") == 8321
        assert cfg.get("orchestrator.workflows.retry.max_attempts") == 3
        assert cfg.get("orchestrator.workflows.retry.backoff_seconds") == 2.0
        assert cfg.get("orchestrator.mcp.sandbox.enabled") is True
        assert cfg.get("orchestrator.mcp.sandbox.deny_paths") == [
            "/etc", "/usr", "/boot", "/var",
        ]

    def test_default_validate(self):
        cfg = Config.load(None)
        cfg.validate()  # must not raise


class TestConfigAccess:
    def test_get_dotted(self):
        cfg = Config.load(None)
        assert cfg.get("orchestrator.web.port") == 8321
        assert cfg.get("orchestrator.web.port", 1234) == 8321

    def test_get_missing_default(self):
        cfg = Config.load(None)
        assert cfg.get("no.such.key", "dflt") == "dflt"
        assert cfg.get("no.such.key") is None

    def test_getitem(self):
        cfg = Config.load(None)
        assert cfg["orchestrator"]["web"]["port"] == 8321

    def test_as_dict(self):
        cfg = Config.load(None)
        d = cfg.as_dict()
        assert d["version"] == SCHEMA_VERSION
        assert isinstance(d["orchestrator"], dict)


class TestValidation:
    def _cfg(self, **over):
        import copy
        data = copy.deepcopy(DEFAULT_CONFIG)
        node = data
        for k in over:
            parts = k.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = over[k]
        return Config(data)

    def test_weights_must_sum_to_one(self):
        cfg = self._cfg(**{"orchestrator.routing.weights.util": 1.0})
        with pytest.raises(ValueError):
            cfg.validate()

    def test_mmr_lambda_range(self):
        cfg = self._cfg(**{"orchestrator.context.mmr.lambda": 1.5})
        with pytest.raises(ValueError):
            cfg.validate()

    def test_cag_max_tokens_floor(self):
        cfg = self._cfg(**{"orchestrator.context.cag.max_tokens": 512})
        with pytest.raises(ValueError):
            cfg.validate()


class TestMiniYaml:
    def test_scalars(self):
        d = miniyaml_load(
            "a: 1\nb: 1.5\nc: true\nd: no\ne: null\nf: hi\n"
        )
        assert d == {"a": 1, "b": 1.5, "c": True, "d": False, "e": None, "f": "hi"}

    def test_nested(self):
        d = miniyaml_load(
            "top:\n  child: 2\n  kid:\n    deep: x\n"
        )
        assert d == {"top": {"child": 2, "kid": {"deep": "x"}}}

    def test_sequence(self):
        d = miniyaml_load("items:\n  - a\n  - b\n")
        assert d == {"items": ["a", "b"]}

    def test_inline_list(self):
        d = miniyaml_load("deny: [/etc, /usr]\n")
        assert d == {"deny": ["/etc", "/usr"]}

    def test_sequence_of_mappings(self):
        d = miniyaml_load(
            "steps:\n"
            "  - name: fetch\n"
            "    tool: email.inbox\n"
            "  - name: summarize\n"
            "    tool: rag.search\n"
        )
        assert d == {
            "steps": [
                {"name": "fetch", "tool": "email.inbox"},
                {"name": "summarize", "tool": "rag.search"},
            ]
        }

    def test_comment_stripped(self):
        d = miniyaml_load("a: 1  # trailing note\n")
        assert d == {"a": 1}

    def test_quoted_string(self):
        d = miniyaml_load('expr: "0 8 * * *"\n')
        assert d == {"expr": "0 8 * * *"}
