"""frozenfish - local-first AI workflow orchestration layer for the MCP ecosystem.

Subpackage map
--------------
config     YAML config schema (v0.9), stdlib-only mini-YAML loader, validation.
security   RBAC, audit event model, PII redaction, egress allowlist.
audit      SQLite-backed audit log (JSONL mirror optional).
mcp        MCP registry: in-process servers, stdio JSON-RPC transport, sandbox.
gpu        GPU/backend registry with utilization, warm-model and queue state.
router     Scoring router (util / warm / queue), sticky sessions, policies.
context    CAG cache, CAR vector partitions, RAG fallback, hybrid context router.
workflow   Declarative workflow engine: triggers, per-step retry, dead-letter.
orchestrator  Service supervisor tying router + context + workflows together.
dashboard  Zero-dependency HTTP dashboard (stdlib http.server).
cli        The `ff` command line interface.
"""

__version__ = "0.9.4"
__all__ = ["__version__"]
