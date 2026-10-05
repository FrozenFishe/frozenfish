# Changelog

All notable changes to Frozen Fish are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); project follows [Semantic Versioning](https://semver.org/).

## [0.9.4] - 2025-09-14

### Added
- CAR (Context-Augmented Retrieval) partitioning — vector indexes aligned to MCP tool boundaries; `ff context car partition --by tool|topic`.
- Hybrid CAG→CAR→RAG context router with per-source confidence thresholds.
- Config schema v2 (`version: "0.9"`).
- Benchmark harness (`ff bench`) writing `docs/BENCHMARKS.md`; measured MuSiQue-style +5.6 F1 on the hybrid pipeline (reference hardware).

### Changed
- `ff context stats` now reports CAG/CAR/RAG hit counts and p50/p95/p99 per source.
- Landing page (`site/index.html`) now ships the project logo.

## [0.9.3] - 2025-09-02

### Added
- GPU router weights configurable: `routing.weights {util, warm, queue}`.
- Sticky sessions: `routing.sticky: true` pins a session to the backend it first hit.
- Router policies: `gpu_aware`, `round_robin`, `least_queue` (`ff router policy set|get`).

## [0.9.2] - 2025-08-21

### Added
- CAG cache warm-up on orchestrator boot.
- `ff context cag invalidate` API + HTTP endpoint `POST /api/context/cag/invalidate`.

## [0.9.1] - 2025-08-09

### Added
- MCP registry UI (dashboard `#mcp` panel) and custom server registration via `ff mcp register <name> --cmd ...`.
- Built-in MCP servers: filesystem, sqlite, http_client, shell, vector, markdown, git, email (local), slack (webhook), web_scraper.

## [0.9.0] - 2025-07-28

### Added
- First public release.
- MCP orchestration core (JSON-RPC 2.0 over stdio, dual client/server).
- GPU-aware inference router (util / warm / queue scoring).
- Workflow engine: declarative YAML, event/cron/interval triggers, per-step retry, dead-letter queue, SQLite state.
- RAG fallback with MMR re-ranking.
- Security: RBAC, audit log, egress allowlist, PII redaction, sandboxed tools.

## [0.8.x] - 2025-06 (closed beta)
- RAG + router prototypes.

## [0.7.x] - 2025-05 (internal alpha)
- Plugin system.

## [0.6.x] - 2025-04 (research prototype)
- No MCP; context research only.
