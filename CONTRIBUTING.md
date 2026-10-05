# Contributing to Frozen Fish

Frozen Fish is Apache-2.0 licensed and open to contributions of any scale:
bug fixes, new MCP servers, workflow YAMLs, docs, and (eventually) probe
implementations for non-CUDA accelerators.

## Ground rules

1. **Zero hard dependencies.** The runtime must stay pure standard library
   (Python ≥ 3.10). PyYAML stays optional — if you touch `config.py`, make sure
   the built-in `miniyaml_load` path still parses `examples/config.example.yaml`.
2. **Local-first, no silent egress.** Nothing new may phone home. Any outbound
   HTTP call goes through `security.EgressFilter`.
3. **Discover, don't launch.** Frozen Fish discovers backends via env vars and
   config; it never spawns Ollama / vLLM / LM Studio.
4. **Test what you touch.** Run the suite before opening a PR:
   `pip install -e ".[dev]" && pytest -q`.
5. Keep the config schema versioned (`version: "0.9"`). Adding keys is fine;
   changing the meaning of an existing key is a breaking change — say so in the
   PR and bump the version in `pyproject.toml` + `CHANGELOG.md`.

## Setting up

```bash
git clone https://github.com/FrozenFishe/frozenfish.git
cd frozenfish
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # adds pytest; PyYAML optional via ".[yaml]"
pytest -q
ff --help
```

## Adding an MCP server

Subclass `mcp.McpServer`, implement `name`, `tools()` (list of tool dicts with
`name` / `description` / `input_schema`), and `call(tool, arguments) -> dict`.
Built-ins are registered in `McpRegistry.__init__`; the sandbox applies to all
of them automatically. Keep `ADVERTISED_NAMES` in sync if the server is meant
to ship in the default registry.

## Adding a workflow

Drop a YAML in `workflows/` following the shape in `docs/workflows.md` and
deploy it with `ff workflow deploy --path workflows/<name>.yaml`.

## PR checklist

- [ ] `pytest -q` passes
- [ ] no new third-party runtime imports (grep: `import ` / `from `)
- [ ] new keys documented in `docs/config.md`
- [ ] `CHANGELOG.md` entry under `Unreleased`
- [ ] no secrets, no telemetry, no default egress

## Good first issues

- `rocm` / `metal` utilization probes (currently report 0)
- native `postgres` and `browser` runtimes (currently optional-runtime stubs)
- OTLP metrics export from the dashboard
- CI workflow (`.github/workflows/ci.yml`)
