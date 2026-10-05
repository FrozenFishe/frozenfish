"""GPU / inference backend registry.

Tracks every inference backend the router can place work on: Ollama, vLLM,
llama.cpp, LM Studio, HuggingFace Transformers, and any OpenAI-compatible
endpoint.  Each backend reports GPU utilization, warm models and queue depth.

On machines without a GPU, backends run as CPU entries; the scoring router
degrades gracefully (util stays 0.0, warm dominates).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any

VALID_ACCELERATORS = ["cuda", "rocm", "metal", "vulkan", "openvino", "cpu"]


class Backend:
    def __init__(
        self,
        name: str,
        kind: str = "ollama",
        endpoint: str = "http://127.0.0.1:11434",
        accelerator: str = "cpu",
        devices: int = 1,
        memory_gb: float = 0.0,
    ):
        self.name = name
        self.kind = kind                # ollama | vllm | llama.cpp | lmstudio | transformers | openai
        self.endpoint = endpoint
        self.accelerator = accelerator  # one of VALID_ACCELERATORS
        self.devices = devices
        self.memory_gb = memory_gb
        # live state, mutated by probes and the router
        self.utilization: float = 0.0   # 0.0..1.0 aggregate GPU busy
        self.warm_models: dict[str, float] = {}  # model -> last_warm_ts
        self.queue_depth: int = 0
        self.in_flight: int = 0
        self.last_probe: float = 0.0
        self.healthy: bool = True
        self._lock = threading.Lock()

    def mark_warm(self, model: str) -> None:
        with self._lock:
            self.warm_models[model] = time.time()

    def is_warm(self, model: str) -> bool:
        with self._lock:
            return model in self.warm_models

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "kind": self.kind,
                "endpoint": self.endpoint,
                "accelerator": self.accelerator,
                "devices": self.devices,
                "memory_gb": self.memory_gb,
                "utilization": round(self.utilization, 3),
                "warm_models": sorted(self.warm_models),
                "queue_depth": self.queue_depth,
                "in_flight": self.in_flight,
                "healthy": self.healthy,
                "last_probe": self.last_probe,
            }


class GpuRegistry:
    """Holds known backends; probes them for utilization/warmth."""

    def __init__(self):
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()

    # -- management -----------------------------------------------------------
    def add(self, backend: Backend) -> Backend:
        with self._lock:
            self._backends[backend.name] = backend
            return backend

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._backends.pop(name, None) is not None

    def get(self, name: str) -> Backend:
        b = self._backends.get(name)
        if b is None:
            raise KeyError(f"unknown backend: {name!r}")
        return b

    def names(self) -> list[str]:
        return sorted(self._backends)

    def all(self) -> list[Backend]:
        return list(self._backends.values())

    # -- probing ----------------------------------------------------------------
    def probe(self, name: str) -> dict[str, Any]:
        """Probe one backend: HTTP liveness + GPU utilization if available."""
        b = self.get(name)
        b.last_probe = time.time()
        b.healthy = self._probe_liveness(b)
        b.utilization = self._probe_utilization(b)
        return b.snapshot()

    def probe_all(self) -> list[dict[str, Any]]:
        return [self.probe(n) for n in list(self._backends)]

    @staticmethod
    def _probe_liveness(b: Backend) -> bool:
        import urllib.request

        url = b.endpoint.rstrip("/") + "/api/tags" if b.kind == "ollama" else \
            b.endpoint.rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _probe_utilization(b: Backend) -> float:
        """Try nvidia-smi, fall back to a stable 0.0 (CPU machines)."""
        if b.accelerator == "cuda" and shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3.0,
                )
                vals = [float(x) for x in out.stdout.split() if x.strip().isdigit()]
                if vals:
                    return sum(vals) / len(vals) / 100.0
            except (ValueError, subprocess.SubprocessError):
                pass
        return 0.0

    # -- discovery ----------------------------------------------------------------
    def discover(self, base_dir: str = ".") -> list[Backend]:
        """Best-effort discovery of local backends from the process environment.

        Honors: OLLAMA_HOST, VLLM_HOST, LMSTUDIO_PORT, LOCALHOST_LLAMA (llama.cpp
        server on 8080).  Never shells out to launch anything - discovery only
        records what is already listening.
        """
        found: list[Backend] = []
        ollama = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        found.append(Backend("ollama-local", "ollama", ollama,
                             accelerator="cuda" if shutil.which("nvidia-smi") else "cpu"))
        vllm = os.environ.get("VLLM_HOST")
        if vllm:
            found.append(Backend("vllm-local", "vllm", vllm, accelerator="cuda"))
        lm = os.environ.get("LMSTUDIO_PORT")
        if lm:
            found.append(Backend("lmstudio-local", "lmstudio",
                                 f"http://127.0.0.1:{lm}/v1", accelerator="cpu"))
        for b in found:
            self.add(b)
        return found


def backend_from_config(cfg) -> GpuRegistry:
    """Build the registry from config's ``orchestrator.backends`` list."""
    reg = GpuRegistry()
    for spec in cfg.get("orchestrator.backends", []) or []:
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        reg.add(Backend(
            name=spec["name"],
            kind=spec.get("kind", "openai"),
            endpoint=spec.get("endpoint", "http://127.0.0.1:8000"),
            accelerator=spec.get("accelerator", "cpu"),
            devices=int(spec.get("devices", 1)),
            memory_gb=float(spec.get("memory_gb", 0.0)),
        ))
    return reg
