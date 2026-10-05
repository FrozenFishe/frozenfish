"""Shared fixtures for the frozenfish test suite."""

from __future__ import annotations

import pytest

from frozenfish.gpu import Backend, GpuRegistry


@pytest.fixture
def registry() -> GpuRegistry:
    reg = GpuRegistry()
    a = reg.add(Backend("gpu-a", accelerator="cuda"))
    a.utilization = 0.0
    b = reg.add(Backend("gpu-b", accelerator="cuda"))
    b.utilization = 0.9
    return reg
