"""Daemon entry point: ``python -m frozenfish._daemon --base . --config ff.yaml``."""

from __future__ import annotations

import sys


def run(argv: list[str]) -> int:
    from .cli import _daemon_main

    return _daemon_main(argv)


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
