#!/usr/bin/env python3
"""Backward-compatible entry point for the unified crossover checks."""

# Scientific assertions must never disappear under python -O / PYTHONOPTIMIZE.
if not __debug__:
    raise RuntimeError("Verification requires assertions: rerun without -O / PYTHONOPTIMIZE.")

from verify_unified_crossover import main


if __name__ == "__main__":
    main()
