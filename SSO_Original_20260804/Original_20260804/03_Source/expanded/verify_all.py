#!/usr/bin/env python3
"""Run the complete deterministic verification suite."""

from pathlib import Path
import subprocess
import sys


SCRIPTS = (
    "verify_rate_constants_exact.py",
    "verify_unified_crossover.py",
    "verify_partial_isometry.py",
    "verify_direction_rates.py",
    "verify_theorems.py",
    "verify_corank2_strict_center.py",
    "verify_arbitrary_corank_strict_law.py",
    "verify_direction_rate_general_corank.py",
    "verify_intrinsic_constants.py",
    "verify_smoothing_family.py",
)


def main():
    root = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, str(root / script)], check=True)
    print("\nALL VERIFICATION SCRIPTS PASSED.")


if __name__ == "__main__":
    main()
