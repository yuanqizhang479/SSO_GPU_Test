#!/usr/bin/env python3
"""Fetch the two public Xie repositories at audited commits; installs nothing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    "Spectral-Sphere-Optimizer": (
        "https://github.com/Unakar/Spectral-Sphere-Optimizer.git",
        "304d7a4f67c2221cda04b891831db94e5c049092",
    ),
    "Megatron-LM": (
        "https://github.com/Unakar/Megatron-LM.git",
        "00a07da9f684d6d0acf661a9cb8eee0c3b0a5aac",
    ),
}


def run(*cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("vendor"))
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    report = {"status": "source_checkout_only_not_GPU_validation", "repositories": {}}
    for name, (url, sha) in PINS.items():
        target = root / name
        if target.exists():
            if not (target / ".git").is_dir():
                raise SystemExit(f"Refusing to overwrite non-git directory: {target}")
            head = run("git", "rev-parse", "HEAD", cwd=target)
            dirty = run("git", "status", "--porcelain", cwd=target)
            if head != sha or dirty:
                raise SystemExit(f"Existing checkout differs from pinned clean source: {target}; use a new --root")
        else:
            target.mkdir()
            run("git", "init", str(target))
            run("git", "remote", "add", "origin", url, cwd=target)
            run("git", "fetch", "--depth", "1", "origin", sha, cwd=target)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
        head = run("git", "rev-parse", "HEAD", cwd=target)
        if head != sha:
            raise SystemExit(f"Commit verification failed: {name}")
        report["repositories"][name] = {"url": url, "commit": head, "path": str(target)}
        print(f"Verified {name}: {head}")
    for filename in ["pyproject.toml", "README.md"]:
        f = root / "Megatron-LM" / filename
        report.setdefault("megatron_metadata", {})[filename] = hashlib.sha256(f.read_bytes()).hexdigest()
    (root / "checkout_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print("No dependencies installed. Configure CUDA/PyTorch/TransformerEngine before launch.")


if __name__ == "__main__":
    main()
