#!/usr/bin/env python3
"""Use the pinned official preprocessor on user-selected JSONL (not original Xie indices)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PIN = "00a07da9f684d6d0acf661a9cb8eee0c3b0a5aac"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("vendor/Megatron-LM"))
    p.add_argument("--input-jsonl", type=Path, required=True, help='Each line must contain a string "text"')
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--tokenizer-id", default="allenai/OLMo-2-1124-7B")
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    args.repo = args.repo.resolve(); args.input_jsonl = args.input_jsonl.resolve()
    args.output_prefix = args.output_prefix.resolve(); args.tokenizer_dir = args.tokenizer_dir.resolve()
    script = args.repo / "tools/preprocess_data.py"
    if not script.is_file() or not args.input_jsonl.is_file():
        raise SystemExit("Missing official preprocessor or input JSONL; run bootstrap and provide selected documents")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip()
    if head != PIN or dirty:
        raise SystemExit("Preprocessing requires the exact clean pinned Megatron checkout; run bootstrap with a new root")
    generated_prefix = str(args.output_prefix) + "_text_document"
    if any(Path(generated_prefix + ext).exists() for ext in (".bin", ".idx")):
        raise SystemExit("Output indexed data already exists; select a new --output-prefix")
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = HfApi().model_info(args.tokenizer_id, revision=args.tokenizer_revision).sha
    if args.tokenizer_dir.exists() and any(args.tokenizer_dir.iterdir()):
        raise SystemExit("Choose a new/empty tokenizer directory to avoid mixing tokenizer revisions")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, revision=revision, trust_remote_code=False)
    tokenizer.save_pretrained(args.tokenizer_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open() as f:
        n = 0
        for n, line in enumerate(f, 1):
            item = json.loads(line)
            if not isinstance(item.get("text"), str):
                raise SystemExit(f"Missing string text at JSONL line {n}")
    if n == 0:
        raise SystemExit("Input JSONL is empty")
    env = os.environ.copy(); env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(script), "--input", str(args.input_jsonl),
           "--output-prefix", str(args.output_prefix), "--tokenizer-type", "HuggingFaceTokenizer",
           "--tokenizer-model", str(args.tokenizer_dir), "--json-keys", "text", "--append-eod",
           "--workers", str(args.workers)]
    subprocess.run(cmd, cwd=args.repo, env=env, check=True)
    for ext in (".bin", ".idx"):
        if not Path(generated_prefix + ext).is_file():
            raise SystemExit(f"Preprocessor did not produce expected {generated_prefix + ext}")
    manifest = args.output_prefix.parent / (args.output_prefix.name + "_manifest.json")
    bin_sha = file_sha256(Path(generated_prefix + ".bin"))
    idx_sha = file_sha256(Path(generated_prefix + ".idx"))
    manifest.write_text(json.dumps([{"weight": 1, "prefix": generated_prefix,
                                    "bin_sha256": bin_sha, "idx_sha256": idx_sha}], indent=2) + "\n")
    metadata = {"tokenizer_id": args.tokenizer_id, "tokenizer_revision": revision,
                "input": str(args.input_jsonl), "argv": cmd, "exact_xie_indices": False,
                "sample_provenance": "new_user_selected_documents_not_the_original_Xie_sample",
                "code_commit": head, "documents": n, "input_sha256": file_sha256(args.input_jsonl),
                "bin_sha256": bin_sha, "idx_sha256": idx_sha,
                "tokenizer_sha256": file_sha256(args.tokenizer_dir / "tokenizer.json")}
    manifest.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Created data manifest: {manifest}")


if __name__ == "__main__":
    main()
