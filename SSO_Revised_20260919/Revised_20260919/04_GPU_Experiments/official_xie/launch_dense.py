#!/usr/bin/env python3
"""Launch audited public Dense SSO source with explicit local data and provenance.

This is an independent run of the public code, not certification of Xie Table 3.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

PIN = "00a07da9f684d6d0acf661a9cb8eee0c3b0a5aac"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data(path):
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("data-manifest must be a nonempty list of {weight, prefix}")
    tokens = []
    normalized = []
    for item in data:
        weight = float(item["weight"])
        if not 0 < weight < float("inf"):
            raise ValueError("Dataset weights must be finite and positive")
        prefix = Path(item["prefix"]).expanduser()
        if not prefix.is_absolute():
            prefix = (path.parent / prefix).resolve()
        for extension in (".bin", ".idx"):
            f = Path(str(prefix) + extension)
            if not f.is_file() or f.stat().st_size == 0:
                raise ValueError(f"Required nonempty Megatron indexed-data file missing: {f}")
        print(f"Hashing indexed dataset (streamed): {prefix}", flush=True)
        bin_sha = file_sha256(Path(str(prefix) + ".bin"))
        idx_sha = file_sha256(Path(str(prefix) + ".idx"))
        for key, actual in (("bin_sha256", bin_sha), ("idx_sha256", idx_sha)):
            if key in item and item[key] != actual:
                raise ValueError(f"Manifest hash mismatch for {prefix}: {key}")
        tokens.extend([str(weight), str(prefix)])
        normalized.append({"weight": weight, "prefix": str(prefix),
                           "bin_bytes": Path(str(prefix) + ".bin").stat().st_size,
                           "bin_sha256": bin_sha, "idx_sha256": idx_sha})
    return tokens, normalized


def command(args, data_tokens):
    full_steps = args.tokens // (args.global_batch * args.seq_length)
    if full_steps <= 0:
        raise ValueError("Token budget must cover at least one global batch")
    steps = args.stop_after if args.stop_after is not None else full_steps
    if not 1 <= steps <= full_steps:
        raise ValueError("--stop-after must be between 1 and the full training budget")
    if args.global_batch % (args.gpus * args.micro_batch):
        raise ValueError("global-batch must be divisible by gpus * micro-batch (TP=PP=1)")
    ckpt = args.out / "checkpoints"
    flags = [
        "--tokenizer-model", str(args.tokenizer_dir), "--tokenizer-type", "HuggingFaceTokenizer",
        "--data-path", *data_tokens,
        "--data-cache-path", str(args.out / "data_cache"),
        "--split", "99,1,0", "--train-iters", str(steps),
        "--num-dataset-builder-threads", str(args.dataset_threads), "--num-workers", str(args.workers),
        "--no-mmap-bin-files", "--distributed-timeout-minutes", "60",
        "--lr", "5e-3", "--lr-warmup-iters", "500", "--lr-decay-style", "cosine",
        "--min-lr", "5e-4", "--lr-decay-iters", str(full_steps),
        "--adam-beta1", "0.9", "--adam-beta2", "0.95", "--adam-eps", "1e-8",
        "--clip-grad", "1.0", "--weight-decay", "0.1",
        "--optimizer", "spectral_ball_dist", "--spectral-ball-momentum", "0.9",
        "--spectral-ball-use-nesterov", "--spectral-ball-msign-steps", "8",
        "--spectral-ball-radius-mode", "spectral_mup", "--spectral-ball-scale-mode", "spectral_mup",
        "--spectral-ball-solver", "bisection", "--spectral-ball-solver-tolerance-f", "2e-4",
        "--spectral-ball-power-iteration-steps", "100", "--spectral-ball-solver-max-iterations", "20",
        "--spectral-ball-retract-mode", "hard", "--spectral-ball-qkv-split-mode", "head",
        "--num-layers", "28", "--hidden-size", "2048", "--ffn-hidden-size", "6144",
        "--group-query-attention", "--num-attention-heads", "16", "--num-query-groups", "8",
        "--norm-epsilon", "1e-6", "--kv-channels", "128",
        "--seq-length", str(args.seq_length), "--max-position-embeddings", "40960",
        "--attention-dropout", "0", "--hidden-dropout", "0", "--bf16",
        "--use-rotary-position-embeddings", "--rotary-base", "1000000", "--swiglu",
        "--untie-embeddings-and-output-weights", "--normalization", "RMSNorm", "--qk-layernorm",
        "--cross-entropy-loss-fusion", "--disable-bias-linear", "--transformer-impl", "transformer_engine",
        "--attention-backend", "fused", "--init-method-std", "0.02", "--split-qkv-init-mode", "head",
        "--spectral-mup-init", "--use-cpu-initialization",
        "--ckpt-format", "torch_dist", "--save-interval", str(args.save_interval), "--save", str(ckpt),
        "--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
        "--micro-batch-size", str(args.micro_batch), "--global-batch-size", str(args.global_batch),
        "--seed", str(args.seed), "--log-interval", "1", "--log-throughput", "--log-memory-to-tensorboard",
        "--log-validation-ppl-to-tensorboard", "--tensorboard-dir", str(args.out / "tensorboard"),
    ]
    if args.resume_from is not None:
        flags.extend(["--load", str(args.resume_from)])
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(args.gpus),
            str(args.repo / "pretrain_gpt.py"), *flags], full_steps


def check_environment(args, env):
    code = '''import json, importlib, importlib.metadata as im
import torch, numpy
import transformers
import transformer_engine.pytorch
import megatron.core
if int(numpy.__version__.split('.')[0]) >= 2:
    raise RuntimeError("Pinned Megatron pyproject requires numpy<2")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
if torch.cuda.device_count() < GPU_COUNT:
    raise RuntimeError("Not enough visible GPUs")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 unsupported")
print(json.dumps({"torch":torch.__version__, "cuda":torch.version.cuda,
 "numpy":numpy.__version__, "transformers":transformers.__version__,
 "transformer_engine":im.version("transformer_engine"),
 "gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
'''.replace("GPU_COUNT", str(args.gpus))
    result = subprocess.run([sys.executable, "-c", code], cwd=args.repo, env=env, text=True, capture_output=True)
    (args.out / "preflight.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError("GPU/dependency preflight failed. Nothing launched.\n" + result.stderr)
    return result.stdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path("vendor/Megatron-LM"))
    p.add_argument("--data-manifest", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=1024)
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=100_000_000_000)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dataset-threads", type=int, default=8)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--resume-from", type=Path, help="Megatron checkpoint root containing latest_checkpointed_iteration.txt")
    p.add_argument("--dry-run", action="store_true", help="validate inputs/build command; do not require GPU or execute")
    args = p.parse_args()
    try:
        if min(args.gpus, args.micro_batch, args.global_batch, args.seq_length, args.workers, args.dataset_threads) <= 0:
            raise ValueError("Resource/batch sizes must be positive")
        args.repo = args.repo.resolve(); args.out = args.out.resolve()
        args.tokenizer_dir = args.tokenizer_dir.resolve(); args.data_manifest = args.data_manifest.resolve()
        if args.resume_from is not None:
            args.resume_from = args.resume_from.resolve()
            if not (args.resume_from / "latest_checkpointed_iteration.txt").is_file():
                raise ValueError("--resume-from must contain latest_checkpointed_iteration.txt; verify checkpoint completeness")
        if not (args.repo / "pretrain_gpt.py").is_file():
            raise ValueError("Run bootstrap.py first; --repo must contain pretrain_gpt.py")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
        if sha != PIN:
            raise ValueError(f"Expected pinned commit {PIN}, found {sha}")
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip()
        if dirty:
            raise ValueError("Pinned public repository has changes; use a clean checkout for this reference launcher")
        if not (args.tokenizer_dir / "tokenizer.json").is_file():
            raise ValueError("tokenizer-dir must contain OLMo-2 tokenizer.json; see prepare_megatron.py / README")
        data_tokens, data = read_data(args.data_manifest)
        cmd, full_steps = command(args, data_tokens)
        env = os.environ.copy(); env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"; env["WANDB_MODE"] = "offline"
        env["PYTHONPATH"] = str(args.repo) + os.pathsep + env.get("PYTHONPATH", "")
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("--out must be a new or empty directory; refusing to mix runs")
        args.out.mkdir(parents=True, exist_ok=True)
        runtime = "not checked in dry run" if args.dry_run else check_environment(args, env)
        report = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "status": "dry_run_only" if args.dry_run else "launch_requested_not_completed",
                  "exact_xie_reproduction": False, "code_commit": sha, "argv": cmd,
                  "data_manifest": data, "full_schedule_steps": full_steps, "runtime": runtime,
                  "tokenizer_json_sha256": file_sha256(args.tokenizer_dir / "tokenizer.json"),
                  "protocol_changes": ["new explicit local data manifest; original Xie indices unavailable",
                       "explicit seed; hardware/rank count/microbatch as specified",
                       "checkpoint optimizer state retained; save frequency configurable",
                       "worker counts reduced; optional monitor flags omitted; benchmark evaluation disabled"]}
        manifest = args.out / "launch_manifest.json"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        print(shlex.join(cmd))
        if args.dry_run:
            print("DRY RUN: no GPU or training test performed."); return
        log_path = args.out / "training_stdout_stderr.log"
        report["stdout_stderr_log"] = str(log_path)
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        with log_path.open("w", buffering=1) as logfile:
            with subprocess.Popen(cmd, cwd=args.repo, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    logfile.write(line)
                returncode = process.wait()
        report["status"] = "process_exit0" if returncode == 0 else "failed"
        report["exit_code"] = returncode
        report["result_claim"] = "Process status only; inspect logs, checkpoints and metrics before scientific interpretation"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(returncode)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
