# Layered Experiment Suite for the SSO Math Research (2026-09-19)

This is independently implemented small-scale experiment code serving the research question "do the feasible direction, the certificate, and the actual training direction agree?" **It is not a reproduction of Xie et al.'s Megatron/1.8B/100B-token experiments, and it does not presuppose that our repair improves model quality.** The manuscript's mathematical theorems do not depend on GPU training; claims of better training quality, failure frequency in real training, or actual speedup do require real GPU experiments.

Priority order: E0 (correctness and precision) → TinyStories (pipeline check) → scaled-down OLMo-Mix experiment (same data source) → FineWeb-Edu (external distribution). It is not advisable to spend resources training all of Xie's models one-to-one before these checks. The checkpoints, data indices, and scoring protocols missing for a strict reproduction of the original paper are covered in the companion XIE_PROTOCOL_REVIEW.md.

## One-to-one mapping of files and data

| Script / config | Data | Download site / purpose |
|---|---|---|
| `oracle_bench.py` | Random matrices + explicit 2×2 counterexamples; no external data | E0; CPU/CUDA precision, residual, solve time. Cannot estimate training failure frequency. |
| `test_smoke.py` | Local synthetic byte-token documents | Engineering check; not a TinyStories experiment. |
| `configs/data_tinystories.json` + `lm_tinystories.json` | The train source of `roneneldan/TinyStories`, re-split into train/val/test by document hash | https://huggingface.co/datasets/roneneldan/TinyStories — synthetic stories, suitable for pipelines and small models; not representative of general text. |
| `configs/data_olmo_mix.json` + `lm_olmo_mix.json` | Seven sources of `allenai/olmo-mix-1124` | https://huggingface.co/datasets/allenai/olmo-mix-1124 — the dataset collection used in the original paper; this suite re-extracts and re-splits, so it cannot be called an exact reproduction of the original data. |
| `configs/data_fineweb_edu.json` + `lm_fineweb_edu.json` | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — educational-quality filtered web text, as an independent extension; not Xie's original data. |
| Tokenizer for all three real datasets | Only the tokenizer of `allenai/OLMo-2-1124-7B` is downloaded, not the 7B weights | https://huggingface.co/allenai/OLMo-2-1124-7B — same tokenizer source as the original paper; version automatically resolved to an immutable commit. |

The datasets have their own licenses: CDLA-Sharing for TinyStories, ODC-By plus source conditions for OLMo-Mix, ODC-By per the FineWeb-Edu data card, etc. Please keep sources and data cards; this code package no longer distributes real token corpora or tokenizers; actual download verification keeps only manifests and run logs.

## 1. Install and check first

Recommended: Linux, Python 3.11/3.12. First choose the CUDA wheel matching the server driver per the official PyTorch install page, then install the remaining dependencies. No CUDA version that has not been verified on this machine is pinned here.

```bash
python -m venv .venv
source .venv/bin/activate
# First install torch with support for this machine's CUDA per https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
# If the server uses a SOCKS proxy and complains about missing socksio: python -m pip install 'httpx[socks]'
python -m pip freeze > environment.lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python test_smoke.py --out results/local_cpu_check
python oracle_bench.py --device cuda --out results/e0_cuda.json
```

The delivery environment has CPU only; the engineering checks and E0 that were actually run are in `results/`, with `gpu_executed=false`. In addition, 19 documents were successfully downloaded from real TinyStories and processed with the OLMo-2 tokenizer, and a 4-step CPU training of a tiny decoder was completed; this only validates the real-data pipeline. CUDA, FineWeb-Edu / OLMo seven-source downloads, and formal training were not validated here; see `VALIDATION_STATUS.json` for details. Code runnability checks must not be written up as completed GPU results.

## 2. Download and pin data

```bash
python prepare_data.py --config configs/data_tinystories.json --out data/tinystories --cache hf_cache
python prepare_data.py --config configs/data_olmo_mix.json --out data/olmo_mix --cache hf_cache
python prepare_data.py --config configs/data_fineweb_edu.json --out data/fineweb_edu --cache hf_cache
```

Only prepare the one you are working on now. Defaults are about 32M train tokens, 500k val, and 500k test each; all use `<u4`, the raw token files are about 132 MB, and the HF cache and per-document manifests take additional space. Sources may download/scan more than the retained token count; this is especially true for the seven OLMo sources, so do not treat this number as an upper bound on network traffic.

`prepare_data.py` first resolves the commit for each repository and tokenizer, and saves `manifest.json`, tokenizer file SHA256s, token file SHA256s, and for each accepted document its source ID, traversal position, normalized-text SHA256, and token start/end. The SHA256 of the normalized text determines train/val/test, and exact duplicates are removed across sources, so the three splits do not overlap; semantic near-duplicate removal is not done. The official TinyStories validation set is not used; the local re-split differs from the original.

Each source uses a buffered shuffle with a fixed seed and is then traversed up to its quota; **uniform random sampling over the whole dataset is not guaranteed**. The seven OLMo sources are allocated by the data card's token scale 3700:20.8:58.6:83:11.8:12.2:3.66, and training windows are sampled by those weights; it does not simply take the beginning of default (which could be almost entirely arXiv), nor is it guaranteed to equal Xie's unpublished exact 100B mixture, cleaning, indexing, and order. Training uses window sampling with replacement; `tokens_seen` is the number of exposed tokens and may include repeats.

To rebuild from an already downloaded version, lock the revision with the old manifest:

```bash
python prepare_data.py --replay-manifest data/tinystories/manifest.json --out data/tinystories_replay --cache hf_cache
```

The same Python dependency versions are required. Do not modify prepared files; training startup verifies SHA256. A failed half-finished directory should be renamed and kept as a failure record, then retried in a new directory.

## 3. A trial run on one GPU

```bash
python train.py --config configs/lm_tinystories.json --data data/tinystories --optimizer sso_ns --seed 11 --device cuda --out runs/tiny_sso11 --stop-after 20
python train.py --config configs/lm_tinystories.json --data data/tinystories --optimizer sso_ns --seed 11 --device cuda --out runs/tiny_sso11 --resume
```

Default small decoder: 4 layers, width 256, 4 heads, length 256, microbatch 4, accumulation 8, 1000 steps, i.e. 8.192M exposed training tokens. Word embeddings are tied with the output head; the parameter count is determined by the tokenizer and the actual value is logged. It is not OLMo or the original Dense architecture, so loss values in the paper cannot be compared directly. Adjusting batch/accum while keeping their product fixed controls the effective batch; any change requires a new experiment with the config recorded. Resume requires config, source code, data, seed, optimizer, and CPU/CUDA type to be unchanged.

All optimizers under the same seed have the same parameter initialization, data RNG, and budget. Hidden matrices all start from the same PI-normalized initialization; embedding, position, LayerNorm, etc. are all updated by AdamW. The default is matrix WD=0 and auxiliary-parameter WD=0.1; this is a fixed controlled choice and does not claim to reproduce the original paper's optimal hyperparameters. The forward pass defaults to BF16, weights and optimizer state to FP32, and NS to FP32; use `--ns-dtype bfloat16` to explicitly run the NS precision ablation, and do not mix the two into one set of results.

## 4. Method naming, and which ones are not efficient optimizers

| `--optimizer` | Exact meaning in this suite |
|---|---|
| `adamw` | AdamW for both hidden matrices and auxiliary parameters; the two groups can have different lr. |
| `muon` | EMA/Nesterov + 8-step Polar-Express + `sqrt(nout/nin)` update scaling; an independent controlled Muon-style implementation, not a bit-exact copy of every published Muon version. |
| `muon_sphere` | Same as above, with PI normalization to the target radius before the update; no tangent root-finding. |
| `sso_ns` | Same as above, plus tangent root-finding with finite NS; keeps the standalone default of 10 bracket expansions / fall back to the old λ on failure, and counts failures explicitly. Not an exact polar oracle. |
| `sso_ns_repair_svd` | Applies T6 feasibility repair to the finite NS output using the true normal + SVD norm; **an expensive diagnostic control, not a proposed scalable new optimizer**. The default tuning grid does not run it. |

Sphere methods use **pre-update retraction**; therefore the weights after the update may deviate from the target radius and are normalized again at the next step. The normal, NS direction, and diagnostics are all computed at the same pre-update weight state. PI is FP32, 100 steps by default, with an all-ones initial vector; NS uses the original published coefficients, root-finding default tolerance is 1e-4, and fallback/residual_failed are recorded. SVD repair only handles direction feasibility; it does not guarantee better training and cannot improve the error of the PI normalization itself. The "true normal" from floating-point SVD is only a numerical reference; near repeated singular values the log flags non-uniqueness, and no rigorous interval certificate is given.

## 5. Fair tuning and formal three-seed experiments

One must not tune more parameters only for the repair method and then draw conclusions against Muon with a fixed lr. The script uniformly uses 3 matrix lrs per method, fixed auxiliary lr, pilot seed 7, 200 steps; the lowest val loss is selected. The formal runs use independent paired seeds 11/22/33, 1000 steps, and a single final test. The range is a pre-declared small grid; it cannot be claimed that each method's global best parameters were found.

```bash
python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda
python run_grid.py --phase select --dataset olmo_mix --data data/olmo_mix --out studies/olmo
python run_grid.py --phase main --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda
python summarize.py --root studies/olmo/main --out studies/olmo/summary.json
```

For TinyStories/FineWeb-Edu, just change `--dataset` to `tinystories`/`fineweb_edu` respectively and change data/out. Use `--dry-run` first to view the commands about to run. With 2 GPUs, it is recommended to split the experiments rather than use DDP for a small model; the two terminals correspond to:

```bash
CUDA_VISIBLE_DEVICES=0 python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=1 python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda --shard 1 --num-shards 2
```

After both pilots finish, run `select` separately; then change `--phase pilot` in the two commands above to `--phase main`. Do not launch the same training job in the same output directory concurrently. The grid script shards by job, and checkpoints resume automatically.

## 6. Quantities to save and report

- `metrics.jsonl`: training loss, fixed validation loss/PPL, tokens, per-step time, gradient norm, solver status. PPL is only comparable under the same tokenizer, data, and evaluation rules.
- `diagnostics.jsonl`: residuals of the **actual finite NS/repaired direction** against the estimated normal and the SVD reference normal; separately lists the residuals after substituting the compact polar (one column each for the same estimated problem / the true-normal problem), which must not be passed off as residuals of the actual direction.
- Also saved: direction spectral norm, PI normal error, top gap before the weight update, radius error after the update, normal component of the actual displacement, and the T6 repaired primal-dual gap for the true-normal problem. This gap is a numerical upper bound after repairing a given candidate direction; it is not a rigorous floating-point certification, nor the decrease in the actual training objective.
- `checkpoint.pt`: model, matrix/auxiliary optimizers, Python/NumPy/Torch/CUDA RNG, training sampler RNG and position statistics, step, config, and data/code hashes. Only load checkpoints you trust (they contain Python-serialized state).
- `sessions.jsonl` and `result.json`: wall clock and end status for each resume session, peak allocated memory. Unfinished sessions do not get fabricated durations; sessions not closed before a restart cannot be traced precisely. Detailed cumulative checkpoint counts may miss the last write; for total time across restarts, the complete session wall time is authoritative.
- `summarize.py` groups by full config / data manifest / CPU or CUDA; it does not average across data or precisions. It gives the seed mean and sample standard deviation and does not automatically claim significance. Throughput excludes warmup and the entire SVD diagnostic step; when comparing actual total training time, those steps and validation/saving costs must be included, and the two accounting methods reported separately.

Validation windows are fixed and drawn with replacement by source weights; all methods see the same val/test windows. Three seeds can only give a preliminary picture of variability and cannot prove general superiority. When comparing the final test, use paired seed differences and report all individual results; do not keep picking lr or changing methods based on test.

## Sources and implementation boundaries

- Xie et al.: Controlled LLM Training on Spectral Sphere, https://arxiv.org/abs/2601.08393
- Original SSO standalone: https://github.com/Unakar/Spectral-Sphere-Optimizer , verified commit `304d7a4f67c2221cda04b891831db94e5c049092`, `sso.py` SHA256 `1b7416bff73e70bdb923c2f6744127682d826cf0283528605dc7a7f2708545d1`. This package does not pose as its production Megatron implementation; only the mathematical structure, coefficients, and default fallback behavior were verified.
- HF streaming API: https://huggingface.co/docs/datasets/en/stream
- PyTorch Muon docs: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html (explains different scaling conventions; this implementation states its own convention explicitly).

The original paper's 8 downstream QA tasks and original scoring scripts are handled separately by the companion protocol. This suite does not provide an unvalidated lm-eval interface and does not substitute training loss for those 8 accuracies.
