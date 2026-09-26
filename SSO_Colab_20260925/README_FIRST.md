# SSO / SIMAX: Colab Experiment Package for This Round

Compiled and tested: 2026-09-25. **First complete the four stages of the notebook, then send back `SSO_RETURN_RESULTS.zip`. Do not enable the optional training grids yet.**

## 1. Where the research stands now

The main line of this project is nuclear-norm minimization along a matrix line, certificate completion at rank-deficient points, and center selection and error rates along a specified smoothing path. It is a matrix-analysis study with SSO as background; the original SSO is the work of Xie et al.

What the handoff package actually recovered is the 9/19 manuscript and code. Some 9/13 conclusions exist only as historical summaries; the exact deliverables were not recovered. This round uses the actual files as the baseline and newly writes the missing experiment modules. It does not claim to have recovered the lost version, and it did not re-audit every proof in the whole paper.

The 9/19 GPU package had a substantive gap: it implemented only finite-step NS, not the main manuscript's pseudo-Huber path. Therefore, even if the old package had run all three datasets to completion, it could not validate the manuscript's δ exponents. This package adds a numerical reference implementation of that path and keeps finite NS as a separate experimental object.

| Object | Status this round | What it can show |
|---|---|---|
| Manuscript theory and the original 14 math checks | Historical evidence exists in the handoff package; not re-proven item by item / not fully re-run this round | GPU wins/losses cannot decide whether a theorem is true |
| pseudo-Huber path | Newly implemented; 36 CPU numerical checks pass | Verifies the implementation, the directional error rates on specified instances, and the transition regime |
| Six training methods, resume, full-matrix snapshots | Actually run on CPU and passing | Correctness of the engineering pipeline |
| TinyStories real-data micro-training and snapshot diagnostics | Actually completed this round | Data interface and real-tensor pipeline work; not a formal experiment for comparing effectiveness |
| FineWeb-Edu data interface | Small-sample download and tokenization passed this round | The specified repo and subset are accessible; does not mean training at default scale is done |
| CUDA, actual runtime, memory, formal training advantage | **Not yet executed** | To be obtained by you running this package on Colab |

Validation details and environment records for this round are in `validation/VALIDATION_REPORT.json`. CPU checks must not be labeled as completed GPU experiments.

## 2. What exactly you do

1. Open [Google Colab](https://colab.research.google.com/) and upload `SSO_Colab_Run.ipynb`.
2. Under "Runtime → Change runtime type", select GPU.
3. Run the notebook from top to bottom. Upload this round's `SSO_Colab_20260925.zip` and connect your own Google Drive when prompted, to save checkpoints and results.
4. Run through to the end of "Download results"; send back `SSO_RETURN_RESULTS.zip` as-is, not just screenshots of loss curves.

This time you do not need to download any corpus manually. The scripts download a small portion of text from Hugging Face, pin versions, and generate a data manifest; only the tokenizer is downloaded, not the 7B model.

If a T4 does not support the native BF16 this setup requires, the first config generation will explicitly choose an FP32 forward pass and skip the BF16 operator ablation. GPUs that support BF16 use a BF16 forward pass, FP32 weights/NS, and an FP64 SVD reference. The actual precision of each group is written to the config; training results must not be mixed across precisions. FP64 SVD may be slow on a T4, so the matrices this round are intentionally kept small.

## 3. Run order: how far you must run this time

| Order | Notebook / command stage | What actually runs | Purpose and next step |
|---|---|---|---|
| A | `preflight` | CPU vs. CUDA path comparison; 4 steps of synthetic-data training for each of the six methods; resume and data-integrity checks | On failure, stop later stages and send back logs |
| B | `tiny` | TinyStories: SSO-NS runs to step 15, then resumes from checkpoint to step 30 | Checks real data and the resume flow; does not judge performance |
| C | `capture`, default FineWeb-Edu | Two pre-specified trajectories: MuonSphere and SSO-NS, each seed 11, 100 steps | Collects full matrices from actual training; no winner selection |
| D | `audit`, same FineWeb-Edu | On the same inputs saved by C, compares actual direction, pseudo-Huber, NS steps/precision, normal error, root-finding settings | Determines whether the applied problem is real, where it comes from, and whether more training is worthwhile |
| E | Download results | Automatically collects config, environment, logs, metrics, snapshots, diagnostics | **Stop here this round and send it back to me** |

Model for C: 2 layers, width 64, 4 heads, sequence 128, microbatch 2, gradient accumulation 2. Each trajectory has 100×2×2×128 = 51,200 training target tokens of exposure, with repeated sampling allowed. This budget is for preliminary mechanism diagnostics and is not enough to establish conclusions about language-model training performance.

At steps 1/50/100 of each trajectory, layer 0's Q, FFN expansion matrix, and FFN contraction matrix are saved, of sizes 64×64, 256×64, and 64×256 respectively. The two trajectories together give **18 full-matrix snapshots**. This is a small, correlated, fixed selection, not a randomly sampled estimate of "failure rate", and does not represent other layers, later training, or large models.

Default FineWeb-Edu keeps about 2,097,152 train / 65,536 val / 65,536 test tokens; TinyStories about 262,144 / 16,384 / 16,384. Keeping whole documents slightly exceeds the quota. The raw uint32 token files are about 8.9 MB and 1.2 MB respectively; caches, scan traffic, and model checkpoints are extra, so do not treat these numbers as total download volume or total disk usage.

**No untested "finishes in a few minutes" forecast is given.** First look at the actual runtime and memory of A/B, then decide whether to keep C/D on the current GPU. Each snapshot in D is saved separately upon completion; on restart, snapshots already completed under the same code and environment can be skipped.

## 4. What each piece of code is responsible for

| File | Input → Output | Scientific/engineering role |
|---|---|---|
| `SSO_Colab_Run.ipynb` | Upload this package → run A–E | User entry point; does not auto-run large grids |
| `run_stage.py` | Stage, work directory → logs, status, results ZIP | Freezes config, resumes, records errors, packages the return |
| `prepare_data.py` | Data config → train/val/test token files and manifest | Document-hash splitting, exact-duplicate removal, data and tokenizer version/hash records |
| `train.py` | Fixed config, corpus, method, seed → loss, checkpoint, full snapshots | Unified initialization and data sampling; actual updates and resume |
| `optimizers.py` | Gradient, weights → NS / repair / PH updates | Keeps finite NS; fixes the original diagnostic's mixing of estimated/reference normals |
| `oracles.py` | G, Θ, fixed δ → λδ, Zδ, numerically feasible repair and objective bounds | Same smoothing function as the manuscript definition; SVD is a numerical reference, not a fast implementation |
| `theory_path.py` | Specified analytic matrices → error rates, independent reference differences, transition-regime records | regular/strict/boundary, crossover, degenerate counterexample, rectangular and transpose checks |
| `audit_snapshots.py` | Saved full W/G/Θ/Z → per-snapshot diagnostics and summary | Paired comparisons on the same inputs to identify error sources; no artificial truncation to manufacture rank deficiency |
| `test_smoke.py` | Local synthetic byte-token fixture → engineering check report | Forward/backward for six methods, resume parameter consistency, rejection of corrupted data and changed-config resume |
| `study.py` | Pre-fixed plan → pilot, frozen hyperparameter selection, paired main experiment, summary | **Conditionally run** fair training comparison |
| `summarize.py` | Individual training directories → grouped summary | Retained basic summary tool; for the formal optional study, `study.py --phase summarize` is authoritative |

The original `official_xie/` Megatron package is not included as an execution entry point this round. It targets a different set of dependencies and a large-model protocol, and it also lacks reproduction prerequisites such as exact data indices. The old package's `oracle_bench.py` and `run_grid.py` are replaced by this round's path checks and staged scripts targeting the current question; you do not need to run both sets. The complete originals remain in the handoff ZIP you uploaded.

## 5. How the mathematical claims map to experiments

Fix

\[
A(\lambda)=G+\lambda\Theta,\quad \phi(\lambda)=\|A(\lambda)\|_*.
\]

The path newly added this round is defined exactly as

\[
Z_\delta(A)=U\operatorname{diag}\!\left(\frac{\sigma_i}{\sqrt{\sigma_i^2+\delta^2}}\right)V^T,
\quad \langle\Theta,Z_\delta(G+\lambda_\delta\Theta)\rangle=0.
\]

δ is fixed during each root-find; δ is not changed with λ and A is not re-normalized. G in the real snapshots is **the Nesterov/EMA direction formed from clipped gradients, then Frobenius-normalized**, not the unprocessed raw gradient. Raw gradients and momentum are also saved for re-checking.

| Paper/application question | Code and setting | Observables | Conclusions it can support |
|---|---|---|---|
| Local directional error along the specified path | `theory_path.py`; fixed matrices, shrinking δ | log–log slopes of three examples; independent NumPy/Brent root comparison | These instances and the code match expectations; does not replace the general theorem |
| strict → boundary transition | Explicit 2×2 family; Δ = ζδ^(2/3) | Scaled displacement and positive root of the limiting cubic | Transition test for the specified family; not a theorem for arbitrary high corank or spectral-gap collapse |
| Whether the original tangency condition holds | Snapshot `actual_saved_update` | `tangency`, `spectral_norm` | Numerical feasibility for the current full matrices and current actual direction |
| NS polynomial error | Same G/Θ, k = 5/8/12; FP32/BF16 | Residual, spectral norm, objective bound, distance to finite-δ reference | Precision sensitivity of finite NS; k must not be treated as δ |
| Normal estimation error | PI normal vs. FP64 SVD reference normal | top gap, normal error, residual after switching to the reference normal | Helps separate the effects of PI and NS; interpretation must be qualified when top singular values coincide |
| Effect of root-finding settings | 1e−4 / 1e−6 tolerance; 10 / 20 expansions | Solver status, residual, failure fallback | Whether some failures stem from numerical settings; does not imply global monotonicity |
| Whether the objective is near optimal | Candidate λ and repaired direction under the same true-normal problem | Upper bound minus lower bound | Numerical objective suboptimality bound; not a distance bound to the unique certificate |
| Whether repair improves training | Conditionally triggered `study.py --plan repair` | Paired test NLL, all seeds, failures, total time/memory | Supports results only under this small-scale controlled protocol |

The directional error orders 2, 1, 2/3 are the expected orders for three chosen non-degenerate instances; not every matrix must have the same nonzero leading term. The numerical rank threshold only records the "number of small singular values" and must not be called a strict rank-deficiency probability. The finite-δ PH reference must not be passed off as the exact δ→0 center either.

`repaired_gap` has the form \(\|G+\lambda\Theta\|_*-\langle G,\widehat Z\rangle\), where \(\widehat Z\) satisfies the tangency/spectral constraints (in the sense of exact arithmetic) via projection and scaling. Ordinary FP64 SVD and floating-point residuals still do not constitute a rigorous interval-arithmetic certificate; small negative gaps are kept in the results and are not silently truncated to zero.

## 6. How different results would change the paper

| Observed result | Judgment | Paper and next step |
|---|---|---|
| A's PH path or resume check fails | Later results cannot yet be interpreted | First check code, precision, root-finding, and instance assumptions; re-check theory if needed; do not cover it up with training curves |
| In C/D, NS is basically feasible and the reference objective gap is very small | No evidence of practical failure yet | Converge to a matrix-analysis paper; application as background and limited diagnostics; do not write "real training fails widely"; do not auto-expand grids |
| Deviations appear only with BF16 or small k | Candidate evidence mainly of a precision/polynomial-setting issue | Prioritize the `precision` plan; discuss implementation error; do not claim discovery of a new smoothing-center effect |
| Deviation drops markedly after switching to the SVD normal, or top gap is extremely small | Normal estimation / non-uniqueness cannot be ignored | Add stratified analysis; do not attribute all PI issues to rank-deficient completion |
| Failures disappear after widening the bracket or changing tolerance | Root-finding engineering factors may dominate | Report failures and fallbacks separately; do not use exact nuclear-norm theory to vouch for the root-finding properties of finite NS |
| Residual is reproducible; repair improves feasibility but training does not improve / gets worse | Mathematical repair is effective; application benefit not established | Keep the certificate and numerical analysis; remove claims of optimizer performance advantage |
| After repair, paired multi-seed loss is better but cost rises significantly | Trade-off between effectiveness and efficiency on small models | Report effectiveness vs. total time/memory; must not write "faster" |
| Advantage remains consistent after fair tuning and independent seeds | Worth an application extension | Add one more model scale / independent corpus and lock the protocol; still not a reproduction of the original large-model work nor a guarantee of acceptance |

In D, `actual_material_residual_flag` uses `|<Θ,Z>|>1e−3` or `||Z||₂>1.001` as a **pre-declared engineering screening line**. It is neither a statistical-significance criterion nor a theoretical threshold. Always look at the raw continuous values, reference error, top gap, layer/step, and solver status together; do not rely on a single boolean.

Important distinction: **training not being superior does not overturn pure matrix theorems; a failed theory check cannot be compensated for by training superiority.** Under the original SSO's rank-one normal, one must not claim that different smoothings have different limiting centers just to tell an application story.

## 7. After the literature search: what to add and what not to add

For detailed sources, repository versions, and corresponding judgments, see `docs/LITERATURE_AND_PROTOCOL.md`. This round actually read SSO's three `numeric_tests`, the standalone `sso.py`, the PolarExpress implementation, GPT-opt's model and learning-rate configs, and the official Muon/nanoGPT documentation.

Added now: the PH path consistent with the manuscript; paired diagnostics on real full matrices; k/precision ablation; PI vs. SVD normal comparison; root-finding failure/fallback; separate learning-rate tuning under equal budgets; checkpoint and data-identity records.

Not added for now: CIFAR-10, ImageNet, eight/nine downstream QA tasks, full Megatron large-model reproduction, and more corpora without a clearly corresponding claim. None of these can currently take priority in resolving the gap of "whether the smoothing path and finite NS are being conflated". TinyStories is for debugging; FineWeb-Edu is the current external web-corpus diagnostic; the seven-source OLMo-Mix is an option for later when closer alignment with SSO's data sources is needed. FineWeb-Edu is not the same as the original FineWeb used in other papers, and the tokenizer also differs, so its loss values or speed leaderboards cannot be compared.

## 8. Training plans to run only after reviewing A–D

`study.py` provides three independent plans; **do not start all of them by default**.

| Plan | Methods | Tuning / formal budget | Reason to trigger |
|---|---|---|---|
| `repair` | AdamW, Muon, MuonSphere, SSO-NS, SSO-NS+SVD repair | Per method 3 LRs × 200-step pilot; after freezing LR on val, seeds 11/22/33 × 1000 steps | To determine whether feasible repair translates into training gains |
| `precision` | The same SSO-NS, k = 5/8/12 | Same number of LR candidates, budget, and paired seeds for each | k has a clear effect and "unfairness caused by a fixed LR" needs to be ruled out |
| `ph` | SSO-NS vs. the expensive SVD PH reference | Same procedure; δ fixed at 1e−3 | A training reference for the finite-scale path is clearly needed, and substantial extra SVD cost is accepted |

The default optional model is changed to 4 layers, width 128, 1000 steps, 2048 target tokens per step. It is still a small-scale controlled study, not a GPT-2 / original SSO paper experiment. `ph` cannot infer the merits of the limiting center from δ = 1e−3 results alone; if δ is selected further, candidates must be pre-declared and other methods must be given a matching tuning budget.

For example, in the same work directory as the notebook:

```bash
python study.py --phase pilot --plan repair --dataset fineweb_edu --work /content/drive/MyDrive/SSO_SIMAX_20260925 --device cuda
python study.py --phase select --plan repair --dataset fineweb_edu --work /content/drive/MyDrive/SSO_SIMAX_20260925 --device cuda
python study.py --phase main --plan repair --dataset fineweb_edu --work /content/drive/MyDrive/SSO_SIMAX_20260925 --device cuda
python study.py --phase summarize --plan repair --dataset fineweb_edu --work /content/drive/MyDrive/SSO_SIMAX_20260925 --device cuda
python run_stage.py --stage export --work /content/drive/MyDrive/SSO_SIMAX_20260925
```

You can first add `--dry-run` to view the pilot/main jobs. After an interruption, repeat the same stage command; existing checkpoints resume and completed runs are skipped. Do not change code or training config and then keep writing into the old experiment directory. If the best learning rate falls on the edge of the grid, first mark it as "search range limited"; do not peek at test and then re-tune one side.

The formal `repair` plan is 15 pilot + 15 formal runs, not a few short runs. Single-GPU cost must be measured first; if C/D do not give sufficient reason, it is not worth paying this compute cost now. The per-step SVD time of the repair method counts toward training; do not show only throughput with the repair cost removed.

## 9. Return, resume, and error handling

`SSO_RETURN_RESULTS.zip` contains the environment, complete logs, config/data manifests, per-step metrics, stage status, and matrix snapshots; it does not contain the large raw corpus, HF cache, or model checkpoints. The latter stay in your own Drive directory for resuming. Raw text does not need to be sent back; the manifest keeps sources and hashes.

Disconnection: reopen the same notebook, upload the same version of the ZIP, connect the same Drive, then re-run the unfinished stages. If resume is rejected because the GPU/software version or forward precision changed, keep the old directory and do an independent run in a new work directory; do not modify identity records to force continuation.

If a download is interrupted and there are only partial token files and no `manifest.json`: keep that directory as failure evidence, rename it, and prepare again; do not treat a half-finished product as complete data. When errors occur, you can also directly run the final export cell and send back the failure logs.

**The only thing I most need from you this time: the complete results ZIP from running A–D.** Once received, we will first determine the actual problem and its source, then decide which empirical results to keep in the main manuscript and whether to start an optional training plan.
