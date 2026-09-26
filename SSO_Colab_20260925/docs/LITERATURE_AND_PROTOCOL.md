# Related Literature, Official Code, and Decisions on Experiment Additions

Search date: 2026-09-25. Only papers, author repositories, and official data cards are used as the basis. This is a targeted search for this round's experiment design, not a completed comprehensive novelty review.

## What was read and its concrete impact

| Literature/code | What was actually found | What this package adopts | What cannot be claimed from it |
|---|---|---|---|
| Xie et al., *Controlled LLM Training on Spectral Sphere*, arXiv:2601.08393v3; [author repository](https://github.com/Unakar/Spectral-Sphere-Optimizer) | The official repo has `numeric_tests/check_lambda.py`, `check_msign.py`, `check_power_iteration.py`, which check the scalar function, NS/SVD differences, and PI/SVD differences respectively; there is also a Megatron route | Separates the three kinds of error and replays them on pre-fixed real full matrices; keeps the fallback | "monotonicity verified" on a finite random grid is not a proof of general monotonicity; this suite is not a Megatron reproduction |
| Amsel et al., *The Polar Express*, [paper](https://arxiv.org/html/2505.16932v2), [PolarExpress](https://github.com/noahamsel/PolarExpress), [GPT-opt polar branch](https://github.com/modichirag/GPT-opt/tree/polar) | Has a BF16 stability design; learning-rate comparisons for training GPT-2 on FineWeb; the official implementation README points to GPT-opt; configs contain multiple LR candidates and NS step counts | Adds paired diagnostics over k and dtype; tunes LR separately for training comparisons; states explicitly that coefficients come from the fixed SSO implementation | Higher orthogonalization accuracy does not automatically mean better or faster training; k=5 in this package is the first five steps of the SSO coefficients, not Jordan's fixed quintic applied five times |
| Shulgin et al., *Beyond the Ideal: Analyzing the Inexact Muon Update*, [paper](https://arxiv.org/html/2510.19933v1) | NanoGPT/FineWeb and CIFAR-10 experiments; analyzes the coupling of approximation accuracy with learning rate and momentum | Sets up a conditionally triggered `precision` plan; LR selected independently with equal budget for each k | This round does not claim to have found and reproduced a complete independent repository from the authors; no momentum grid was done, so it cannot be called thoroughly tuned |
| [KellerJordan/Muon](https://github.com/KellerJordan/Muon) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Muon is used for hidden matrices and AdamW for other parameters; the speedrun fixes data/evaluation and emphasizes measured time; the current main-track hardware assumes multiple H100s | Fixed parameter grouping, auxiliary AdamW, initialization, sampling seeds; full runtime and memory reported separately | Colab small-model numbers are not compared directly with speedrun leaderboard numbers |
| Penedo et al., *The FineWeb Datasets*, [NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/370df50ccfdf8bde18f8f9c2d9151bda-Abstract-Datasets_and_Benchmarks_Track.html), [FineWeb-Edu data card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Officially released web corpus and an educational-quality filtered subset | Continues 9/19's use of FineWeb-Edu as a downloadable external text diagnostic; fixes subset, version, and split rule | FineWeb-Edu and FineWeb must not be conflated; perplexity with other tokenizers is not directly comparable |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), [OLMo-Mix](https://huggingface.co/datasets/allenai/olmo-mix-1124) | The former suits small-model pipeline checks; the latter provides data sources relevant to the SSO literature | TinyStories debugging first; OLMo-Mix reserved for a later source comparison | Small-story data or a re-extracted OLMo subset is not the same as SSO's original training corpus |
| Do, Dereich & Jentzen, [*On MUON optimization…*](https://arxiv.org/abs/2608.04607), 2026-08 preprint | Directly analyzes NS/PolarExpress in implementations, and non-convergence/errors on specific stochastic problems | Needs discussion in future related work; reiterates that "finite NS" should be treated as a separate mathematical object | Reading the abstract is not enough to judge whether it covers this manuscript's nuclear-norm-slice smoothing singular rates; this round does not claim to have ruled out all theoretical overlap |

## Pinned external code versions

This round verified and read the original files via the GitHub API, recorded in `UPSTREAM_PROVENANCE.json`. The latest online branches were not silently swapped into the training kernel.

- SSO: `304d7a4f67c2221cda04b891831db94e5c049092`. The `sso.py` on main at the time of this round is identical to this pinned version, SHA256 `1b7416bff73e70bdb923c2f6744127682d826cf0283528605dc7a7f2708545d1`.
- PolarExpress: `71cc37943d99cae780024c1d198977f2f8795407`; its implementation and README were read.
- GPT-opt polar: `cf8f059c790a7351843cbf1e6b12a03910bd122a`; README and `configs/gpt-Large-fine1B.yaml` were read.

Upstream code is used for provenance verification and design comparison; this package did not run all of its tests or reproduce its training results. `UPSTREAM_SSO_LICENSE.txt` and `NOTICE.md` are kept, and the original contributions of external algorithms are not attributed to this project.

## Why no more datasets are added this round

The manuscript targets SIMAX matrix analysis. The most discriminating evidence at present is: same input, same normal, actual output, explicit smoothing scale, real failure logs. Adding a new dataset without these controls cannot resolve the theory–implementation correspondence question.

If the goal later shifts to broad performance claims about machine-learning optimizers, the experimental bar rises: larger model/training budget, additional independent data or tasks, consistent accounting of tuning budget, throughput, and total cost; standard downstream evaluations where necessary. But this is not a prerequisite for the theoretical main line of this round, and it cannot be substituted by the current short trajectories of tens of thousands of tokens.

The above trade-offs are research-design judgments made this round based on the papers and compute constraints; they do not use the cited literature to prove benefits for this project.
