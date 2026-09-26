# SSO 数学研究的分层实验套件（2026-09-19）

这是独立实现的小规模实验代码，服务于“可行方向、证书与实际训练方向是否一致”的研究。**不是 Xie 等人的 Megatron/1.8B/100B-token 实验复现，也不预设我们的修复能提升模型效果。** 主稿的数学定理不以 GPU 训练为成立条件；若要声称更好的训练效果、现实训练中故障频率或实际加速，则需要实际 GPU 实验。

优先次序：E0（正确性和精度）→ TinyStories（流程检查）→ OLMo-Mix 缩小实验（同数据来源）→ FineWeb-Edu（外部分布）。不建议在这些检查前耗费资源一比一训练 Xie 的全部模型。原文严格复现所缺的 checkpoint、数据索引、评分协议见配套 XIE_PROTOCOL_REVIEW.md。

## 文件和数据一一对应

| 脚本／配置 | 数据 | 下载网站／用途 |
|---|---|---|
| `oracle_bench.py` | 随机矩阵＋明确的 2×2 反例；无外部数据 | E0；CPU/CUDA 精度、残差、求解耗时。不能估计训练故障频率。 |
| `test_smoke.py` | 本地合成 byte-token 文档 | 工程检查；不是 TinyStories 实验。 |
| `configs/data_tinystories.json`＋`lm_tinystories.json` | `roneneldan/TinyStories` 的 train 来源重新按文档哈希分为 train/val/test | https://huggingface.co/datasets/roneneldan/TinyStories — 人工合成故事，适合流程与小模型，不能代表一般文本。 |
| `configs/data_olmo_mix.json`＋`lm_olmo_mix.json` | `allenai/olmo-mix-1124` 七来源 | https://huggingface.co/datasets/allenai/olmo-mix-1124 — 原文使用的数据集合；本套件重新抽取、重新分割，不能称原文精确数据复现。 |
| `configs/data_fineweb_edu.json`＋`lm_fineweb_edu.json` | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — 教育质量过滤的网页文本，作为独立扩展；不是 Xie 原文数据。 |
| 三种真实数据的 tokenizer | 只下载 `allenai/OLMo-2-1124-7B` 的 tokenizer，不下载 7B 权重 | https://huggingface.co/allenai/OLMo-2-1124-7B — 与原文 tokenizer 来源一致，版本自动解析到 immutable commit。 |

数据集有各自许可：TinyStories 的 CDLA-Sharing、OLMo-Mix 的 ODC-By 及来源条件、FineWeb-Edu 数据卡的 ODC-By 等。请保留来源和数据卡；本代码包不再分发真实 token 语料或 tokenizer；实际下载验证仅保留 manifest 和运行日志。

## 1. 安装和先检查

推荐 Linux、Python 3.11/3.12，先按 PyTorch 官方安装页选择适配服务器驱动的 CUDA wheel，再装其余依赖。不在此固定一个未经本机验证的 CUDA 版本。

```bash
python -m venv .venv
source .venv/bin/activate
# 先按 https://pytorch.org/get-started/locally/ 安装支持本机 CUDA 的 torch
python -m pip install -r requirements.txt
# 若服务器使用 SOCKS 代理而报缺 socksio：python -m pip install 'httpx[socks]'
python -m pip freeze > environment.lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python test_smoke.py --out results/local_cpu_check
python oracle_bench.py --device cuda --out results/e0_cuda.json
```

本交付环境只有 CPU；已实际运行的工程检查和 E0 位于 `results/`，`gpu_executed=false`。另已成功从真实 TinyStories 下载19篇文档并用 OLMo-2 tokenizer 处理，完成一个极小 decoder 的4步CPU训练；这只验证真实数据链路。CUDA、FineWeb-Edu/OLMo七源下载与正式训练未在这里验证，具体见 `VALIDATION_STATUS.json`。不能把代码可运行性检查写成已完成 GPU 结果。

## 2. 下载与固定数据

```bash
python prepare_data.py --config configs/data_tinystories.json --out data/tinystories --cache hf_cache
python prepare_data.py --config configs/data_olmo_mix.json --out data/olmo_mix --cache hf_cache
python prepare_data.py --config configs/data_fineweb_edu.json --out data/fineweb_edu --cache hf_cache
```

只需准备当前要做的一种。默认每种约 32M train tokens、500k val、500k test；全部使用 `<u4`，纯 token 文件约 132 MB，HF 缓存和逐文档清单另外占空间。来源可能下载／扫描超过保留 token 量；OLMo 七来源尤其如此，勿把这个数字当网络流量上限。

`prepare_data.py` 对每个仓库和 tokenizer 先解析 commit，保存 `manifest.json`、tokenizer 文件 SHA256、token 文件 SHA256、每个接受文档的来源 ID、遍历位置、规范化文本 SHA256、token 起止。同一规范化文本的 SHA256 决定 train/val/test，且跨来源去除完全重复，三组不交叉；未做语义近重复消除。TinyStories 官方 validation 不用，本地重新分割与原文不同。

每个来源固定 seed 做 buffered shuffle，再遍历到配额；**不保证对全库均匀随机抽样**。OLMo 七来源以数据卡 token 规模 3700:20.8:58.6:83:11.8:12.2:3.66 分配并按该权重采样训练窗口；不是直接取 default 开头（那可能几乎全是 arXiv），也不保证等同 Xie 未公开的 100B 精确混合、清洗、索引和顺序。训练使用有放回窗口采样，`tokens_seen` 是暴露 token 数，可能重复。

要按已下载版本重建，使用旧 manifest 锁定 revision：

```bash
python prepare_data.py --replay-manifest data/tinystories/manifest.json --out data/tinystories_replay --cache hf_cache
```

需要同样的 Python 依赖版本。不要修改已准备的文件；训练启动会核验 SHA256。失败的半成品目录应改名保留故障记录，再用新目录重试。

## 3. 一个 GPU 的试跑

```bash
python train.py --config configs/lm_tinystories.json --data data/tinystories --optimizer sso_ns --seed 11 --device cuda --out runs/tiny_sso11 --stop-after 20
python train.py --config configs/lm_tinystories.json --data data/tinystories --optimizer sso_ns --seed 11 --device cuda --out runs/tiny_sso11 --resume
```

默认小 decoder：4 层、宽度 256、4 heads、长度 256、microbatch 4、accumulation 8、1000 steps，即 8.192M 暴露训练 tokens。词嵌入与输出头绑定；参数量由 tokenizer 决定，日志记录实际值。它不是 OLMo 或原文 Dense 架构，故不能直接比较论文中的 loss 数值。调整 batch/accum 保持乘积可控制有效 batch；任何改动都须新开实验并记录配置。恢复要求 config、源码、数据、seed、optimizer、CPU/CUDA 类型不变。

所有优化器在同 seed 下有相同参数初始化、数据 RNG 和预算。隐藏矩阵统一从同一个 PI 归一化初始化开始；embedding、position、LayerNorm 等统一由 AdamW 更新。默认矩阵 WD=0、辅助参数 WD=0.1，这是固定的受控选择，不宣称复现原文最优超参数。前向默认 BF16，权重和优化器状态 FP32，NS 默认 FP32；用 `--ns-dtype bfloat16` 明确做 NS 精度消融，不能把两者混成一组结果。

## 4. 方法命名和哪些不是高效优化器

| `--optimizer` | 本套件的确切含义 |
|---|---|
| `adamw` | 隐藏矩阵与辅助参数均用 AdamW；两组 lr 可不同。 |
| `muon` | EMA/Nesterov＋8 步 Polar-Express＋`sqrt(nout/nin)` update scaling；独立受控 Muon 类实现，不是所有 Muon 发布版本的逐位复制。 |
| `muon_sphere` | 同上，更新前用 PI 归一化到目标半径；无切向求根。 |
| `sso_ns` | 同上，加有限 NS 的切向求根；保留 standalone 默认的 10 次 bracket expansion/失败回旧 λ，并显式计数失败。不是精确 polar oracle。 |
| `sso_ns_repair_svd` | 对有限 NS 输出用真法向＋SVD 范数做 T6 可行性修复；**昂贵诊断对照，不是已提出的可扩展新优化器**。默认调参网格不运行它。 |

Sphere 方法采用 **pre-update retraction**；因此更新后的权重可以偏离目标半径，下一步再次归一化。法向、NS 方向和诊断都在更新前的同一权重状态计算。PI 为 FP32、默认100步、ones 初值；NS 用原公开系数，求根默认容差1e-4，并记录 fallback/residual_failed。SVD 修复仅处理方向可行性，不保证训练更优，且不能改善 PI 归一化本身的误差。浮点 SVD 的“真法向”只是数值参考；接近重奇异值时日志标记不唯一，不给出严格区间证书。

## 5. 公平调参与三 seed 正式实验

不能只替修复方法多调参数，再用固定 lr 的 Muon 作结论。脚本统一每方法 3 个 matrix lr、辅助 lr固定、pilot seed7、200 steps；用 val 选最低 loss。正式使用独立配对 seeds 11/22/33、1000 steps，最终一次 test。范围是预先声明的小网格，不能声称找到了各方法全局最佳参数。

```bash
python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda
python run_grid.py --phase select --dataset olmo_mix --data data/olmo_mix --out studies/olmo
python run_grid.py --phase main --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda
python summarize.py --root studies/olmo/main --out studies/olmo/summary.json
```

TinyStories/FineWeb-Edu 只需分别将 `--dataset` 改为 `tinystories`/`fineweb_edu` 并换 data/out。先用 `--dry-run` 查看即将运行的命令。2张GPU建议分跑实验，而非小模型做 DDP；两个终端对应：

```bash
CUDA_VISIBLE_DEVICES=0 python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda --shard 0 --num-shards 2
CUDA_VISIBLE_DEVICES=1 python run_grid.py --phase pilot --dataset olmo_mix --data data/olmo_mix --out studies/olmo --device cuda --shard 1 --num-shards 2
```

等两边 pilot 都完成后单独 `select`；再把上面两条的 `--phase pilot` 改为 `--phase main`。同一个输出目录的同一训练任务不要并发启动。网格脚本按任务分片，checkpoint 自动续跑。

## 6. 需要保存和报告的量

- `metrics.jsonl`：训练 loss、固定 validation loss/PPL、tokens、逐步耗时、梯度范数、solver status。PPL 只在同 tokenizer、数据与评估规则下可比较。
- `diagnostics.jsonl`：**实际有限 NS/修复方向**对估计法向和 SVD 参考法向的残差；另列替换成 compact polar 后的残差（同估计问题／真法向问题各一列），不得冒充实际方向残差。
- 同时保存方向谱范数、PI 法向误差、权重更新前 top gap、更新后半径误差、实际位移法向分量、对真法向问题的 T6 repaired primal-dual gap。这个 gap 是给定候选方向修复后的数值上界，不是严格浮点认证，也不是实际训练 objective 的下降量。
- `checkpoint.pt`：模型、矩阵/辅助优化器、Python/NumPy/Torch/CUDA RNG、训练 sampler RNG与位置统计、step、配置与数据/代码哈希。只读取自己信任的 checkpoint（含 Python 序列化状态）。
- `sessions.jsonl` 和 `result.json`：各次恢复会话墙钟、结束状态、峰值 allocated 显存。unfinished session 不会伪造耗时；重启前未闭合会话不能精确追溯。详细 checkpoint 累计计数可能少最后一次写盘，跨重启总时间以完整 session wall 为准。
- `summarize.py` 按完整配置/数据 manifest/CPU或CUDA分组；不混数据或精度求均值。给 seed 均值与样本标准差，不自动宣称显著。吞吐排除 warmup 和整个 SVD diagnostic step；比较实际训练总时间时须包含这些步骤及验证/保存成本，两种口径分别报告。

验证窗口固定，按来源权重有放回抽取；所有方法看到相同 val/test 窗口。三 seed 只能初步刻画波动，不能证明普遍优越。比较最终 test 时用配对 seed 差值并报告全部单次结果；不要根据 test 继续挑 lr 或改方法。

## 来源与实现边界

- Xie 等：Controlled LLM Training on Spectral Sphere, https://arxiv.org/abs/2601.08393
- 原 SSO standalone： https://github.com/Unakar/Spectral-Sphere-Optimizer ，核对 commit `304d7a4f67c2221cda04b891831db94e5c049092`，`sso.py` SHA256 `1b7416bff73e70bdb923c2f6744127682d826cf0283528605dc7a7f2708545d1`。本包未冒充其生产 Megatron 实现；只核对数学结构、系数及默认 fallback 行为。
- HF streaming API： https://huggingface.co/docs/datasets/en/stream
- PyTorch Muon 文档： https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html （说明不同 scaling 约定；本实现明确自己的约定）。

原文下游 8 个问答任务及原评分脚本的问题由配套协议单独处理。本套件不提供未经验证的 lm-eval 接口，也不把训练 loss 代替那8项准确率。
