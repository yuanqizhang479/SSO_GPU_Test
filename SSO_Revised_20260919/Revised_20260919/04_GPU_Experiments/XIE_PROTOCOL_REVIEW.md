# Xie 基础论文的对照协议与 GPU 研究推进建议

核对日期：2026-09-19。基础论文为 Xie 等，*Controlled LLM Training on Spectral Sphere*, arXiv:2601.08393v3（2026-03-05）。本文档只核对公开资料，并区分数学验证、实现验证和训练复现；没有运行新的 GPU 训练，也没有获得作者未公开的检查点或数据索引。

## 1. 是否需要与 Xie 完全相同的 GPU 测试

当前主稿研究的是矩阵直线上的核范数最佳逼近、秩亏处证书补全及奇异平滑渐近律。**这些数学结论不以 GPU 训练为成立条件。** CPU 高精度实例与严格证明是直接证据。GPU 实验的价值是检验有限精度实现和真实训练矩阵上现象的范围。如果论文进一步主张“改进后的优化器更快、更稳定、损失更低”，则必须新增训练实验；仅有内层最优性改善不足以推出这些结论。

因此，建议先完成下文 E0–E2，再决定是否扩大规模。无须先重跑 Xie 的 Dense 1.7B/100B tokens、MoE 和 200 层网络，才能发表一个成立的矩阵分析结果。若要声称“复现了 Xie 表 3 的数字”或“在原协议下优于 SSO”，则确实需要对应的数据、模型、训练和评测协议；换成小模型、TinyStories 或 FineWeb 的结果应称为独立实验或缩小规模的机制检验。

## 2. 原论文实际上使用什么

以下来自用户提供的基础论文 §5–§6、算法 1，与[公开 v3](https://arxiv.org/html/2601.08393v3)对照。文件名中出现 `1.8B`，而正文称 Dense 1.7B；应保留这一命名差异，不能自行认定是不同模型。

| 项目 | 论文/公开脚本中的具体设置 | 对我们的含义 |
|---|---|---|
| 训练语料 | **OLMo-Mix-1124**，100B 训练 tokens、1B 验证 tokens | 不是 FineWeb，也不是 FineWeb-Edu；更不是 Dolmino-Mix-1124 |
| 分词器 | OLMo-2；脚本指向 `models/OLMo-2-1124-7B` | 更换分词器会改变 token 预算与 loss/PPL 可比性 |
| Dense 架构 | Qwen3-1.7B 配置；28 层、宽 2048；GQA、QK-Norm、SwiGLU、RoPE、RMSNorm；输入输出嵌入不共享 | 小型 GPT 即使参数接近，也不能叫完全同架构复现 |
| 批次与序列 | 1024 × 4096 = 4,194,304 tokens/step | 缩小批次会改变优化动力学；梯度累计可维持逻辑批次但不维持时间性能 |
| 日程 | 500 步 warmup；余弦降到峰值 LR 的 10%；主设置 LR 0.005 | 独立实验应记录自己的日程，并给各优化器相同调参预算 |
| 精度/近似 | 正文：BF16 训练；PI 用 BF16，msign 用 FP32、8 步 Polar Express；Nesterov | 独立 `sso.py` 的 msign 是 BF16、PI 是 FP32；不能把两者视为同一实现 |
| 约束求解 | 容差 2e-4；最多 20 次求根迭代 | 应另外记录失败/达到上限状态以及最终实际输出残差 |
| 几何粒度 | attention 按 head、FFN gate/up 分拆；隐藏矩阵不用 weight decay | 不分拆 QKV 的小模型实验属于改变协议 |
| 公开 dense 脚本 | PI = 100；每节点启动 8 进程；本地 `.bin/.idx` 数据及 cache | README 的 PI 默认 20 与脚本不同；不能只抄 README 默认值 |
| 时间结果 | B200、约 4M tokens/step 的端到端时间 | 2 张 A100 上的实测时间不能验证 B200 的原始延迟数字 |

公开主仓库：[Unakar/Spectral-Sphere-Optimizer](https://github.com/Unakar/Spectral-Sphere-Optimizer)。2026-09-19 API 查询的主分支提交仍为 `304d7a4f67c2221cda04b891831db94e5c049092`。公开训练仓库：[Unakar/Megatron-LM，SSO_main](https://github.com/Unakar/Megatron-LM/tree/SSO_main)，提交仍为 `00a07da9f684d6d0acf661a9cb8eee0c3b0a5aac`。本次查询返回的 JSON 和相关原始文件已留作参考。

**当前不能保证精确复现的缺口：**已检查文件没有给出可逐项核验的“表 3 单元格 → checkpoint 哈希 → 训练提交 → 原始 100B/1B 样本及 token 索引 → 全量逐题预测”的完整链条。公开 shell 引用本地 `data/merged_data`、`data/merged_data_cache`，本身不是那个确定样本序列。README 的默认 benchmark 列表与表 3 也不完全一致。即使解决环境依赖并成功启动，仍不能自动宣称精确复现。

这是复现材料的不完整与协议歧义，不能单独推出实验伪造。已发现的数学缺陷、公开实现问题、评测装载问题应各自陈述；目前没有足够证据认定捏造结果。

## 3. 代码、数据集和下载入口

下面是新 GPU 套件的预定实验入口；实际完整命令与依赖以 `gpu_suite/README` 和套件配置文件为准。所列网站是数据来源，不要求先下载完整 TB 级语料。脚本采用有预算的数据准备，并保存数据来源、版本和样本清单。

| 代码/配置 | 数据库 | 官方下载或数据卡 | 用途与边界 |
|---|---|---|---|
| `oracle_bench.py` | 无；构造矩阵与固定种子随机矩阵 | 不需要下载 | E0：实际返回方向的精度、可行性、证书 gap、耗时；不证明训练效果 |
| `prepare_data.py --config configs/data_tinystories.json`；`configs/lm_tinystories.json` | TinyStories | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | 小词汇量的合成短故事；适合低成本训练与排错。不是通用语料，也不是 Xie 数据 |
| `prepare_data.py --config configs/data_fineweb_edu.json` | FineWeb-Edu，`sample-10BT` | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 教育质量筛选的网页语料；新数据来源验证。它与未筛选 FineWeb 不可混写 |
| `prepare_data.py --config configs/data_olmo_mix.json` | OLMo-Mix-1124 的七来源预算子集 | [allenai/olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) | 最接近 Xie 的语料来源；仍不是作者的原始 100B 随机样本及顺序 |
| 可自行增加的同格式数据配置 | FineWeb，`sample-10BT` | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 检查结论对教育质量筛选是否敏感；本轮不应把它与 FineWeb-Edu 视为两个完全独立总体 |

OLMo-2 分词器可从[官方模型页面](https://huggingface.co/allenai/OLMo-2-1124-7B)获取，仅调用 `AutoTokenizer.from_pretrained(...)` 不需要下载 7B 模型权重。TinyStories 要固定使用初版或 GPT-4-only V2：两者不是同一版本，原作者数据卡明确区分了它们。FineWeb 和 FineWeb-Edu 的 `sample-10BT` 名称表示以 GPT-2 分词估计约 10B tokens 的官方子集；使用其他分词器时，不能把这个名字当成本次实际训练 token 数。

建议的数据准备调用：

```bash
python prepare_data.py --config configs/data_tinystories.json --out data/tinystories
python prepare_data.py --config configs/data_fineweb_edu.json --out data/fineweb_edu
python prepare_data.py --config configs/data_olmo_mix.json --out data/olmo_mix
```

**OLMo 抽样要点：**不要对默认 `data/**/*` 流直接取前 N 条，再把它称为原始混合分布。文件顺序可能使子集集中于单一来源。官方七个 config 为 `dclm`、`arxiv`、`pes2o`、`starcoder`、`algebraic-stack`、`open-web-math`、`wiki`。新套件按官方披露的来源 token 规模分配近似 token 配额；这些规模经过四舍五入，因此仍只是新的明确抽样方案。不同来源必须先划分训练/验证文档，再打包 token；保存文档哈希以检查跨集合重叠。使用任何数据都遵循对应数据卡许可，不把代码的 Apache 许可延伸为全部数据的许可。

认可度不等于无偏：OLMo-Mix 是开放模型训练的可追溯混合语料；FineWeb 系列有公开处理过程；TinyStories 特别适合小模型，但分布很窄。我们的选择标准应是任务匹配、处理透明、数据可固定以及结论跨分布是否稳定，而不是仅按下载量或名气评价。

## 4. 建议的实验层次：先核验内层，再讨论外层

以下是资源规划建议，并非已经测得的运行时间。A100 40GB/80GB、CUDA 版本、编译、SVD 次数会显著影响可行预算，必须用前 50–100 步测速与显存实测再外推。最多两张卡时，可各自跑一条相同规模、不同 seed/方法的单卡作业；这通常比立即引入分布式训练更易控制变量。

| 层次 | 建议内容 | 需要的证据 | 能支持的结论 |
|---|---|---|---|
| E0，优先 | 单卡矩阵实验：解析 2×2 反例、其嵌入、方阵/长矩阵/宽矩阵；FP64 参考、FP32、BF16；相同 G、Θ | 原始矩阵、种子、实际输出方向、求解状态、残差、谱范数、修复后的 primal-dual gap、GPU 同步计时 | 有限精度实现的失败方式或改进；不估计训练发生率 |
| E1，优先 | 小型 Transformer 的真实训练矩阵快照；在同一快照离线比较各内层求解器 | 固定层与步骤抽样规则；G/W/动量；真实 SVD 法向与 PI 法向的差异；A(λ) 的最小奇异值；分层残差统计 | 病态情况是否出现在这组真实矩阵；不能把近秩亏计数称为严格秩亏概率 |
| E2，推荐 | TinyStories 小规模训练，随后 OLMo 子集或 FineWeb-Edu；先小预算，后扩到 100M–1B tokens 的选定配置；至少 3 个配对 seed 作初步比较 | 相同数据序列/架构/预算，等额 LR 搜索；验证 loss、tokens/s、time-to-loss、峰值显存；全部失败运行 | 在明确小模型协议下的实用性；不能外推到 1.7B/100B 或 MoE |
| E3，可后续 | 小模型在第二语料和另一宽度上的确认；必要时少量下游任务 | 预先选定主指标，保留每 seed 曲线及逐题分数 | 跨数据/宽度的有限外部有效性 |
| E4，单独项目 | 原 Qwen3-1.7B/100B + 完整下游，以及可选 MoE/DeepNet | 补齐上节作者数据索引/配置/检查点关联；按原硬件或明确变更核算成本 | 原规模独立复现；只有协议和数值均对齐后才能称原表精确复现 |

E0/E1 应同时报告三个容易混淆的量：求解器用估计法向得到的残差、**同一个实际输出方向**相对高精度真实法向的残差，以及修复可行后的目标证书。将输出方向替换为精确 compact polar 后再算的残差，只是另一条方向的诊断，不能冒充原输出的切向误差。相对 Frobenius 误差也不是目标值 gap。

对于 E2，至少保留 AdamW、Muon、MuonSphere、`SSO-NS-reimpl` 与所提修复方案。最后两者应在同一程序中尽量只改变内层步骤。不能把 SVD 修复增加的计算时间隐藏在数据加载或离线分析中；同时报告几何误差与训练代价。新套件的重实现名称是有意的：它不是 Xie Megatron 生产代码的等价认证。

先固定主对比与容差，再看结果。3 个 seed 只是最低限度的初步变异检查，不保证识别微小差异；没有检出显著性也不等于效果为零。只有测过的 GPU 结果才写入论文结果表，尚未运行的配置只放在研究路线或实验计划。

## 5. 与表 3 对照的八项评测

Xie 表 3 的平均数基于八个 accuracy 项：LAMBADA、CSQA、PIQA、HellaSwag、WinoGrande、ARC-Easy、ARC-Challenge、BoolQ；LAMBADA PPL 另列，不能混入 accuracy 平均。README/公开评测 shell 的九任务列表含 SciQ 和 LogiQA，却不含 CSQA，因此不是表 3 的自足复现入口。

| 数据集 | 官方入口/下载页 | 常见独立评测设置，需要显式固定 |
|---|---|---|
| ARC-Easy、ARC-Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | test；分别报告；shot 数、选项长度归一化 |
| CommonsenseQA | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | 有公开标签的 validation；不能把无标签 test 当可本地算准确率的集合 |
| PIQA | [作者数据页](https://yonatanbisk.com/piqa/)、[作者 HF 仓库](https://huggingface.co/datasets/ybisk/piqa) | validation；注意旧 HF 仓库的自定义加载脚本兼容性；保留原始数据哈希 |
| HellaSwag | [作者主页和下载链接](https://rowanzellers.com/hellaswag/) | validation；公开脚本使用 5-shot，但需要固定提示示例及顺序 |
| WinoGrande | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | config、validation、目标词/补全文本计分方式 |
| BoolQ | [google-research-datasets/boolean-questions](https://github.com/google-research-datasets/boolean-questions)、[google/boolq](https://huggingface.co/datasets/google/boolq) | validation；yes/no 字符串、大小写、空格与 likelihood 模板 |
| LAMBADA | [原作者 cimec/lambada](https://huggingface.co/datasets/cimec/lambada)；[OpenAI 预处理版本](https://huggingface.co/datasets/EleutherAI/lambada_openai) | 原始版与 OpenAI 版分开；最后一个词的准确率和 PPL 定义，不能默认为全文 PPL |

可以使用固定版本的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)开展新的独立评测，但“同名任务”不等于“同提示、同样本、同聚合”。每次保存：评测框架 commit、task YAML、样本 ID、shot 示例 ID、tokenizer revision、checkpoint SHA256、逐题各选项分数、正确性与聚合脚本。不要把校准用空上下文请求当成新的独立问答题；若使用校准，保留题目—校准请求映射，并明确校准公式。

2026-09-19 已重新下载公开 [Megatron benchmark 请求包](https://github.com/richardodliu/Megatron_benchmark)，17,711,385 字节，SHA256 为 `904dfc3c2ba874d92d6dfb521ae2fe919951cbbd6a63e69a8ecdaff63f3f5568`，与此前审计相同。此次实际只读计数如下；偏移 ID 对应的请求上下文为 `Answer:`，与主 ID 一一配对。

| 请求包任务路径 | shot | 主问题数 | 偏移 ID 组数 | 直接按原始 doc_id 分组的组数 |
|---|---:|---:|---:|---:|
| `csqa/rc_0shot` | 0 | 1221 | 1221 | 2442 |
| `piqa/rc_0shot` | 0 | 1838 | 1838 | 3676 |
| `arc_easy/rc_0shot` | 0 | 570 | 570 | 1140 |
| `arc_challenge/rc_5shot` | 5 | 299 | 299 | 598 |
| `hellaswag/rc_5shot` | 5 | 1000 | 1000 | 2000 |
| `boolq/rc_0shot` | 0 | 3270 | 0 | 3270 |
| `winogrande/rc_0shot` | 0 | 1267 | 0 | 1267 |
| `lambada/ppl_0shot` | 0 | 5153 | 0 | 5153 |

ARC 的 570/299 是与其 validation 数量一致的规模；不能把它们描述为常见 test 集的 2376/1172。HellaSwag 5-shot 包只含 1000 个主问题，不能当作全部 10042 条 validation。仅凭数量一致也不足以认证逐条样本完全相同。这些差异加上校准请求处理方式，意味着“拿公开 zip 跑完”仍不自动等于表 3 的原流程。本次没有做 checkpoint 推理，不能从这些计数推断原文分数的偏移方向或认定伪造。

当前阶段不需要把八项全部加入小型 TinyStories 实验。极小模型在常识任务可能接近随机水平，难以诊断内层数学改进。优先用直接对应数学主张的 E0/E1 和真实验证 loss；如增加下游任务，事前选定，不能看哪项有利才报告。

## 6. 论文中如何引用 Xie

建议在背景中承认其研究动机：谱范数约束的训练方法使带切向约束的矩阵线性优化问题具有实际相关性。随后明确，本文研究的是该内层问题在秩亏处的完整证书几何及正则化路径。不能把“某种 compact-polar 单值根不存在”写成“SSO 所有实现无法训练”；也不能把同数据来源的小模型结果写成原论文大型实验被推翻。

可用的英文表述：

> Spectral-sphere training motivates linear optimization over an ambient spectral-norm ball intersected with a tangent hyperplane. Our analysis concerns the complete optimal-certificate set of this inner problem, including rank-deficient matrix-line crossings. It does not, by itself, determine how frequently these crossings affect a finite-precision training trajectory or whether a different certificate improves outer-loop convergence.

可用的引用条目：

```bibtex
@misc{xie2026controlled,
  title={Controlled LLM Training on Spectral Sphere},
  author={Xie, Tian and Luo, Haoming and Tang, Haoyu and Hu, Yiwen and
          Liu, Jason Klein and Ren, Qingnan and Wang, Yang and Zhao, Wayne Xin
          and Yan, Rui and Su, Bing and Luo, Chong and Guo, Baining},
  year={2026},
  eprint={2601.08393},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note={Version 3, 5 March 2026},
  url={https://arxiv.org/abs/2601.08393v3}
}
```

本次推进的合理边界是：修正文稿中确定的公式/退化情形错误，收紧与 Xie 和相关工作的关系陈述，提供可追踪的 CPU/GPU 研究脚本和运行计划。尚未取得的 GPU 结果、严格区间算术证书、大规模训练优势与实验真实性判断，都不能通过文字修改补出来。
