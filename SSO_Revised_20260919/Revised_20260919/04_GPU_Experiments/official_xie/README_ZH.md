# Xie 官方代码的固定版本入口

这里直接获取并启动 Xie 公开的 Megatron 实现，与旁边 `gpu_suite` 的独立重实现分开。**目前状态：精确复现被原始样本/索引、检查点和评测链条的缺失阻断；本目录提供新的公开代码运行入口，不声称重现了原文数字。** 当前助手环境没有 CUDA，不能认证 GPU 启动或完整训练成功。

## 文件与一次性准备

- `bootstrap.py`：固定两个仓库的提交，核验 checkout；不自动安装或升级任何 CUDA 依赖。
- `prepare_megatron.py`：把你选定的、每行含 `text` 的 JSONL 文档交给官方预处理器，下载并固定 OLMo-2 分词器 revision，输出 `.bin/.idx` 及数据 manifest。它不恢复 Xie 的随机样本索引。
- `launch_dense.py`：检查提交、tokenizer、数据文件、GPU 和关键依赖；调用真实的 `pretrain_gpt.py`。不拼接 shell、不依赖原脚本中的相对目录。

在本目录运行：

```bash
python bootstrap.py --root vendor
```

得到 `vendor/Spectral-Sphere-Optimizer` 与 `vendor/Megatron-LM`。对应 commit 分别为：

```text
304d7a4f67c2221cda04b891831db94e5c049092
00a07da9f684d6d0acf661a9cb8eee0c3b0a5aac
```

原始 dense shell 的位置是：

```text
vendor/Spectral-Sphere-Optimizer/megatron_scripts/Dense-1.7B/spball/spball_1.8B_mupinit_muplr5e-3_wd0.1_hard_headfc1split.sh
```

原始代码的许可证保留在 checkout 中；这些 wrapper 是本次新增的研究辅助代码。

## GPU 环境：公开材料没有给出完整锁文件

固定仓库的 `pyproject.toml` 声明 Python ≥3.10、`torch`（未锁定）、`numpy<2.0.0` 和 `packaging>=24.2`；HF 训练相关 extra 包含 `transformers`、`sentencepiece`、`tiktoken`、`wandb`。`tool.uv.sources` 给出 TransformerEngine 的 `release_v2.9`，但相关 dev 依赖行被注释。**这不是作者生产环境的完整可重建锁定。** 不能据此编造一个“已验证原环境”的 Docker 标签或 CUDA 版本。

请在单独的 GPU 环境中准备与本机驱动相容的 CUDA PyTorch、TransformerEngine、Megatron 所需包，再根据固定源码安装。本脚本不自动安装二进制扩展，避免修改现有 GPU 环境。安装前应阅读固定 checkout 的 `pyproject.toml`；`pip install -e '.[mlm,dev]'` 会引入很多额外依赖，不能当作已经验证的一键安装方案。不要在本次数学实验使用的 NumPy 2.x 环境中原地安装这个 Megatron 分支。

launcher 的预检实际导入 PyTorch、NumPy、Transformers、TransformerEngine 和 `megatron.core`，检查 CUDA 与 BF16 支持，记录版本及 GPU 名称。预检通过仍不能保证后续所有分布式/融合算子都正常；出现缺失依赖或编译错误时，保存完整终端日志并先解决环境，不把异常跳过后当成成功实验。

## 数据准备

原始训练语料下载页：[OLMo-Mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)。分词器：[OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B)。**不要改成 Dolmino-Mix-1124 或 FineWeb 后仍称原协议复现。** 公开完整语料非常大；先用明确抽样的新子集做启动检查。

若你已经取得作者原始的 `.bin/.idx` 文件及确切有序权重列表，直接创建下面的 manifest。否则先选择、保存来源明确的 JSONL 文档，再运行：

```bash
python prepare_megatron.py \
  --repo vendor/Megatron-LM \
  --input-jsonl /absolute/path/olmo_selected_documents.jsonl \
  --output-prefix data/olmo_independent \
  --tokenizer-dir data/olmo2_tokenizer
```

每行的格式是 `{"text": "one complete document"}`。新样本的抽样方法、数据 commit、各来源比例、文档 ID/哈希需要另外保留；此处不自动从全库首部截取文档，以免把单一来源误当作混合分布。官方 preprocess 为这些文档加入文档结束标记。它生成 `data/olmo_independent_text_document.bin`、`.idx`、`data/olmo_independent_manifest.json`。launcher 使用与原 shell 相同的 `--split 99,1,0`；这仍不能证明训练和验证集合与论文中的 100B/1B token 集合相同。

已准备好 Megatron 数据时，manifest 格式如下。顺序有意义；相对路径相对于 manifest 文件所在目录，不依赖 shell 的 `find` 顺序。

```json
[
  {"weight": 1.0, "prefix": "/absolute/path/shard0_text_document"},
  {"weight": 1.0, "prefix": "/absolute/path/shard1_text_document"}
]
```

每个 prefix 必须同时有非空 `.bin` 与 `.idx`。这些格式与独立套件的 NumPy token 文件不同；不能直接改后缀混用。预检检查存在性、大小，并分块读取文件保存 `.bin/.idx` SHA256；不会一次把大语料读入内存，但 TB 级文件的哈希扫描需要额外磁盘时间。若 manifest 已提供哈希，会检查一致性。预处理还记录输入 JSONL 哈希与文档数。这些检查不代替官方 loader 对实际二进制内容的完整校验。

## 一条启动命令：先做 50 步启动检查

准备好 GPU 环境、数据和 tokenizer 后：

```bash
python launch_dense.py \
  --repo vendor/Megatron-LM \
  --data-manifest data/olmo_independent_manifest.json \
  --tokenizer-dir data/olmo2_tokenizer \
  --gpus 2 --micro-batch 1 \
  --stop-after 50 \
  --out runs/xie_public_smoke_seed1234
```

只检查文件并查看精确命令可添加 `--dry-run`，它不检查 CUDA，也不启动训练。输出目录必须是新目录或空目录，避免覆盖已有实验。完整运行时移除 `--stop-after 50` 并选择新目录。默认总预算为 100B tokens、4,194,304 tokens/global step，完整日程为 23,841 步；短跑仍使用该完整日程，因此 50 步全部处于 warmup，不能据此判断收敛优劣。

默认 2 卡、micro-batch=1 对应每卡每 optimizer step 512 个 microbatches（TP=PP=1）。这保留逻辑 global batch，却改变硬件和 microbatch 调度，也可能很慢。原始 shell 每节点 8 卡、micro-batch=8；若具备资源可改成 `--gpus 8 --micro-batch 8`。2 张 A100 40GB 是否足够应以首个训练步的实测峰值显存为准，不能保证；80GB 会更宽裕，但同样未经本环境验证。不要把 A100 的耗时与论文 B200 的表格直接比较。

wrapper 保留原 28 层、2048 hidden、6144 FFN、16 query heads/8 KV groups、4096 context、LR=0.005、PI=100、NS=8、2e-4 solver tolerance 等训练设置，默认 seed=1234。它有如下有意记录的变化：单机 `torchrun --standalone`；显式数据清单；较少数据线程；可调保存间隔；保留 optimizer 状态以支持后续恢复；省略部分昂贵健康监控；关闭默认 benchmark 链路；WandB 离线。所有参数保存到 `launch_manifest.json`，真实训练 stdout/stderr 同时写入 `training_stdout_stderr.log`；非零退出记录为 failed。实际分词器字节也记录哈希，但“指定为 OLMo-2”仍应与作者给出的 tokenizer 哈希进一步比对。

恢复时可在同一配置的启动命令中增加 `--resume-from /absolute/path/previous_run/checkpoints`，使用新的 `--out`，并保留原总 token 预算、数据 manifest、tokenizer 和 seed。该选项传给官方 `--load`；不会禁用 optimizer/RNG 加载。目录必须包含 `latest_checkpointed_iteration.txt`，但分布式 checkpoint 完整性及跨 GPU 数量恢复仍需官方 loader 实际确认。本环境没有验证 GPU 中断恢复，建议在昂贵长跑前先做短跑保存/恢复检查。

这一路径用于先验证公开源代码是否能在你的机器上运行。比较算法的低成本训练、几何诊断、数据下载及配对 seed 试验，优先使用独立 `gpu_suite`；它的结果不要填进 Xie 原文的结果表。

## 评测与向作者索取的最小材料

完整数据集官方入口及八项任务映射见 `../XIE_PROTOCOL_REVIEW.md` 第 5 节。公开 shell 列出 SciQ/LogiQA，却未列出表 3 的 CSQA；不能直接把该列表输出命名为“Table 3 reproduction”。此外，之前复核已发现公开 benchmark 装载/校准请求的处理存在问题，尚未证明表 3 实际经过这条链路。此 wrapper 不默认开启它。可用 `python inspect_benchmark_archive.py --download` 下载并只读检查已核验的公开请求包；归档哈希不一致则停止。随附 `benchmark_inventory_20260919.json` 是本次实际重新下载、计数的结果，未做模型推理。

若未来要做原论文的严格复现，需要作者提供或确认：

1. 每个表 3 行对应的 checkpoint、SHA256、训练步、训练代码 commit、完整 launch/config。
2. 原 100B/1B token 的来源 snapshot、抽样 seed、tokenizer 文件与哈希、分片有序权重及离线 document/sample/shuffle indices。
3. 生产环境 lock/container digest：PyTorch、CUDA、TransformerEngine、Megatron、Triton 与自定义 kernel 的版本。
4. FP32/BF16 的真实配置：论文 §5.2 与独立 `sso.py` 精度分工不同，需要明确实际生产实现。
5. 每项 benchmark 的数据版本、split、shot、prompt、上下文长度、长度归一化/校准规则、逐样本预测，以及最终聚合脚本。
6. 表 2 计时的 GPU 数量/型号、并行策略、同步方法、warmup、是否含评测/数据加载及重复次数。

没有这些材料时可以报告“固定公开代码的独立运行”及发现的差异；不宜宣称原论文已经完全复现，也不宜据此断言作者造假。
