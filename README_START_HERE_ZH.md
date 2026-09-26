# GPU实验包：从这里开始

本包有两条独立路线，均未在本环境执行CUDA训练。

- `gpu_suite/`：独立实现的小模型受控实验。推荐先读其中README_ZH.md，先E0，再TinyStories短跑，再OLMo-Mix与FineWeb-Edu；每个数据集有独立data/lm配置，下载网址和命令一一对应。五方法CPU前反传与精确恢复、18项E0及真实TinyStories4步CPU流程已有记录；正式GPU网格尚待执行。
- `official_xie/`：固定公开SSO/Megatron版本的下载、数据预处理及Dense启动包装。它更接近Xie原实验协议，但没有原始数据抽样索引/checkpoint，不能称一比一结果复现。官方Dense架构不是可任意两卡运行的承诺，显存和运行时间需真实预检。不要与小模型路线混用环境。
- `XIE_PROTOCOL_REVIEW.md`：基础论文v3、训练语料、八项评测任务、代码版本和已知协议缺口。

推荐先运行（已安装正确CUDA版PyTorch后）：

```bash
cd gpu_suite
python -m pip install -r requirements.txt
python test_smoke.py --out results/local_cpu_check
python oracle_bench.py --device cuda --out results/e0_cuda.json
python prepare_data.py --config configs/data_tinystories.json --out data/tinystories --cache hf_cache
python train.py --config configs/lm_tinystories.json --data data/tinystories --optimizer sso_ns --seed 11 --device cuda --out runs/tiny_sso11 --stop-after 20
```

然后按README继续恢复训练/公平调参/三配对seed。若发生错误，保留完整日志与环境，不删除失败状态重新包装成成功。真实训练得到的loss、方向残差、速度和失败率分别报告，不能互相替代。基础论文：https://arxiv.org/html/2601.08393v3 。
