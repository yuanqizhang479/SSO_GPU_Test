# 数值复现复核与修改说明（2026-09-19）

本文件只报告实际执行的 CPU 数学实验与数值文稿修改。没有执行 GPU 训练，没有把合成矩阵实验写成语言模型训练结果，也没有把工作精度或多精度一致性写成机器严格认证。

本次最终执行结果：原始 10/10、修订 14/14 均正常返回。原版与修订版均未修改科学判据容差来消除失败。

## 1. 对既有建议的重新判断

| 建议 | 判断 | 本次落实 |
|---|---|---|
| 原始 10 脚本整体复跑 | 合理；能检验给定例子是否复现，不能证明一般定理 | 原始源码保持不动，安装其已声明但当前环境缺失的 `mpmath==1.3.0` 后完整复跑，10/10 通过 |
| 消除 `python -O` 假绿 | 必须修复 | 所有保留裸 `assert` 的验证入口在优化模式下主动抛错；总入口同样拒绝 `-O` 和 `PYTHONOPTIMIZE=1`；新增两项理论检验使用显式异常 |
| 非对角例子真正高精度复算 | 必须修复 | 新增 80 位计算和独立 100 位重复；有限尺度比值一致至绝对差小于 `1e-60`，内禀常数一致至小于 `1e-65` |
| 所有文稿表格从同一程序生成 | 必须修复 | `verify_publication_tables.py` 统一写入 JSON、LaTeX 数字宏和 8 张表；`--check` 重算并逐字比较快照 |
| 改正 Huber、p=3 与非对角方向比的末位 | 必须修复，属于数值生成链问题 | 全部改为高精度值，理论极限和指数不变 |
| 补充 p=3、oscillatory 完整表 | 合理；原稿有承诺但未完整展示 | 补齐 profile 多尺度表、明确交替路径及其 9 行结果 |
| 重复 SVD 不应默认返回同一零空间基 | 合理 | `verify_intrinsic_constants.py` 的 `s=0` 分支复用定义核块时的同一组基 |
| 实际返回方向与另算 compact polar 分开 | 必须明确 | 加入已有证书修复数值例，并明确 float64 八步多项式不是生产 BF16、也不自动属于 Fenchel 平滑家族 |
| 增加 GPU 结果才能成立数学定理 | 不成立 | 本文的矩阵分析定理、局部渐近与反例不依赖 GPU；训练实验只能支持额外的应用主张 |
| 用数值全通过认定论文绝无错误 | 不成立 | 开放集、唯一性分类、一般渐近余项仍须依靠证明；本次执行不构成形式化验证 |

## 2. 可以直接复跑的入口

在修订源码目录中：

```bash
python -m pip install -r requirements.txt
python verify_all.py
```

总入口依次运行原有 10 项与 4 项新增检查，并生成：

- `results/verification_latest.json`：Python、依赖版本、平台、线程环境、每项退出码、耗时、源码与输出 SHA-256；
- `results/verification_logs/`：逐脚本原始输出；
- `results/publication_numbers.json`：论文数值的机器可读快照；
- `results/certificate_example_results.json`：实际输出与替换 compact-polar 的比较；
- `results/optimization_guard_checks.json`：本次实际运行的三种优化模式拒绝检查。

单独检查或更新表格：

```bash
python verify_publication_tables.py --check
python verify_publication_tables.py
```

第一条重算并核对已交付快照；第二条显式重新生成快照。最终 `verify_all.py` 使用第一条，防止悄悄覆盖已有论文数字后仍宣称“与快照一致”。

## 3. 结果生成与数值修正

| 项目 | 原文/原脚本显示 | 本次高精度值（保留部分位数） | 对结论的影响 |
|---|---:|---:|---|
| Huber boundary，δ=10⁻⁸ | 1.9999999656 | 1.9999999500000019 | 不改变线性律及极限 2 |
| p=3 boundary，δ=10⁻⁹ | 1.2778860473 | 1.277886045147… | 不改变 3/4 指数及极限 |
| 非对角方向比，δ=10⁻⁵ | 0.671768254 | 0.671768177074147… | 不改变非零一阶方向项 |
| 非对角方向比，δ=10⁻⁶ | 0.6717760680 | 0.671774660292359… | 不改变其收敛至 0.6717753807… |
| 非对角有限尺度 t₁，δ=10⁻⁵ | 文稿 0.7010473954；float64 脚本 0.7010476154367672 | 0.70104739544355205579… | 文稿原值正确；本次补上真正高精度复现 |

非对角例的 4×4 矩阵可以按坐标 `(1,3)`、`(2,4)` 分成两个 2×2 块。新程序使用块的显式极平滑公式，并在每个正 δ 上与未使用块结构的 Gram 特征分解公式交叉检查。将 `s=δt` 后解析延拓到 δ=0，利用高精度数值微分计算 `t₁` 和完整方向矩阵 `M_R`，避免 float64 近等数相减。随后在 80、100 位分别从输入有理数重新计算，不把 float64 常数转换成高精度输入。

这里的“80 位”“100 位”和两次结果的一致性表示工作精度与数值一致性，不提供向外舍入的误差包络。并非宣称所有打印出的 60 位都已获严格认证。正文只展示 10–12 位，与理论极限比较时仍显式保留有限 δ 的偏差。

## 4. 数字—数据—代码对应表

当前主文和补充中的这些表均为合成矩阵实验，没有外部训练数据库，也不需下载数据集。

| 文稿对象 | 确定性输入与生成函数 | 快照 |
|---|---|---|
| regular / strict / boundary 三分支 | `fixed_data()`；文稿给出的三个 2×2 有理矩阵 | `table_fixed.tex`，JSON `fixed` |
| Huber、L=1/2、p=3 尾 | `profile_data()`；同一 boundary 矩阵，仅平滑 profile 改变 | `table_profiles.tex`，JSON `profiles` |
| corank-two 两中心 | `corank_data()`；对角 4×4，核块奇异值 1 与 1/2 | `table_corank.tex`，JSON `corank` |
| 非对角一阶常数及完整方向 | `non_diag(dps)`；正文固定有理 4×4 矩阵 | `table_nondiagonal.tex`，JSON `nondiagonal` |
| compact-root cubic/quadratic | `compact_data()`；对角 3×3 | 数字宏，JSON `compact` |
| 二次消去 quintic/quartic | `quintic_data()`；固定有理 4×4 | `table_quintic.tex`，JSON `quintic` |
| critical / strict-side / oscillatory crossover | `crossover_data()`；调用原有 100 位 Decimal 闭式函数 | `table_critical.tex`、`table_strict.tex`、`table_oscillatory.tex` |
| 实际输出的可行修复与 gap | `verify_certificate_example.py`；Xie 论文背景的 2×2 反例与八步系数 | `certificate_example_results.json` |
| 新增有限消去判据 | `verify_finite_cancellation.py`；Fraction 精确矩与 100 位标量根 | 对应逐脚本日志 |
| 新增 boundary 方向必非零 | `verify_boundary_noncancellation.py`；固定 2×2 边界模型 | 对应逐脚本日志 |

生成的 LaTeX 位于 `submission/generated/`。主文、技术补充都直接读取这些文件；不能只复制两个 `.tex` 主入口而漏掉 `generated/`。

## 5. 本次执行的环境与边界

本次原版与修订版使用 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、mpmath 1.3.0。最终退出状态、精确平台字符串、脚本哈希以 `results/verification_latest.json` 为准。环境中最初缺少 mpmath 是已声明依赖未安装，不是代码缺文件；补装后原版 10 项通过。

新增 4 项是表格快照检查、实际方向证书例、有限消去检验、boundary 方向非消去检验。原 10 项的数值判据没有通过放宽容差来获得通过。新增高精度流程修复了文稿数字与随包代码之间的缺口；它不替代全部原有独立检查。

Python 优化模式的三项实际检查为：`python -O verify_all.py`、`PYTHONOPTIMIZE=1 python verify_all.py`、`python -O verify_rate_constants_exact.py`；均以非零退出码拒绝执行。

## 6. 与 Xie 论文及 GPU 实验的关系

Xie 等的 *Controlled LLM Training on Spectral Sphere* 提供应用动机；本次补入的实际输出诊断明确引用该文，使用其八步多项式系数，但保持独立 float64 数值例身份。这些文件并不是 Xie 的完整训练复现包，也没有生成其训练表格。

如果论文仍定位为矩阵分析，则 GPU 训练不是“缺少了所以理论不能成立”的必要条件。若另外希望主张完成证书会改变真实训练、提高性能、降低耗时，必须新增受控训练实验。没有必要先重复对方所有大模型实验才能检验这些较窄问题；但要宣称“复现了 Xie 的某一张表”，则必须匹配该表的数据处理、模型、token 数、初始化/随机种子、优化器实现、精度、批量、学习率、评测设置及代码版本，并说明任何偏离。
