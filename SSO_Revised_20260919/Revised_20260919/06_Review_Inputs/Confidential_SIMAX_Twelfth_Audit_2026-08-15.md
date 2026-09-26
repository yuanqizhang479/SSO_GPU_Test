# 第十二版保密综合审计与 Go/No-Go 判断

**审计日期：** 2026-08-15  
**审计对象：** 主文 22 页、技术补充 11 页，以及此前研究梳理与审稿核验材料  
**时间增量窗口：** 2026-08-05—2026-08-15（采用比“最近十个自然日”更保守的闭区间）

## 0. 先给结论

### 最终判定

| 判断层面 | 结论 | 含义 |
|---|---|---|
| 核心数学正确性 | **GO** | 未发现推翻主定理、改变主要结论或需要重构论文的错误 |
| 新颖性 | **Conditional GO** | 十天窗口内无直接撞题；但现有文献定位有明显遗漏，原样投稿存在可避免的审稿风险 |
| 当前收到的投稿包 | **HOLD / 暂时 NO-GO** | 论文声称附带复现脚本，但实际只收到两份 PDF；且有三处有限尺度数值需统一重算 |
| 完成本文列出的 P0 修改后 | **GO for SIMAX** | 论文与 SIMAX 的矩阵理论、分析及潜在应用范围匹配 |
| 包装为优化器性能或外层收敛论文 | **NO-GO / NO-CLAIM** | 现稿没有外层复杂度、下降保证、发生频率或学习性能证据 |

一句话决策：**理论核心可以投；当前文件包先不要提交。补证明展示、补近邻文献、重跑并附上源码后，建议以 SIMAX 为第一目标投稿。**

## 1. 保密边界与检索方法

本次公开检索只使用了通用、已公开的主题词组合，例如 nuclear/trace norm、affine/subspace approximation、subdifferential、rank-deficient polar decomposition、smoothing、matrix nearness、Muon 等。没有把以下内容提交给任何公开搜索服务：

- 未公开的论文句子、定理陈述或自定义术语；
- 特殊矩阵、数值常数、实验参数和内部研究路线；
- 能反推出作者研究偏好或下一步计划的查询组合。

因此，网络侧只能观察到普通学术检索行为，不能从查询文本还原本文的具体贡献链或研究倾向。

## 2. 当前研究内容的重新梳理

### 2.1 论文真正研究的对象

当前稿件已经从早期较宽的优化器/外层算法档案，收缩为一个明确的矩阵分析问题：研究矩阵仿射直线上核范数最佳逼近，并分析最优点落在降秩集合时的证书几何和光滑正则化路径。

逻辑主线可压缩为四层：

1. **完整最优性图，而不是单个极因子。** 先把一维凸函数的次微分写成精确区间，说明紧致极因子只是完整核范数次微分中的一个选择。
2. **等式约束证书切片。** 分类该切片何时非空、何时唯一，并给出最小 Frobenius 范数证书的显式 water-filling 结构。
3. **硬中心与光滑中心。** 区分 Huber 所选的最小 Frobenius 中心和 pseudo-Huber 一般所选的另一中心，并刻画二者何时一致。
4. **奇异平滑渐近。** 在规则点、严格降秩点、紧致极根和横截边界上给出位移与返回方向的精确阶、常数、高阶消去和统一 crossover。

最后的开放失效集结果承担“为什么不能把集合值最优性图替换为单值紧致极方程”的结构性反例，而不是优化器性能结论。

### 2.2 与此前研究档案的关系

第十二版相对于此前较长档案做了正确的收缩：

- 删除了大部分外层 SSO、语言模型训练、curl、发生频率和性能叙事；
- 将主定理推广到一般矩阵法向、任意形状和任意余秩的严格奇异点；
- 正确区分了最小 Frobenius 中心与 pseudo-Huber 中心。早期“pseudo-Huber 总选 canonical/min-Frobenius completion”的表述只在较特殊情形成立，第十二版已修正；
- 保留优化器只作为动机和一个可解释的应用入口；
- 明确不声称外层收敛、复杂度或学习性能改进。

这是一次实质性的定位改善。当前版本不应再被称为“SSO 修正论文”，而应被称为“矩阵最佳逼近、证书几何与奇异平滑论文”。

## 3. 论文基调与投稿定位

### 3.1 推荐基调

最稳妥的一句话定位是：

> 本文研究矩阵直线上的核范数最佳逼近，刻画降秩交叉处的等式约束证书切片及其硬/光滑中心，并导出正则化路径的奇异渐近律。

贡献顺序建议固定为：

1. 证书切片的几何与唯一性；
2. 显式中心选择；
3. 奇异扰动的阶、常数与 crossover；
4. 紧致极单值方程的开放失效集；
5. 优化器动机与可认证内层工作流。

### 3.2 必须避免的语气

不要把以下内容写成新颖性：

- 一般仿射核范数最小化；
- 核范数的标准对偶和次微分公式；
- 一般解唯一性或核范数球的平坦性；
- 极因子在接近降秩时的病态性；
- 一般 pseudo-Huber/Fenchel 平滑；
- “传统 LMO 错了”。正确说法是：**LMO 和集合值对偶仍然精确，失败的是把完整最优性图替换成单个紧致极选择。**

### 3.3 SIMAX 匹配度

[SIMAX 官方范围](https://www.siam.org/publications/siam-journals/siam-journal-on-matrix-analysis-and-applications/)明确包括矩阵/张量理论、分析、应用和计算，也接受具有潜在应用影响的理论论文。本文的矩阵范数、极分解、奇异值扰动、最佳逼近和计算验证均在范围内。

风险不在 scope，而在审稿人是否认为：

- 一维矩阵直线足够重要；
- crossover 定理的证明足够完整；
- 与既有核范数唯一性、二阶变分分析和 matrix nearness 文献区分得足够清楚。

因此投稿策略仍建议 **SIMAX 首投，LAA 作为稳健备选**。

## 4. 文献综述总图

### 4.1 已有文献覆盖的七层背景

| 文献层 | 既有覆盖 | 本文应主张的增量 |
|---|---|---|
| 最佳逼近与 Birkhoff–James 正交 | Singer、Watson、Ziętak、Bhatia–Šemrl、Grover、Johnston、Grover–Gupta | 不主张一般最优性条件；主张特定证书切片和奇异路径 |
| 仿射核范数最小化 | [Liu–Vandenberghe 2010](https://epubs.siam.org/doi/10.1137/090755436) 已直接研究 affine matrix-valued nuclear norm minimization | 一维降秩交叉的显式结构、中心和精确渐近 |
| 核范数球平坦性与唯一性 | [Hoheisel–Paquette 2023](https://link.springer.com/article/10.1007/s10957-023-02167-7)、[Fadili–Nghia–Phan 2025](https://link.springer.com/article/10.1007/s10957-025-02723-3) | 区分原问题唯一性与对偶证书切片唯一性 |
| 非光滑奇异值分析 | [Lewis–Sendov 2005](https://link.springer.com/article/10.1007/s11228-004-7197-7) | 不主张一般子微分；主张沿矩阵直线的显式分支和常数 |
| 二阶变分分析 | [He–Kan–Song 2026](https://link.springer.com/article/10.1007/s11228-026-00804-7) 已给出核范数二阶 epi 导数 | 强调本文的具体正规形、分数幂和高阶消去 |
| matrix nearness / 子空间最佳逼近 | [Roy 2025](https://arxiv.org/abs/2505.12059)、[Faber–Liesen–Tichý 2025](https://arxiv.org/abs/2506.09687)、[Wang–Li–Lim 2026](https://arxiv.org/abs/2605.30181) | 等式切片的中心公式与奇异平滑渐近仍是本文核心 |
| 极分解扰动与平滑 | Higham、[Mathias 1993](https://epubs.siam.org/doi/10.1137/0614041)、Gawlik–Leok、Nesterov、Lobos、Move-on-Muon、Softsign | 一般病态性/平滑不新；特定驻点方程失效及约束中心新 |

### 4.2 现稿必须补入的最接近文献

按优先级，至少应在主文引言中直接讨论：

1. **Hoheisel–Paquette 2023。** 核范数球平坦性、精细子微分和仿射约束唯一性，与本文主题最近；必须说明本文研究的是对偶证书切片、中心选择及路径渐近。
2. **Liu–Vandenberghe 2010。** 一般“仿射矩阵值函数的核范数最小化”早已存在，所以问题形式本身不能成为 novelty。
3. **Roy 2025。** trace-class 到有限维子空间的距离和最佳逼近；应区分一般支撑面/annihilator 条件与本文的有限维显式切片。
4. **He–Kan–Song 2026。** 核范数二阶 epi 可微与二阶最优性；应防止本文被读成遗漏最近的二阶变分理论。
5. **Lewis–Sendov 2005 Part I/II。** 现稿只引 2001 年光滑谱函数论文，不足以覆盖所用非光滑奇异值几何。
6. **Fadili–Nghia–Phan 2025。** radial/descent cone、唯一性和稳定性；明确本文不是重新证明原问题唯一性。
7. **Faber–Liesen–Tichý 2025 与 Wang–Li–Lim 2026。** 两者可放入 matrix best approximation/nearness 段，说明本文更窄但给出精确的降秩局部结构。
8. **Mathias 1993（可连同 Kenney–Laub 1991）。** 承认极分解接近降秩时的敏感性是经典事实。

## 5. 2026-08-05—08-15 文献增量

截至 2026-08-15 当前可公开索引的记录，**没有发现直接覆盖本文核心组合的新论文**。相邻工作如下：

| 日期 | 工作 | 与本文关系 | 决策影响 |
|---|---|---|---|
| 08-05 | [Do–Dereich–Jentzen, On MUON optimization](https://arxiv.org/abs/2608.04607) | 有限 Newton–Schulz/Polar Express、Muon 非收敛和误差分析 | 优化器背景；不碰矩阵直线证书几何 |
| 08-05 | [Singh, The Loss Does Not See the Basis, but Adam Does](https://arxiv.org/abs/2608.05136) | 明确指出精确 polar/msign 在降秩附近 Lipschitz 商发散，有限 Newton–Schulz 则连续 | 若保留优化器动机应引用；不构成核心撞题 |
| 08-06 | [Solonko–Molozhavenko–Rakhuba, Muon on the Stiefel Manifold](https://arxiv.org/abs/2608.06218) | Stiefel 上精确闭式 Muon 更新和收敛保证 | tangent-LMO 语境的近邻；无核范数直线渐近 |
| 08-12 | [Zhou et al., MOON](https://arxiv.org/abs/2608.11749) | 谱—核范数几何的多目标更新与外层收敛/实验 | 只属应用背景 |
| 08-13 | [Weighted Nuclear Elastic Net](https://arxiv.org/abs/2608.12838) | 核范数正则化与统计 oracle inequality | 同词但问题不同 |

窗口外但必须注意的边界项：

- [Liu et al., Smoothed Matrix-Polar Spectral Gradient Flows](https://arxiv.org/abs/2608.01911)，2026-08-03。它直接讨论 canonical polar 在降秩处失光滑/病态以及平滑谱势。虽然早于本稿日期且不研究等式证书切片，但引言应补引。

因此，十天增量的结论是：**没有 novelty collision；有若干必须吸收的语境文献。** 由于 2026-08-15 是周六，周末提交可能尚未进入下一次 arXiv 公告，正式提交前应在下一公告批次后再做一次轻量复查。

## 6. 数学推导审计

### 6.1 主文定理链

| 模块 | 审计结论 |
|---|---|
| Thm. 2.1 | 强对偶、完整解面、紧性和乘子界正确 |
| Thm. 2.2 | 标量次微分区间、紧致极根判据、无根时唯一性正确 |
| Thm. 3.1–3.2 | (2\times2) 失效族、全空间相对开放性和尖锐增长界正确 |
| Thm. 3.4 | 证书切片的可行性及唯一性分类完整，端点和矩形情形处理正确 |
| Thm. 3.5–3.6 | 最小 Frobenius completion 的 water-filling、clipping 与 partial-isometry 判据正确 |
| Lem. 4.1–Thm. 4.3 | pseudo-Huber Fenchel 对、强凸性、唯一光滑 oracle 和 (R)-center 公式正确 |
| Cor. 4.4 / Prop. 4.5 | (K_F=K_R) 的条件与一般 profile 中心正确 |
| Thm. 5.1 | full-rank (O(\delta^2)) 位移的符号和系数正确 |
| Thm. 5.2 / 5.4 / Prop. 5.5 | 任意形状/余秩严格奇异点的 blow-up、解析展开、(t_1,M_R) 正确 |
| Cor. 5.6 | cubic/quadratic 与二次消去后的 quintic/quartic 层级正确 |
| Thm. 5.10–Cor. 5.11 | pseudo-Huber (2/3) 律、一般 (p/(p+1)) 和有限饱和线性律正确 |
| Thm. 5.14–Cor. 5.15 | 统一 cubic crossover 与方向级极限可信且与重导一致；证明展示需加强 |

关键基础式独立重导一致：

\[
\partial\phi(\lambda)
=a+\{\langle C,K\rangle:\|K\|_2\le1\}
=[a-\|C\|_*,a+\|C\|_*].
\]

严格奇异情形的 blow-up 目标、边界平衡

\[
\kappa x^3+\Delta x^2=\frac{\delta^2}{2\beta}
\]

以及 (2/3) 常数均与文稿一致。重复正奇异值、空 kernel block、矩形满秩、(C=0)、边界 (kappa=0) 失效和 (A^\star=0) 等边界情况没有被遗漏。

### 6.2 技术补充

S2 的集合值二分、S3 的 Bouligand tangent cone、法向敏感度、二阶径向漂移、projective 距离和最坏切向泄漏，以及 S4 的 repair 与 primal–dual gap 分解均正确。补充材料没有暗中证明外层收敛；S3.8 是法向泄漏界，S4.1 是内层 gap repair，这一边界在正文中表达得较克制。

### 6.3 三个主要证明展示风险

1. **Thm. 5.14 是当前最大审稿风险。** 统一余项界 (95)–(96) 只用 joint analyticity/grouped expansion 快速带过。建议逐项写出

   \[
   \ell'_\delta(-x,z)-\tau'(-x,z)
   =-\Delta(z)-\kappa(z)x+O(x^2+\delta^2),
   \]

   和小奇异模态修正

   \[
   \frac{\delta^2}{2\beta(z)x^2}
   +O\!\left(\frac{\delta^2}{x}+\frac{\delta^4}{x^4}\right),
   \]

   并说明所有常数在缩小后的 (z)-邻域上一致。
2. **Thm. 5.4 的解析块分解建议单列引理。** 明确 Riesz projections、矩形左右小簇和 exact block reduction 的关系。
3. **Lemma 5.13 补一句解析平方根。** 从 (mu=s^2w)、(w>0) 到 (	au=s\sqrt w) 时，写明取正的联合实解析平方根。

这些是证明完整度/可读性风险，不是已发现的反例。

## 7. 数值与代码审计

### 7.1 独立复算的总体结果

使用与作者实现无关的高精度计算，主要极限和常数全部复现：

| 检查 | 独立复算 |
|---|---:|
| regular normalized shift | (-1.7777777777776645\), 极限 (-16/9) |
| strict corank-one | (-0.3227486121645070\), 极限 (-5/(4\sqrt{15})) |
| boundary | (1.2599210498155031\), 极限 (2^{1/3}) |
| corank-two (eta_R) | (0.7150800689800525) |
| (|K_F-K_R|_F) | (0.0409961383770594) |
| non-diagonal (t_1) | (0.701053060\ldots) |
| non-diagonal (|M_R|_F) | (0.671775380676\ldots) |
| compact-root cubic coefficient | (-0.375) |
| double-cancellation quintic coefficient | (-0.125) |

这足以支持公式和极限结论，但不能替代作者源码的逐行审计。

### 7.2 需要更正的有限尺度数值

| 位置 | 文中值 | 高精度复算建议值 | 影响 |
|---|---:|---:|---|
| 补充 p.9，Huber，(delta=10^{-8}) | 1.9999999656 | **1.9999999500000019**（可报 1.99999995） | 末位/双精度根误差；不影响线性常数 2 |
| 主文 p.20，(p=3)，(delta=10^{-9}) | 1.2778860473 | **1.277886045147…** | 末位误差；不影响极限 1.2778862085 |
| 补充 p.10，direction ratio，(delta=10^{-5}) | 0.671768254 | **0.671768177074** | 小量相减精度污染；极限正确 |
| 补充 p.10，direction ratio，(delta=10^{-6}) | 0.6717760680 | **0.671774660292** | 同上；极限正确 |

最可能的原因是用舍入后的中心与 (Phi_\delta) 相减，再除以很小的 (delta)，造成灾难性消去。应由同一份高精度、精确输入脚本重生成主文和补充中的所有数字。

注意：补充 p.9 的排版实际是 (|K_F|_F=\sqrt{0.45})，PDF 正确；纯文本提取会丢失根号，不应把它列为笔误。

### 7.3 主文—补充一致性缺口

- 主文称 (p=3) profile 的完整表格和执行细节在 supplement，但 supplement 没有该行；
- supplement 说验证了 deliberately oscillatory paths，却没有给出路径定义、参数和结果表；
- 补充 p.6 的 `Theorem S4.1 (T6: ...)` 是内部标签遗留，应删除；
- supplement 的外部链接指向 `main.pdf`，与当前主文附件名不一致，单独打开时链接失效；
- “full tables and execution details” 应当真正补全，或改成更准确的表述。

### 7.4 源码审计边界

主文 Data and code availability 声称 assertion-based Python scripts 位于 supplementary source archive；本次收到的只有两份 PDF，且 PDF 无嵌入附件，也未在已有材料中找到源码包。因此目前只能说：

- 数学公式与绝大多数数值已独立复现；
- 作者实际实现**未被逐行审计**；
- 若投稿系统中没有另行上传源码包，当前 availability statement 将与实际提交物不一致。

投稿前源码包至少应包含：

1. 每张表/每个常数对应的独立 assertion；
2. 固定精度与 mpmath 高精度两条实现路径；
3. scalar root bracket 覆盖性和绝对残差检查；
4. SVD 实现与闭式 (2\times2) 公式的交叉验证；
5. 非对角 (M_R,t_1)、quintic cancellation、(p=3) profile 和 oscillatory crossover 的完整输入；
6. 一键运行入口、Python/NumPy/SciPy/mpmath 版本锁定、随机种子（如有）、输出快照和 commit/SHA256；
7. 数值秩阈值、谱间隙失败分支和“exact interval oracle”在浮点实现中的边界说明。

实现上不要在普通双精度中直接依赖 (A^\top A+\delta^2I) 形成超高阶结果；它会平方恶化条件数。优先用 SVD，并稳定计算

\[
\frac{\sigma}{\sqrt{\sigma^2+\delta^2}}
=\frac{1}{\sqrt{1+(\delta/\sigma)^2}}.
\]

quintic 检查在 (delta=10^{-5}) 时的位移约为 (1.25\times10^{-26})，需要约 40–50 位工作精度才能稳定报告十位系数。

## 8. 投稿前修改清单

### P0：未完成则不提交

1. **补入实际源码包并核对 availability statement。**
2. **从精确输入统一重跑数值。** 更正上表数值，补 (p=3) 与 oscillatory 路径的定义、表格和 assertions。
3. **展开 Thm. 5.14 的统一余项证明。** 这是最容易触发 major revision 的技术点。
4. **补最接近文献并收紧 novelty。** 至少加入 Hoheisel–Paquette、Liu–Vandenberghe、Roy、He–Kan–Song、Lewis–Sendov 2005、Fadili–Nghia–Phan、Faber–Liesen–Tichý、Wang–Li–Lim，以及 2026-08-03/05 的相邻 polar 工作。
5. **修正文内一致性。** 删除 `T6`，修复 `main.pdf` 链接，确保“full tables/details”与附件内容一致。

### P1：强烈建议同轮完成

1. 把 Thm. 5.4 的解析块分解单列为引理；
2. 在 Lemma 5.13 写明正解析平方根；
3. 在引言加入一段清晰的 novelty gap map；
4. 将优化器动机进一步压缩，避免审稿人要求外层算法实验；
5. 把 “certifiable reference workflow” 改为 “certificate-ready reference workflow”，除非代码真正实现 outward rounding/interval-safe certificate。

## 9. 可直接使用的 novelty 边界文字

建议在引言中加入类似表述：

> Affine nuclear-norm minimization, its convex duality, and the instability of polar factors near rank loss are classical. Our contribution is narrower: for a matrix line, we characterize the equality-constrained certificate slice and its hard and Fenchel centers, and derive explicit singular multiplier and direction asymptotics, including constants and a uniform strict-to-boundary crossover.

对于优化器动机，可用：

> The norm-ball LMO remains exact. The obstruction concerns only a single-valued compact-polar reduction of the full set-valued optimality graph at rank loss; no outer-complexity or learning-performance claim is made.

## 10. 最终 Go/No-Go

**今天是否直接提交当前文件：NO-GO。** 原因不是数学失败，而是四个可修复的投稿包问题：证明展示不足、近邻文献遗漏、有限尺度数值不统一、承诺的源码未在材料中出现。

**完成 P0 后是否提交 SIMAX：GO。** 核心定理链可靠，主要常数和分支已被独立重导/复算，十天窗口内未发现直接撞题，论文也与 SIMAX 范围相符。

**若希望把它改成优化器性能/外层算法论文：NO-GO。** 现有材料没有支持该基调的外层定理或实验。最强、最可守的版本就是当前的矩阵分析定位。

---

### 审计置信边界

- 数学结论：高置信；主文与补充均已逐节复核，并做了独立实例计算。
- 文献无撞题：截至 2026-08-15 当前公开索引的高置信暂定结论；下一 arXiv 公告批次后应再做一次短检。
- 作者代码：未提供，不能给“实现无误”的结论；只能给“公式和输出大体可独立复现”的结论。
