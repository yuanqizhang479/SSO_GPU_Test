# 第十二版保密全网复审、Go/No-Go 与美国 PhD 文书表述

**复审日期：** 2026-09-10  
**材料：** 2026-08-04 第十二版主文（22 页）、技术补充（11 页）、完整 LaTeX/Python 源码包及 2026-08-15 历史审计  
**公开检索截止：** 2026-09-10 当前可检索记录；重点增量窗口为 2026-08-17—2026-09-10

## 0. 结论先行

| 判断层面 | 结论 | 说明 |
|---|---|---|
| 核心研究方向 | **GO** | 问题明确，贡献链完整，适合定位为矩阵分析、最佳逼近与非光滑/奇异扰动研究 |
| 主定理数学正确性 | **Conditional GO** | 未发现会推翻主结果的错误；新发现一处不向后传播的 \(\beta=0\) 边界漏项，必须改正 |
| 截至 2026-09-10 的 novelty | **GO，需补文献** | 未发现覆盖“证书切片＋中心选择＋精确奇异渐近＋统一 crossover”这一组合的直接撞题；但最接近文献仍未被现稿充分讨论 |
| 标准环境下代码复现 | **GO** | 干净环境中 10/10 脚本通过，重复运行与不同线程数下输出一致 |
| 机器严格认证 | **NO-GO** | 现有实现没有区间算术、向外舍入或严格数值误差包络；Python 优化模式还会移除大量裸 `assert` |
| 当前投稿包今天直接提交 | **HOLD / NO-GO** | 尚有已知错误陈述、文献缺口、有限精度数字与复现承诺不一致、证明展示和投稿格式问题 |
| 完成本文 P0 后投 SIMAX | **GO** | 题目与 SIMAX 的矩阵理论、分析、计算及潜在应用范围吻合[^1] |
| 写入美国 PhD 申请文书 | **GO** | 应写成“正在推进的矩阵分析研究项目/在撰稿件”，突出发现—抽象—证明—复现流程，不写成已发表成果或优化器性能突破 |

一句话决策：**研究本身继续做、可以写入申请；第十二版当前文件不要原样投稿。完成范围明确的 P0 修订后，建议首投 SIMAX。**

## 1. 保密边界与检索方法

公开检索仅使用了核范数、矩阵最佳逼近、仿射矩阵函数、降秩极分解、平滑谱映射等通用学术词；没有上传或检索未公开的定理原句、特殊矩阵、内部常数、实验表格、下一步研究路线或作者身份。网络检索无法数学上证明“不存在任何相似工作”，因此本文的无撞题判断是基于多数据库关键词、arXiv 分类页、题名/摘要和近邻引用链交叉核查后的高置信判断，不是绝对不存在证明。

本轮把四件事分开判断：文献新颖性、数学正确性、代码复现性、投稿包可提交性。这样不会把“脚本运行通过”误写成“定理被代码证明”，也不会把“没有直接撞题”误写成“现稿已可投稿”。

## 2. 研究内容的准确定位

论文研究矩阵仿射直线上的核范数最佳逼近：

\[
\min_{\lambda\in\mathbb R}\|G+\lambda\Theta\|_*.
\]

当最优矩阵发生降秩时，核范数不再可微。紧致极因子只是完整次微分中的一个选择，单独用它写出的标量驻点方程可能跳过真正的最优点。论文的可守贡献不是一般核范数对偶，也不是“LMO 错误”，而是：

1. 刻画附加等式约束下的对偶证书切片，分类其可行性与唯一性；
2. 推导最小 Frobenius 范数证书的 water-filling 结构，并区分 Huber 与 pseudo-Huber 所选择的中心；
3. 在规则点、严格降秩点、紧致极根和横截边界上给出平滑路径的位移与方向渐近，包括高阶消去、分数阶速率和统一 crossover；
4. 给出紧致极单值方程在开放集合上失效的结构性解释。

最安全的论文基调是：**matrix best approximation, equality-constrained certificate geometry, and singular smoothing asymptotics**。优化器只应作为动机，不应声称外层收敛、事件发生频率、训练性能或普遍算法收益。

## 3. 文献与撞题复审

### 3.1 最接近的既有工作

| 文献层 | 既有工作已经覆盖什么 | 本文可主张的较窄增量 |
|---|---|---|
| 一般仿射核范数最小化 | Liu–Vandenberghe 已明确研究 affine matrix-valued function 的核范数最小化[^2] | 一维降秩交叉处的显式证书切片、中心和局部渐近 |
| 核范数球平坦性与解唯一性 | Hoheisel–Paquette 给出核范数球 flatness、精细次微分和仿射约束解唯一性条件[^3]；Fadili–Nghia–Phan 从 radial cone 研究凸优化与核范数问题的唯一性[^4] | 区分“原问题解唯一”与“等式约束证书切片唯一”，并研究切片上的中心选择 |
| 非光滑奇异值与二阶变分 | Lewis–Sendov 的非光滑奇异值分析是标准背景[^5]；He–Kan–Song 已给出核范数二阶 epi 导数和二阶最优性工具[^6] | 沿具体矩阵直线给出可计算的正规形、常数、分数幂与高阶消去 |
| 最佳逼近与 matrix nearness | Roy 研究 trace-class/operator-norm 距离与最佳逼近[^7]；Faber–Liesen–Tichý 研究谱范数下从矩阵子空间的最佳逼近[^8]；Wang–Li–Lim 研究一般 matrix nearness[^9] | 本文更窄，但给出核范数、单条矩阵直线、等式切片及奇异平滑路径的显式结构 |
| 极分解扰动和平滑 | 极分解在接近降秩时的敏感性是经典事实，Mathias 给出扰动界[^10]；近期工作研究降秩处的平滑 matrix-polar 映射[^11] | 新意应落在“完整集合值最优性图不能被一个紧致极选择替代”及其约束中心，而不是一般病态性 |

现稿的 `.bib` 中已经有 Hoheisel–Paquette 和 Lewis–Sendov 两篇，但主文没有实际引用；Liu–Vandenberghe、Roy、He–Kan–Song、Fadili–Nghia–Phan、Faber–Liesen–Tichý、Wang–Li–Lim 和 Mathias 尚未完整进入文献定位。这会让审稿人误以为论文回避了最接近的先行工作，属于投稿前必须修复的问题。

### 3.2 2026-08-17—2026-09-10 增量

本轮没有找到直接覆盖核心组合的新论文。最需要新增讨论的是 2026-08-26 的 **Muon with Finite Newton–Schulz**：它证明有限 Newton–Schulz 迭代把不连续极映射平滑成 Lipschitz 谱映射，并讨论精确 polar 更新可能不收敛；这与本文的动机和“平滑极映射”语境很接近，但它研究的是非光滑非凸优化收敛，不研究矩阵直线核范数最佳逼近、等式证书切片、Fenchel 中心或本文的奇异渐近定律[^12]。因此它是**必须补引的相邻工作，不是撞题**。

另一篇真正包含“平滑核范数”的 8 月增量是 Kümmerle–Masak–Stöger 的 IRLS 工作：它给出平滑核范数的全局二次 majorizer 以及低秩恢复中 IRLS 的收敛率[^13]。对象是约束恢复算法与 majorization，不是本文的一维仿射 crossing 或分数阶局部律，同样不构成重合。8 月 17 日进入公告的 low-rank adapter/Muon 工作则以线性化和 Frobenius 最小二乘近似低秩更新[^14]，最多支持“实践中会采用 compact-polar/低秩约定”的动机。

9 月的新工作主要把谱下降/Muon 用于 LoRA 切空间或卷积算子：LoRA-TSD 研究固定秩流形内的谱下降与训练收敛[^15]，Muon-C 研究卷积核的算子对齐极方向和实验性能[^16]。二者都不触及本文的证书几何与 crossover。另有一篇题名含 nuclear norm 的量化下界论文，其对象是量化矩阵乘法误差，也不相关[^17]。这些 9 月论文列在这里是为了说明排查覆盖，**不建议为了凑数全部写进论文**；8 月 17 日以后真正需要正文讨论的新增邻居主要是有限 Newton–Schulz 和 IRLS 两篇。

检索覆盖 arXiv 的 math.OC、math.NA、math.FA、cs.LG 分类以及公开期刊/DOI 页面，并对题名、摘要和近邻引用链交叉核查。综合判断：**截至 2026-09-10，novelty collision = 未发现；novelty positioning = 仍需修补。** 申请文书和论文都应避免把一般仿射核范数最小化、一般唯一性、一般极分解病态性或一般平滑原则写成本文首创。

## 4. 数学推导复核

### 4.1 已独立确认的主链

- 强对偶、完整解面、乘子界和标量次微分区间 \([a-\beta,a+\beta]\)；
- 紧致极根判据、无根时的唯一降秩最优点、低维失效族和开放失效集；
- 任意矩形形状与余秩下的证书可行性/唯一性分类；
- 最小 Frobenius completion 的 clipped water-filling 及 partial-isometry 判据；
- pseudo-Huber 与 Huber 的 Fenchel 中心、二者一致条件；
- 规则 \(O(\delta^2)\)、严格降秩 \(O(\delta)\)、边界 \(O(\delta^{2/3})\)、一般 profile 的 \(p/(p+1)\) 速率；
- compact-root 的 cubic/quadratic 及二次消去后的 quintic/quartic 分支；
- 任意余秩严格点的解析 multiplier/direction 展开；
- 统一 cubic crossover 及相对移动中心的方向极限。

未发现错误的指数、符号、首项常数或在定理假设内漏掉的退化分支。矩形 boundary、较高余秩 boundary、\(\alpha=0\) 和 \(\kappa=0\) 被正确留作范围外问题，不是隐藏反例。

### 4.2 新发现：\(\beta=0\) 的错误 remark

`sections/02_exact_geometry.tex` 的 relative-interior remark 无条件声称

\[
|a|<\beta\iff0\in\operatorname{ri}\partial\phi(\lambda^\star),
\]

并把 \(|a|=\beta\) 一律解释为只接触 exposed endpoint face。这在 \(\beta=0\) 时错误。例如取

\[
G=\operatorname{diag}(1,0),\qquad \Theta=e_1e_2^\top,\qquad \lambda^\star=0,
\]

则 \(\phi(\lambda)=\sqrt{1+\lambda^2}\)，且 \(a=C=\beta=0\)、\(\partial\phi(0)=\{0\}\)。因此 \(0\in\operatorname{ri}\partial\phi(0)\)，但 \(|a|<\beta\) 不成立；核块证书切片是整个标量球，而非 endpoint face。

修正方式很局部：把严格/端点二分限制在 \(\beta>0\)，再单独写 \(\beta=0\) 时最优性强制 \(a=0\)，标量次微分为 \(\{0\}\)，而证书切片在核块非零维时为整个谱范数球。该 remark 未被后文引用；后续严格定理的 \(|a|<\beta\) 已自动推出 \(\beta>0\)，boundary 定理的横截条件也推出 \(\beta>0\)。因此**主定理、常数、代码与摘要结论均不变**。

### 4.3 证明展示仍应加强

统一 crossover 定理的余项界是正确的，但当前从 joint analyticity 直接跳到路径一致展开，仍过度压缩。建议显式写出、并说明在缩小邻域后常数对 \(z\) 一致：

\[
\ell_s(-x,z)-\tau_s(-x,z)=-\Delta(z)-\kappa(z)x+O(x^2),
\]

\[
\tau(-x,z)=-\beta(z)x+O(x^2),\qquad \tau_s(-x,z)=\beta(z)+O(x),
\]

以及小奇异模态的

\[
\frac{\delta^2}{2\beta(z)x^2}
+O\!\left(\frac{\delta^2}{x}+\frac{\delta^4}{x^4}\right).
\]

再加 separated-cluster 的 \(O(\delta^2)\) 项即可得到文中统一余项。后续用 \(M_\delta=\Delta_\delta+\kappa_\delta\widehat x_\delta\) 归一化和推出振荡路径结论的步骤是正确的。

### 4.4 其他轻微数学/定义问题

- Fenchel regularizer 中的 \(\sqrt{1-\sigma_i^2}\) 应先作扩展值分段定义，再加谱球约束；否则球外根号在实数域无定义。
- 径向 retraction 应注明 \(W+E\ne0\)。
- smooth unit rescaling 的 \(\sigma_1\ge s_0>0\) 应明确为随 \(\delta\downarrow0\) 的一致下界；事实上 \(p^\star>0\) 可直接给出 \(s_0=p^\star/q\)。
- “no clipping occurs” 在端点处宜改成 “no truncation is required, although the constraint may be active”。
- 补充材料的一个向量出现双逗号，属于纯排版错误。

## 5. 代码与数值复现

### 5.1 执行结果

在 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、mpmath 1.3.0 的干净环境中，`verify_all.py` 所调用的 10 个脚本全部通过并正常返回；连续运行五次的完整输出哈希一致，OpenBLAS/OMP 设为 1、2、4 线程时也一致。测试覆盖了主要数值分支，包括 tall/wide、左右零空间不等、重复正奇异值、任意余秩严格点、\(p=3\) 尾、有限饱和、cubic/quintic 消去和 boundary/critical/strict/oscillatory crossover。

这支持“主要例子和渐近分支可复现”，但不能替代一般定理的证明。开放失效集、完整唯一性分类、强凸共轭和统一余项等仍是纯数学命题。

### 5.2 投稿前必须修复的复现缺口

1. **裸 `assert` 会假绿。** 在 `PYTHONOPTIMIZE=1` 下，大量科学判据被解释器删除，但总入口仍打印全部通过。应改成显式异常或 pytest，并在优化模式下拒绝运行。
2. **“复现每个表格和常数”目前不严格成立。** 补充材料把非对角实例的有限尺度 \(t_1\) 称为 50 位计算；论文值经独立 100 位 mpmath 复算是正确的，但打包脚本只做 float64，并没有 assert 该值。应新增真正的高精度脚本与表格快照。
3. **机器认证措辞过强。** 当前代码没有区间算术、向外舍入、validated SVD 或严格 gap 包络。可以说 exact-arithmetic theorem gives a certifiable construction；不能说 supplied code produces machine-rigorous certificates。
4. **有限尺度数字要统一重生成。** 既有审计发现四个显示值受双精度根求解或小量相减污染：Huber boundary、\(p=3\) 常数和两个非对角方向比值。极限常数与定理不受影响，但论文表格应从同一份高精度脚本生成。
5. **表头和索引不一致。** 主文的 “computed at the smallest \(\delta\)” 与脚本实际更小尺度不符；HTML 索引中的长文件名与包内 `main.pdf`/`supplement.pdf` 不一致。

需要统一处理的具体数值如下：

| 项目 | 文稿/现有脚本 | 独立高精度结果 | 性质 |
|---|---:|---:|---|
| Huber boundary，\(\delta=10^{-8}\) | 1.9999999656 | 1.9999999500000019 | 文稿末位受普通精度求根影响 |
| \(p=3\) boundary，\(\delta=10^{-9}\) | 1.2778860473 | 1.277886045147… | 文稿末位误差 |
| 非对角 direction ratio，\(\delta=10^{-5}\) | 0.671768254 | 0.671768177074 | 小量相减精度污染 |
| 非对角 direction ratio，\(\delta=10^{-6}\) | 0.6717760680 | 0.671774660292 | 小量相减精度污染 |
| 有限尺度 \(t_1\)，\(\delta=10^{-5}\) | 文稿 0.7010473954；脚本 float64 0.7010476154367672 | 0.70104739544355205579… | 文稿正确，但随包脚本未复现所称 50 位结果 |

这些差异均小于会改变极限判断的尺度；它们是结果生成链与复现声明的问题，不是定理反例。

## 6. PDF、LaTeX 与投稿包装

主文和补充材料按 README 指定顺序在隔离副本中成功重编；重编文本与随包 PDF 一致，最终日志无未解析引用或明显版面警告。视觉检查未见裁切、公式越界或表格重叠。

但当前主文使用自定义 10pt A4 `article` 布局，而不是当前 SIAM 投稿类。源码 README 也明确承认 22 页不能证明换用官方类后满足篇幅要求。正式提交前必须使用当时有效的 SIMAX 模板重编并重新核对页数、匿名/作者元数据、running title、补充材料文件名和交叉链接；控制规则应以 [SIMAX Instructions for Authors](https://epubs.siam.org/journal/simax/instructions-for-authors) 为准。

补充材料仍有内容承诺不完全一致：主文称 \(p=3\) 的完整表格和执行细节在 supplement，但 supplement 没有完整对应行；oscillatory path 有代码测试，却缺少足够的路径定义与结果表；内部 `T6` 标签、匿名占位提示和文件名链接也应清理。

## 7. 投稿前优先级

### P0：不完成就不要提交

1. 修正 \(\beta=0\) 的 relative-interior remark；数学严重度虽局部，但已知错误陈述不能留在投稿稿件中。
2. 在引言补入并逐一划清 Liu–Vandenberghe、Hoheisel–Paquette、Lewis–Sendov、He–Kan–Song、Roy、Fadili–Nghia–Phan、Faber–Liesen–Tichý、Wang–Li–Lim、Mathias，以及 2026-08-03/08-26 的平滑 polar 近邻工作。
3. 展开统一 crossover 的路径一致余项推导。
4. 用同一份真正高精度脚本重生成显示数字；新增非对角 50 位测试，消除 `python -O` 假绿，并把表格输出做成可比较快照。
5. 把 “numerically reproducible”“exact-arithmetic certifiable” 与 “machine-certified” 三层措辞分开；当前实现不得声称机器严格认证。
6. 使用当前 SIAM 类重编；核对实际页数与全部提交规则，补齐 supplement 的 \(p=3\)/oscillatory 细节，修复 HTML 索引、附件文件名和内部标签。

### P1：建议同轮完成

1. 修正 Fenchel regularizer 的扩展值定义、retraction 定义域和 \(\sigma_1\) 一致下界；
2. 将解析小簇/大簇分解提前抽成短引理，改善证明顺序；
3. 复用同一组 SVD 零空间基，避免代码依赖重复 SVD 在退化子空间返回同一基；
4. 添加环境元数据、输出 SHA-256/CI 和公开归档许可证；
5. 将优化器背景再压缩，避免编辑或审稿人要求本文没有承诺的外层收敛与训练实验。

## 8. 最终 Go/No-Go

**研究继续与写入 PhD 文书：GO。** 该项目能够真实展示从异常现象中抽象数学问题、构造反例、完成一般化证明、审查边界条件并建立复现链的研究能力。

**第十二版今天原样提交：NO-GO。** 原因不是主定理崩溃，而是已有一处明确的边界错误陈述，加上文献、证明展示、数值复现承诺和投稿包装尚未闭环。

**完成 P0 后投 SIMAX：GO。** SIMAX 的范围包括矩阵理论、分析、应用和计算，也接受具有潜在应用影响的理论工作[^1]。本文最匹配的版本正是矩阵分析论文；若 SIMAX 编辑认为一维线问题的广度不足，可将 Linear Algebra and its Applications 作为备选，但不建议为追逐 ML 语境把论文改写成优化器性能稿。

## 9. 美国 PhD 申请文书可直接使用的中文表述

### 推荐版：研究内容＋研究流程

> 我从谱优化中的一个异常驻点现象出发，将其抽象为矩阵仿射直线上的核范数最佳逼近问题。通过凸对偶、核范数次微分和奇异值扰动分析，我刻画了降秩交叉处对偶证书的可行性与唯一性，推导了不同平滑正则化所选择的证书中心，并分析了正则化路径的奇异渐近规律。研究中，我先用低维反例定位单值极因子近似的失效，再推广到一般矩阵情形；随后使用高精度数值、边界案例和可复现代码交叉检验公式，并通过持续的文献审查收紧创新边界。

### 更短版：适合 SOP 中的一小段

> 我研究降秩交叉处的核范数最佳逼近。通过凸对偶、次微分与奇异值扰动分析，我刻画了对偶证书的可行性、唯一性及平滑选择，并推导其奇异渐近规律；随后用反例、高精度计算和可复现代码逐项检验结论。

### 一行版：适合 CV / Research Experience

> 研究矩阵仿射直线上的核范数最佳逼近，重点分析降秩点的证书几何、平滑选择与奇异渐近，并以高精度数值和可复现代码验证理论。

### 状态措辞

若尚未正式提交，建议写“**ongoing research project**”或“**manuscript in preparation**”；只有真实投稿后才写“under review”。不要写“published/accepted”，也不要写“proved a better optimizer”或“improved LLM training”，因为当前成果并未提供这些证据。

## Sources

[^1]: Society for Industrial and Applied Mathematics, [SIAM Journal on Matrix Analysis and Applications: About the Journal](https://www.siam.org/publications/siam-journals/siam-journal-on-matrix-analysis-and-applications/).
[^2]: Z. Liu and L. Vandenberghe, [Interior-Point Method for Nuclear Norm Approximation with Application to System Identification](https://epubs.siam.org/doi/10.1137/090755436), *SIAM Journal on Matrix Analysis and Applications*.
[^3]: T. Hoheisel and E. Paquette, [Flatness of the nuclear norm sphere, simultaneous polarization, and uniqueness in nuclear norm minimization](https://arxiv.org/abs/2205.08442).
[^4]: J. Fadili, T. T. A. Nghia, and D. N. Phan, [Solution Uniqueness of Convex Optimization Problems via the Radial Cone](https://link.springer.com/article/10.1007/s10957-025-02723-3), 2025.
[^5]: A. S. Lewis and H. S. Sendov, [Nonsmooth Analysis of Singular Values. Part I: Theory](https://link.springer.com/article/10.1007/s11228-004-7197-7), 2005; Part II develops applications.
[^6]: J. He, C. Kan, and W. Song, [Twice Epi-Differentiability of Orthogonally Invariant Matrix Functions and Application](https://doi.org/10.1007/s11228-026-00804-7); preprint [arXiv:2412.09898](https://arxiv.org/abs/2412.09898).
[^7]: S. Roy, [Distance and best approximations in operator norm and trace class norm](https://arxiv.org/abs/2505.12059), 2025.
[^8]: V. Faber, J. Liesen, and P. Tichý, [Matrix best approximation in the spectral norm](https://arxiv.org/abs/2506.09687), 2025.
[^9]: R. T. Wang, C.-K. Li, and L.-H. Lim, [Generalized matrix nearness problems II](https://arxiv.org/abs/2605.30181), 2026.
[^10]: R. Mathias, [Perturbation Bounds for the Polar Decomposition](https://epubs.siam.org/doi/10.1137/0614041), *SIAM Journal on Matrix Analysis and Applications*.
[^11]: J. Liu et al., [A Continuous-Time Analysis of Smoothed Matrix-Polar Spectral Gradient Flows for Muon-Type Optimization](https://arxiv.org/abs/2608.01911), submitted 2026-08-03.
[^12]: M. Li and T. Tsuchiya, [Muon with Finite Newton–Schulz: The Smoothing Benefit in Nonsmooth Nonconvex Optimization](https://arxiv.org/abs/2608.26288), submitted 2026-08-26.
[^13]: C. Kümmerle, T. Masak, and D. Stöger, [Tight Majorizations and Convergence Rates of Nuclear Norm Minimization IRLS](https://arxiv.org/abs/2608.23765), submitted 2026-08-24.
[^14]: B. Anson, C. Houghton, and E. Milsom, [Approximate Muon with low-rank adapters](https://arxiv.org/abs/2608.14492), submitted 2026-08-14 and announced 2026-08-17.
[^15]: D. Andriianov, A. Veprikov, and A. Beznosikov, [LoRA-TSD: Tangent-Space Spectral Descent for LoRA via Muon-Style Updates](https://arxiv.org/abs/2609.02734), submitted 2026-09-02.
[^16]: J. Qing and L. Li, [Muon-C: Operator-Aligned Muon for Convolutional Kernels](https://arxiv.org/abs/2609.09676), submitted 2026-09-09.
[^17]: P. Sao et al., [A Nuclear-Norm Lower Bound for Dithered Scalar Quantization of Matrix Products](https://arxiv.org/abs/2609.05641), submitted 2026-09-04.

---

**置信边界：** 数学审计为逐节重导并由独立数值例交叉检查后的高置信结论；无撞题为截至检索日公开索引范围内的高置信结论；代码结论只适用于记录的依赖环境和普通解释器模式，不构成形式化证明或 validated numerics 认证。
