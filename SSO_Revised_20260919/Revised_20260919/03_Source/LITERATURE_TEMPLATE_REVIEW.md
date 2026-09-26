# 文献与模板修订复核（2026-09-19）

结论：保留矩阵分析主线，采纳明确公式错误修正、最接近文献定位、真实数值输出度量和适用范围限定。不同意把所有近邻都列成“必须补引”，也不同意把 GPU 训练或重做 Xie 全套实验作为证明本文数学结论的必要条件。本轮没有证明不存在其他近邻，也没有完成全部既有文献的逐定理优先权审查。

## 本轮已落实

- 引言公式(1)将距离等式放回最小值；不能把每一个逐点核范数都等同于距离。
- Xie 的基础论文明确为 *Controlled LLM Training on Spectral Sphere*, arXiv:2601.08393v3，2026-03-05；准确定位为 §3.2 和 Appendix A.2（Theorem A.3 是定理编号，不是附录编号）。删除本轮未核实的“ICML spotlight”出版状态，保留准确版本引文。
- 不再称 tangency slice 本身为“新对象”；把贡献收窄到明确分类、中心比较、展开系数、消去条件和统一过渡。
- 明确 rank-one SSO 法向对应的硬/pseudo-Huber 中心重合；一般法向中心差异不能当作 SSO 必然具有的现象。
- 加入有限 Newton--Schulz 与精确极因子/Fenchel 路径不同的限定。实际输出切向残差与换成另一方向后的残差不同；不能交换对象。
- 不作训练优越性、失效频率或实验造假结论；应用实验与本文矩阵理论的证据链分开。
- AI 声明不再替人类作者保证“已经独立检查全部来源和证明”；只准确披露辅助工作，并保留人类作者责任。

## 与最接近工作的具体关系

| 一手来源 | 已核对内容及本文修改 |
|---|---|
| [Liu–Vandenberghe，SIMAX 31(3),1235–1256](https://epubs.siam.org/doi/10.1137/090755436)；[作者原文](https://www.seas.ucla.edu/~vandenbe/publications/nucnrm.pdf) | 一般仿射核范数最小化已存在；原文式(6)还给 Huber/Frobenius 正则的谱球对偶表达。因此不能把一般问题、一般对偶或 Huber 平滑本身作为新增贡献。采用电子发表年2009（在线2009-11-04；卷期跨2009–2010）。 |
| [Hoheisel–Paquette，2023](https://link.springer.com/article/10.1007/s10957-023-02167-7)；[公开原稿](https://arxiv.org/pdf/2205.08442) | 原稿 Theorem3.4、Corollary4.1研究核范数球平坦性和仿射约束最小解唯一性。本文区分 approximant 唯一与附加超平面下 certificate 唯一，不泛称首个唯一性理论。 |
| [Lewis–Sendov，2005 Part I](https://link.springer.com/article/10.1007/s11228-004-7197-7) | 非光滑奇异值函数的次微分框架是经典基础。原稿已有 BibTeX，本轮补入实质定位；Part II 一并作为应用背景引用。没有声称重新验证其全文每一公式。 |
| [He–Kan–Song，2026](https://link.springer.com/article/10.1007/s11228-026-00804-7)；[arXiv v2](https://arxiv.org/html/2412.09898v2) | 已读§3及Corollary3.6：秩亏核范数二阶 epi 导数已有明确公式。本文特殊直线上的中心及奇异参数渐近应与一般二阶变分工具区分。使用期刊正式信息：34卷，Article20。 |
| [Mathias，1993](https://epubs.siam.org/doi/10.1137/0614041) | 极分解扰动与小奇异值相关的敏感性已有经典界。此处只作背景引用；不能声称一般病态性新发现。 |
| [Skewon，2026-08-06 v1](https://arxiv.org/html/2608.06218v1) | Proposition1式(7)明确列出核块解族；Theorem2用范数保持补全联系精确 Stiefel 联合谱范数 LMO。不能写所有切空间谱 LMO 无闭式，或核补全首次出现。其特殊 Stiefel 结构不同于本文一般单超平面矩阵线。 |
| [Intrinsic Muon，2026-05-10 v1](https://arxiv.org/html/2605.09238v1) | Appendix H.3 明确 Stiefel block product norm 与 joint ambient norm 不同；P.5使用谱球标量方程。比较时按可行集区别。 |
| [MCSD，2026-08-13 v2](https://arxiv.org/abs/2601.21487v2) | 已下载并读现行PDF；Remark3.3附近给投影商范数，其单位球是投影后的环境球。更新题名与版本，不把该集合默认为 ball∩tangent。PDF封面写Aug14，arXiv替换日期Aug13，引用说明用后者。 |
| [Li–Tsuchiya，有限NS，2026-08-26 v1](https://arxiv.org/html/2608.26288v1) | 特别核对§6限制：证明使用预先固定缩放和经典Taylor系数；作者明确不直接覆盖实际调优quintic/随动Frobenius归一化。因此不能用其定理替SSO实际有限精度代码认证。 |
| [Xie等，2026-03-05 v3](https://arxiv.org/abs/2601.08393v3) | 本文直接应用动机；没有把其GPU表格重新运行、验证或否定写入数学主稿。 |

9/10 审计提出 Roy、Fadili–Nghia–Phan、Faber–Liesen–Tichý、Wang–Li–Lim、IRLS、smoothed polar 等一长串必须补引，作为“继续查阅近邻清单”合理；作为每篇都必须进入主文的硬门槛过强。本轮优先补最直接的仿射问题、唯一性、二阶变分和有限NS邻居。未把泛 matrix-nearness 文献堆入正文，也未为此声称其定理全不重合。

## SIAM 模板与可重编译性

已核对 [SIMAX作者说明](https://epubs.siam.org/journal/simax/instructions-for-authors)、[SIAM作者资源](https://epubs.siam.org/journal-authors#macros) 和 [官方模板说明](https://epubs.siam.org/pb-assets/macros/standard/docsiamart.pdf)。当前类为 `siamart251216.cls`（2025-12-16，v1.4.8）。SIAM强烈建议使用模板；官方说明明确它是可选项。SIMAX有20页政策，但可考虑较长稿，不应把20页解释成绝对禁止。

主/补充已改用官方类入口、官方数字引文样式、官方定理环境、关键词及MSC环境；不伪造作者姓名/单位。未靠缩小页边距制造合规页数。匿名稿仍需作者真实元数据和最终投稿审定。补充材料用官方SM编号。主文保留核心证明，因为SIMAX一般不将补充材料视为完整同行评审对象。

官方整包在本环境的直接下载返回403。通过公开论文源包 `https://arxiv.org/src/2602.09198v1` 取得同名类与bst用于本地排版核对，未修改这些第三方文件。其许可要求与完整宏分发包一同分发，不允许孤立分发类文件，因此不将这两个文件单独放入交付源码。此处不能声称已取得完整官方分发包，亦没有绕过下载限制。

为使源码仍可重编，提供：

1. `python download_siam_template.py`：获取完整官方包，校验8个必需文件存在后保留完整分发；也支持 `--zip PATH` 导入用户已下载的官方ZIP。
2. `python build_pdfs.py --format siam`：使用上述完整分发生成官方格式；支持 `--template-dir PATH` 指向用户已有模板。
3. `python build_pdfs.py --format review`：不依赖第三方SIAM类，使用 `main_review.tex` / `supplement_review.tex` 与完全相同正文生成普通article审阅版。只有版式、编号及参考文献样式不同；不将它的页数冒充官方格式页数。

TeX依赖为常规TeX Live或MiKTeX（amsmath、mathtools、ntheorem、algorithm、cleveref、xr-hyper、hyperref等；Debian完整环境可用 latex-extra、fonts-recommended、science 包）。本地缺少algorithm包时，从CTAN公开algorithms分发生成供本地排版，未修改SIAM类。最终PDF页数、警告及视觉检查由最终编译报告记录。

## 仍需作者决策/不能自动声称完成的事项

- 真实作者、单位、致谢、资助、利益冲突、公开归档许可及最终AI披露由作者核准。
- 全篇证明与文献优先权仍需人工最终审阅；本轮修订不构成录用或数学无误保证。
- 超20页需要压缩或在cover letter解释；不能为了页数把核心证明移到通常不完整送审的补充材料。
- 数值脚本通过不是机器严格认证。GPU预备代码没有在本地完成真实GPU训练，亦没有生成可写成已完成的训练表格。
