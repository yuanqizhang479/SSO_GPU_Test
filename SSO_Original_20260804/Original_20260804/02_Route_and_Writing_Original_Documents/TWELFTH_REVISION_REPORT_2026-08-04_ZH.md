# 第十二版理论审计与修改报告

日期：2026-08-04  
目标期刊：*SIAM Journal on Matrix Analysis and Applications*（SIMAX）

## 一、结论

最终判断是：**数学 Go；SIMAX 选题与贡献定位 Go；投稿状态为 Go- / 强 Conditional Go。**

本轮的重点确实不是纠错，而是把文章从“一个指定 pseudo-Huber 路径的精细分析”提升为“证书切片、正则化中心和尾部速率之间的统一理论”。附件建议中最重要的两项——平滑族与 (chi_star=0) 分支的显式五阶常数——都成立，而且已经实装。正文没有发现新的假命题。

当前唯一实质性投稿风险仍是正式 SIAM 类文件下的页数。第十二版在现有匿名 A4 审稿布局中为正文 22 页、补充材料 11 页；这不是官方类文件下的合规证明。SIMAX 当前作者说明仍以 20 页为通常上限，并要求摘要不超过 250 词；本版摘要约 179 词。[SIMAX Instructions for Authors](https://epubs.siam.org/journal/simax/instructions-for-authors)

## 二、附件建议的有效性与重要度

| 建议 | 判断 | 重要度 | 第十二版处理 |
|---|---|---:|---|
| 用平滑族解释“为什么是 pseudo-Huber” | 核心正确 | 5/5 | 新增 admissible Fenchel family；统一 (K_F) 与 (K_R) |
| Huber 共轭选择 (K_F)，严格区制斜率为 (eta_F) | 正确 | 5/5 | 已证明；同时明确 Huber 乘子可不唯一 |
| 边界指数由 (psi') 的尾部决定 | 正确且可加强 | 5/5 | 从 (p=2) 推广到任意 (p>1)，并加入有限饱和阈值 (L_psi) |
| (chi_star=0) 分支给出五阶常数和实例 | 正确 | 5/5 | 已加入 quintic/quartic 锐利展开、一般奇数阶乘子层级和 100 位验证 |
| Theorem 5.2 改用精确解析簇标架 | 有效 | 4/5 | 已改；原证明未发现反例，但闭合方式不够透明 |
| 调整贡献排序，降低 open failure 的首位权重 | 非常有效 | 5/5 | 摘要、引言、贡献列表、讨论和 cover letter 全部重排 |
| 删除孤立的 curl formula (66) 断言 | 必须 | 4/5 | 已删除，同时去掉无正文对应的随机频率入口 |
| 补 Ziętak 2017、引用 Halická 两篇 | 有效 | 3/5 | 已补 DOI，并精确区分 limiting-center 与 fractional-rate 归因 |
| 核实 Li Appendix P.5 与 Xie ICML 2026 spotlight | 有效 | 3/5 | 两处均已核实，现有引文措辞保留 |
| 正式 SIAM 宏、Zenodo DOI、MathSciNet | 真实瓶颈 | 5/5 | 未伪造完成；列为投稿前外部动作 |

## 三、本轮新增的核心理论

### 1. 一个统一的 Fenchel 平滑族

对满足正文 admissibility 条件的偶凸 (C^1) 标量 profile (psi)，定义

[
Psi_delta^psi(A)
=deltasum_{i=1}^qpsi!left(rac{sigma_i(A)}{delta}ight),
qquad
R_psi(Phi)=sum_{i=1}^qpsi^*(sigma_i(Phi)).
]

本版证明

[
Psi_delta^psi=(delta R_psi)^*,
qquad
Phi_delta^psilongrightarrow
argmin_{Phiinmathcal P^star}R_psi(Phi).
]

在固定严格秩亏极小点，任取平滑目标的乘子极小点，均有

[
rac{lambda_delta^psi-lambda^star}{delta}	o
-operatorname{sign}(a)eta_psi,
qquad
sum_isigma_ipsi'(eta_psisigma_i)=|a|,quad eta_psige0,
]

且核块中心为

[
K_psi=-operatorname{sign}(a)P_C
operatorname{diag}!igl(psi'(eta_psisigma_i)igr)Q_C^	op.
]

两个重要成员是：

- pseudo-Huber：得到原文的 (R)、(eta_R)、(K_R)；
- Huber：
  [
  R_{m hub}(Phi)=	frac12|Phi|_F^2+iota_{{|Phi|_2le1}}(Phi),
  ]
  因而得到 minimum-Frobenius / water-filled 中心 (K_F) 与斜率 (eta_F)。

这回答了“为什么是 pseudo-Huber”：它不是唯一合理平滑，而是一个具有严格单调、解析路径和特定平滑中心的成员。Huber 揭示硬 water-filling 也属于同一变分族，但其乘子一般不唯一。例如 (G=2I_2)、(Theta=operatorname{diag}(1,-1)) 时，完整 Huber 乘子集为 ([-2+delta,2-delta])，所有乘子却返回同一个证书 (I_2)。因此 Huber 适合解释中心，不应被写成 pseudo-Huber oracle 的直接替代品。

### 2. 边界指数是尾部律

若

[
psi'(x)=1-gamma_psi x^{-p}+o(x^{-p}),qquad p>1,
]

则在正文的横截 corank-one 边界假设下

[
lambda_delta^psi-lambda^star
=-operatorname{sign}(a)
left(rac{gamma_psi}{kappaeta^{p-1}}ight)^{1/(p+1)}
delta^{p/(p+1)}(1+o(1)).
]

因此 pseudo-Huber 的 (2/3) 不是证书几何单独决定的；它来自 (p=2)、(gamma_psi=1/2)。若 (psi') 在有限阈值 (L_psi) 首次饱和，则另有独立的线性区制

[
lambda_delta^psi-lambda^star
=-operatorname{sign}(a)rac{L_psi}{eta}delta+o(delta).
]

标准 Huber 的 (L_psi=1)，并可加强为 (O(delta^2)) 余项。

### 3. 双重抵消后的锐利五阶/四阶展开

在 compact-root 分支 (a=0) 中，令

[
J_star=U_rSigma_r^{-2}V_r^	op,qquad
chi_star=langleTheta,J_starangle,qquad
d_0=|C|_F^2.
]

第十一版已经得到 cubic/quadratic 主项。本版进一步证明：若 (chi_star=0)，并令

[
J_2=U_rSigma_r^{-4}V_r^	op,qquad
chi_2=langleTheta,J_2angle,
]

则

[
lambda_delta-lambda^star
=-rac{3chi_2}{8d_0}delta^5+O(delta^6),
]

[
Phi_delta-Phi_R^star
=-rac{delta^2}{2}J_star
+delta^4left[rac38J_2-rac{3chi_2}{8d_0}U_0CV_0^	opight]
+O(delta^5).
]

并且若

[
b_k=(-1)^k4^{-k}inom{2k}{k},quad
J_k=U_rSigma_r^{-2k}V_r^	op,quad
chi_k=langleTheta,J_kangle,
]

且 (a=chi_1=cdots=chi_{p-1}=0)，则完整乘子层级为

[
lambda_delta-lambda^star
=-rac{b_pchi_p}{d_0}delta^{2p+1}
+O(delta^{2p+2}).
]

这表明五阶不是孤立巧合，而是连续正交抵消层级的第二层。

## 四、对附件中需要纠正或限定的地方

1. **Huber 的乘子唯一性不能继承。** 强凸的正则化原始证书唯一，但标量乘子可以形成区间；正文已明确区分两者。
2. **一般有限饱和常数不是恒等于 (1/eta)。** 正确常数是 (L_psi/eta)；标准 Huber 才有 (L_psi=1)。
3. **补丁中的五阶有限尺度表有一组标签/数值错配。** 最终高精度值为
   [
   -0.1248207361, -0.1249833498, -0.1249983476, -0.1249998349
   ]
   对应 (delta=10^{-2},10^{-3},10^{-4},10^{-5})。
4. **relative-interior 说法已修正。** 严格情形是切向超平面穿过 kernel-block spectral ball 的相对内部；边界情形才落在 exposed endpoint face。
5. **Theorem 5.2 的旧证明没有被判定为假。** 本版改用 Riesz 簇投影、解析正交标架和精确块约化，是为了闭合 (O(s^2)) 并允许重正奇异值，而不是掩盖一个已知反例。

## 五、定位与篇幅修改

贡献排序现为：

1. 证书切片的可行性、唯一性与 water-filling；
2. Fenchel 中心族与尾部律；
3. 任意余秩严格展开、compact-root 高阶层级、边界及 crossover；
4. compact-polar open failure，作为为什么必须保留完整切片的结构性动机。

为了抵消新理论增加的篇幅，本版只移动或压缩非核心材料：两条经典基础证明被压成单段；set-valued bisection 的完整证明移入补充材料，正文保留精确三分支规则；有限尺度数值表集中到补充材料；重复的 discussion 与 conclusion 被压缩。核心 §4、§5 和 Theorem 5.4 均保留在正文。

结果是：

- 正文：22 页；
- 技术补充：11 页；
- 摘要：约 179 词；
- 正文与补充材料均无未定义引用、重复标签或版面溢出。

## 六、复现与验证

统一回归现执行十个 assertion-based 脚本。新增脚本验证：

- pseudo-Huber 的 (K_R,eta_R) 与 Huber 的 (K_F,eta_F)；
- (p=2) 和一个独立 (p=3) 尾部；
- (L_psi=1) 与 (L_psi=1/2) 的有限饱和常数；
- Huber 乘子区间与证书唯一性；
- 100 位精度下四个五阶有限尺度值、向 (-1/8) 的单调收敛；
- 完整 quartic 方向矩阵及非对角余项。

旧 `verify_theorems.py` 中正文不存在的 `curl formula (66)` 已删除。非对角余秩二实例统一采用

[
t_1=0.7010530600,qquad
|mathcal M_R|_F=0.6717753807.
]

十组回归在当前环境全部通过。

## 七、投稿前仍需完成的外部动作

1. 用投稿当日官网提供的 SIAM 类文件与 bibliography style 顺序重编译正文和补充材料，核对正式页数、running title、作者元数据与 theorem formatting；本环境未成功取得官方类文件，不能把当前 A4 页数伪装成官方页数。
2. 冻结最终源码和验证脚本到 Zenodo 或同类档案后，填写真实 DOI；不要预写占位 DOI。
3. 有 MathSciNet/zbMATH 权限时核对期刊名缩写、MR 号和最终出版元数据。
4. 由全体作者核准 originality、simultaneous submission、conflict of interest、author approval 与生成式 AI 声明。

## 八、下一篇而非本稿继续扩张的方向

本稿不建议再加新的主定理。高余秩边界会出现多个有符号小模态；非横截接触会改变指数；(sigma_r(A^star)	o0) 还需要新的双尺度一致理论。这三项足以构成下一篇，而继续塞入本稿会削弱当前清晰的 SIMAX 主线。
