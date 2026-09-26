# SSO相关矩阵分析研究：2026-09-19完整修订版

基于8月4日第十二版修订；原稿独立归档，没有覆写。推荐阅读顺序：

1. `02_Route_and_Writing/VALIDATION_AND_DELIVERY_20260919_ZH.md`：这次实际完成了什么、怎样验证、还有什么未完成。
2. `02_Route_and_Writing/REVISION_DECISIONS_20260919_ZH.md`：所有可恢复修改意见的逐项裁定，包括撤回错误批评、保留研究边界。
3. `02_Route_and_Writing/RESEARCH_ROUTE_AND_WRITING_20260919_ZH.md`：研究定位、主线、各节写作任务与GPU路线。
4. `01_Manuscripts/main.pdf`和`supplement.pdf`：SIAM类排版的主稿30页与补充15页；便携article版本使用相同正文但页数/编号不同，请成对阅读。
5. `03_Source/README.md`：完整LaTeX/参考文献、JSON生成表、14项数学验证脚本、环境和运行日志；包含SIAM完整宏包安装器及无需该宏包的article编译方式。
6. `04_GPU_Experiments/README_START_HERE_ZH.md`：GPU代码、数据官网与下载配置、两卡任务分配、独立小模型路线和Xie官方代码路线。
7. `05_Base_Paper/`：用户提供的Xie基础论文v3；`06_Review_Inputs/`为原复核输入，不代表全部赞同。

`CHANGES_FROM_20260804.patch`是可读文本源码差异；`SHA256SUMS.json`覆盖包内文件。原版另有独立ZIP。

本次14项数学数值检查通过；新推进包括有限抵消分类及方向过渡首项的非零性。普通浮点/高精度计算仍不是形式化或区间认证。真实GPU训练未运行，代码内明确记录已完成的CPU工程/真实TinyStories微型流程及尚未执行的CUDA/OLMo/FineWeb正式实验。

这是一份完整研究修订稿，尚不是已压缩至通常20页的投稿短版。作者信息仍匿名；没有投稿或公开发布。研究贡献不以指控Xie伪造为基础，目前证据不足以认定其伪造实验。
