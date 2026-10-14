# Phase2 工作 3：S0–S1/S3 配对翻转分析

本目录是 Phase2 的第 3 项独立工作，研究正式作业 `60011513` 中：

- S0 错误而 S1 正确的题目；
- S0 错误而 S3 正确的题目；
- S0 正确而 S1 错误的题目；
- S0 正确而 S3 错误的题目。

本工作以 S0 为基线，只分析 S1 和 S3。SC 已完成中性 system prompt 对照作用，S2 整体效果较差，因此二者不进入本工作统计、图表或结论。

## 目录用途

- `protocol.md`：本工作的固定分析口径和门禁；
- `scripts/`：可复现分析脚本；
- `derived/`：题目级翻转明细和统计结果；
- `raw/case_images/`：12 个典型案例的原图与 SHA-256 追溯清单；
- `figures/`：翻转数量、题目属性、长度差和长尾图表；
- `report/`：题目级可读分析报告；
- `runs/`：本工作每次分析的运行记录。

正式实验的 raw 输出、冻结样本和基础派生表仍以 2 号工作为权威来源：

`../2.short_output_prompt_study/`

本目录不覆盖或修改 2 号工作的 raw、derived 和 report 文件。

当前成品报告为 `report/paired-flip-analysis-report.html`。报告围绕八种 S0/S1/S3 正确性状态组织，图表和解释相邻，包含任务内部条件率、12 个可展开的原图与原始输出案例。图像来源于 CV-Bench 历史图像归档，每张都与 2 号工作冻结清单的字节数和 SHA-256 一致。完整题目级数据和复现脚本分别位于 `derived/run-60011513/`、`scripts/fetch_case_images.py` 与 `scripts/build_state_report.py`。

`report/paired-flip-analysis-report-final-version.html` 是同内容的另存副本；构建脚本在该副本存在时同步更新，避免阅读到缺少案例图片的旧版。
