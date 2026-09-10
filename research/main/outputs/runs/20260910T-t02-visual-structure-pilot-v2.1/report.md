# 20260910T-t02-visual-structure-pilot-v2.1

阶段：T02

目的：Full 1200-candidate pilot rerun with validated v2 config (non-thinking, norm1000 geometry, config-only canary gate); measures accept/reject distribution on Vision-OPD P-layer and true throughput

## 结果

待运行。

## 提交说明（v2.1）

v2 首次提交（57151182）在 canary 门控按设计熔断：结构指标全过（parse_rate 98.96%、truncated 0、runner 错误 0），但 accept_rate 26% < 40% 阈值。诊断确认 26% 是数据属性而非配置故障：pilot 1,200 条全部为 Vision-OPD-6K P 层候选，其 final question 以 teacher crop 视角撰写（如 vision-opd-000002 问 "bottom center of the image"，而 source_bbox=[2130,431,2232,480] 位于全图右上），非 thinking 模型诚实拒绝"全图无法 ground"。canary 门控已修正为仅拦截配置故障（a3bfeaf）。

本 v2.1 提交沿用同一输入与验证配置，目的：全量 1,200 条 accept/reject 分布、真实吞吐（全量流水线墙钟依据）、norm1000→pixel 换算质量抽检。
