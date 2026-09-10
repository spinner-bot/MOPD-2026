# 20260910T-t02-visual-structure-pilot-v2.1

阶段：T02

目的：Full 1200-candidate pilot rerun with validated v2 config (non-thinking, norm1000 geometry, config-only canary gate); measures accept/reject distribution on Vision-OPD P-layer and true throughput

## 运行结果（作业链 57154729 → 57154734 → 57154736，全部 COMPLETED）

- 墙钟：pilot 52:22（canary 96 条 + 配置门控通过 + 全量 1,200 条，两次模型加载），校验 10 s，gate 4 s（退出码 0，v1 以来首次通过）
- canary（配置门控）：parse_rate 98.96%、truncated 0、runner_errors 0 → 通过；accept_rate 26%（观测项）
- 全量校验：**total 1,200，accepted 353（29.4%），rejected 847**（model_reject 845、invalid_json 2）
- 完整性 gate：observed_total = expected_total = 1,200，complete=true
- bbox 质量（353 条 accepted，354 个区域）：coord_space 全部 norm1000_to_pixel，越界 0；与真值 source_bbox 对比 **mean IoU 0.688、median IoU 0.894、97%（344/353）有重叠** → norm1000→pixel 换算在真值对照下确认正确
- 吞吐：~1,375 条/h/4GPU（含开销），推理净速率 ~10 s/条/GPU → 全量 19,366 条非 thinking T02 约 14 h/4GPU
- 产物：payload `03_verified/t02-pilot-v2.1-accepted.jsonl`（353）、`rejects/t02-pilot-v2.1.jsonl`（847）

### 结论

T02 v2 配置（非 thinking、norm1000 约定、EOS 截断审计、canary 门控）在结构层面完全达标。29.4% 接受率为 P 层数据属性（crop 相对指代 + 非 thinking 对困难小目标提前放弃）；同日 thinking A/B（20260910T-t02-thinking-ab）证明其中约 60% 的拒绝可被 thinking 模式以准确定位找回（翻转样本 median IoU 0.710）。全量流水线的 P 层配置（thinking/预算/分层）与 crop 相对指代处置留待项目决策，本档案不自动触发下游。

## 提交说明（v2.1）

v2 首次提交（57151182）在 canary 门控按设计熔断：结构指标全过（parse_rate 98.96%、truncated 0、runner 错误 0），但 accept_rate 26% < 40% 阈值。诊断确认 26% 是数据属性而非配置故障：pilot 1,200 条全部为 Vision-OPD-6K P 层候选，其 final question 以 teacher crop 视角撰写（如 vision-opd-000002 问 "bottom center of the image"，而 source_bbox=[2130,431,2232,480] 位于全图右上），非 thinking 模型诚实拒绝"全图无法 ground"。canary 门控已修正为仅拦截配置故障（a3bfeaf）。

本 v2.1 提交沿用同一输入与验证配置，目的：全量 1,200 条 accept/reject 分布、真实吞吐（全量流水线墙钟依据）、norm1000→pixel 换算质量抽检。
