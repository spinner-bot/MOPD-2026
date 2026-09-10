# 20260910T-t02-thinking-ab

阶段：T02

目的：Thinking-mode A/B on the same 96 canary candidates (3072 budget) to measure reject->accept flip rate vs non-thinking

## 结果

待运行。

## 提交说明

对照臂：与 v2 canary 完全相同的 96 条候选（每 shard 前 24 条），thinking 开启、max_new_tokens=3072。产出后与 20260910T-t02-visual-structure-pilot-v2/canary/ 逐条对比，统计 reject→accept 翻转率，为 P 层设计决策（是否保留 thinking、是否改为双图输入、是否改写问题）提供证据。翻转率低 => 非 thinking 拒绝是稳定判断（数据问题）；翻转率高 => 非 thinking 存在 deliberation 不足（配置问题）。
