# 20260913T2320-teacher-gap-screen-c1-128-r4

阶段：W2A 候选筛选最小重试。

目的：在固定 C1 128 条上验证 r4 的严格 blind/audit 生成；主尝试失败时仅运行一次紧凑回退，并完整保留两次尝试 provenance。

## 配置边界

- 4×A100，48 小时上限；Qwen3.5-9B-base 固定 revision；greedy、thinking=false。
- primary 768 token，compact fallback 512 token；不使用 repetition/no-repeat 约束。
- fallback 仅由主尝试截断或严格 stage contract 失败触发；gate 从 raw output 独立复算。
- canary 与 full gate 均通过后才进入 audit/构造；不在本 run 扩大 fullpool。

## 结果

待提交。提交时 HEAD 将作为 `MOPD_EXPECTED_CODE_COMMIT` 固定，并由每个 runner 在载入模型前核验。
