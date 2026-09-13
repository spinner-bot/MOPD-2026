# 20260913T2220-teacher-gap-screen-c1-128-r3b

阶段：W2A 候选筛选配置验证。

目的：在固定提交上验证关闭 thinking、greedy、1536 token、无 repetition/no-repeat 约束的严格 JSON 筛选。

## 结果

**FAILED CLOSED**。Slurm job `57568703` 于 2026-09-13 16:47:13–16:57:42（CEST）使用节点 `lrdn1867` 的 4×A100 运行；依赖的 construct job `57568704` 因 `afterok` 未满足自动取消，运行时间为 0。

- blind canary：8/8 schema-valid，EOS 8/8，截断 0，runner error 0；通过。
- blind full：128/128 记录完整；旧提交契约 valid 122（95.3125%），invalid JSON 6，其中 5 条在 1536 token 截断；runner error 0。
- 旧 completion gate 的 parse rate 高于 90%，但“任何截断即失败”的配置门正确触发，audit 未运行。
- 5 条截断输出均在 `visible_evidence` 数组中循环重复观察，证明删除 `no_repeat_ngram_size` 消除了字段名系统性破坏，却没有消除长尾自由文本循环。
- 另有 1 条完整 EOS 输出因漏引号导致字段被吞入 evidence 数组，无法解析。

使用隔离分支 `codex/w3-hardening` 的严格内容契约事后重解析：119/128 合格（92.96875%）；除上述 6 条外，另拒绝 1 条重复 evidence 和 2 条超过 1–4 项边界的 evidence。该数值用于诊断下一版门控，不改变本次提交时 gate 的事实。

## 判断

r3b 回答了两个工程问题：

1. r2 的 0/8 JSON 退化确由 no-repeat n-gram 约束造成；移除后 canary 恢复为 8/8。
2. 单纯把 token 上限升至 1536 仍不能解决约 3.9% 的确定性循环，因此不能直接扩大 fullpool。

下一版必须保留首轮原始输出，并对截断或 schema-invalid 样本使用显式记录的紧凑 fallback；不得恢复会改写 JSON key 的 repetition constraint，也不得做模糊字段修复。严格内容契约与 W2 provenance gate 合并后，再以新 run_id 重跑 C1 128。
