# 20260914T0200-teacher-gap-screen-c1-128-r7

阶段：W2A C1-128 完整答案语义筛选。运行于固定 worktree，代码提交 `21f54f0`。

## 结果

Slurm `57593306`：2026-09-13 20:00:10–20:19:55 CEST，`lrdn1117`，`COMPLETED (0:0)`，4×A100；四个 srun step 均完成。blind/audit canary 都为 8/8 且门禁通过。blind full 为 127/128，9 次 fallback、8 次恢复，最终零截断、零 runner error。audit full 为 127/128，88 次 fallback、87 次恢复；1 条最终截断并被明确拒绝，门禁允许 1/128，零 runner error。

有效 audit 标签：external_required 115、ambiguous 4、image_only R 8；最终 `selected_for_construction=8/128`，其中 2 个单目标、6 个多目标。这是完整 M3VQA 答案契约下的**机器筛选结果**，没有人工验证，也没有 S/T-Q/T-QA/Generic-T 效果。r6 的 13/128 已作废，不能比较为增减。

构造输入 SHA-256：`82f977aafb5c7153aefcfd38e2e005bf38c7d5d5b7ffc6935da2acc19cdb5843`。已依据 1033 取消原依赖 construct 作业 `57593318`（`CANCELLED`，累计运行 0），将由带新门禁的 r7b 独立运行接续。

原题候选只有 8 条，远不足 8+32+96 的 136 条冻结要求；这支持按 1031 启动图像优先重构支路，但不能判断这 128 张图是否缺少 R/PR 可供性。原始输出、解析输出、拒绝记录、逐阶段 gate summary、Slurm 日志和 sacct 均在本运行档。
