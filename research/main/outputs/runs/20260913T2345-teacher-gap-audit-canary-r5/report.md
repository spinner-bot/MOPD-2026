# 20260913T2345-teacher-gap-audit-canary-r5

阶段：W2A audit-only 配置 canary。

目的：在再次运行 C1 全链前，用同一冻结 8 条验证 r5 是否保持 primary 的高层决定并只修复 schema/图结构。

## 结果

**PASSED**。Slurm `57578132` 于 2026-09-13 17:42:28–17:48:14（CEST）在 `lrdn1185` 使用 4×A100，固定提交 `2ab1ec3`。

- 8/8 严格合格，零截断、零 runner error；6/6 fallback 全部恢复。
- 最终标签为 7 条 `external_required`、1 条 `image_only`；与 primary 可恢复的高层标签一致，decision lock mismatch 为 0。
- r4 中被错误翻成 image-only 的国籍问题现保持 external-required；字符串 `"null"` 被第二次模型原始输出正确写为 JSON null，并完整保留两次 raw/provenance。
- gate 独立重建 primary/fallback prompt hash、锁定字段、4 GPU 分片与生成配置，全部通过。

结论：r5 消除了 audit fallback 的单侧语义翻转，可建立新 run_id 的 C1 screen→construct 全链；本 canary 不代替 full gate。
