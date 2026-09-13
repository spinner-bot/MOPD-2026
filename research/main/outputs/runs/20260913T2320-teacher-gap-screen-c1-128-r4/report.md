# 20260913T2320-teacher-gap-screen-c1-128-r4

阶段：W2A 候选筛选最小重试。

目的：在固定 C1 128 条上验证 r4 的严格 blind/audit 生成；主尝试失败时仅运行一次紧凑回退，并完整保留两次尝试 provenance。

## 结果

**FAILED CLOSED**。Slurm `57574779` 于 2026-09-13 17:21:27–17:31:33（CEST）在 `lrdn3398` 使用 4×A100，固定提交 `ae9600f`；依赖 construct `57574781` 自动取消、运行时间 0。

- blind canary：8/8 valid、0 fallback、0 截断、0 error，gate 通过。
- blind full：127/128 valid（99.22%）；9 次 fallback 中 8 次恢复；5 个 primary 循环截断全部在 768 token 及时转入 fallback；最终选中输出 0 截断、0 error，gate 通过。
- 唯一 blind reject 为 `m3vqa-data_0010815`：primary 截断且 invalid JSON，fallback 仍不满足 `visible_evidence`，故保持拒绝。
- audit canary：4/8 valid（50%），低于 75% 门槛，gate 正确熔断；audit full 未运行。

## 诊断

6/8 audit primary 的高层判断均为 `external_required`，但结构字段不合格：其中 4 条把 JSON null 写成字符串 `"null"`，2 条仍附带 R 图/证据。紧凑 fallback 的 image-only 示例产生明显单侧诱导：6 次回退仅 2 次结构恢复，其中 1 次把 `external_required` 翻为 `image_only`，且理由错误地把外貌当国籍证据。其余为 2 次 invalid JSON、1 次漏 explanation、1 次仍写字符串 null。

这说明 blind fallback 已解决工程长尾，但 audit fallback 不能从空白模板重新判定语义。下一版须把 primary 已解析的高层决定作为只读锁定字段，只修正 JSON 类型/非 qualifying 空字段，并同时给出完整 image-only 与 non-qualifying 模板。不得降低 gate 或把字符串 null 静默转换成 JSON null。
