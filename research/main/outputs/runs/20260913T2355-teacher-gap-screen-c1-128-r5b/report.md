# 20260913T2355-teacher-gap-screen-c1-128-r5b

阶段：W2A C1-128 完整筛选。

目的：在 r5 audit-only canary 8/8 后验证 decision-locked blind/audit 全链。

## 结果

**FAILED CLOSED**。Slurm `57579114` 于 2026-09-13 17:50:27–17:59:30（CEST）在 `lrdn2591` 使用 4×A100，固定提交 `88a4426`；construct `57579118` 依赖取消、运行时间 0。

- blind canary：8/8，零 fallback/截断/error，gate 通过。
- blind full：126/128（98.44%），9 次 fallback 恢复 7；5 个 primary 截断。最终有 1 个 fallback 在 512 token 截断，零截断门控触发；audit 未运行。
- 失败样本 `m3vqa-data_0007820` 的 primary 本就以外部植物知识回答且 evidence 超限。r5 又锁定其不可靠 candidate answer，促使 fallback 围绕该答案循环；r4 不锁 blind 时同记录可在 60 token 内结束。

## 判断

高层 decision lock 对 audit 是必要的，对 blind 的开放答案并不可靠且会强化错误前提。下一版恢复已经实证的 r4 blind compact prompt，audit/construct 继续锁定。另将 W2 full gate 的职责调整为：所有不合格/截断记录严格进入 reject，在 canary 仍零截断的前提下允许 full 中至多 1% 的显式最终截断；W3 冻结实验的零截断门不变。这样不以宽松解析晋升样本，也避免 9,463 条候选池因单个已隔离 reject 永远无法推进。
