# 20260913T2345-teacher-gap-audit-canary-r5

阶段：W2A audit-only 配置 canary。

目的：在再次运行 C1 全链前，用同一冻结 8 条验证 r5 是否保持 primary 的高层决定并只修复 schema/图结构。4×A100、48 小时上限；greedy、thinking=false、primary 768、fallback 512。

## 结果

待提交。通过条件为 8 条齐全、严格解析率至少 75%、最终选中 attempt 无截断/缺 EOS、零 runner error、decision lock 和两次 prompt hash 均通过独立复算。
