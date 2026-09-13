# 20260914T0015-teacher-gap-construct-c1-128-r6

阶段：W2B C1 构造。

目的：仅在 `20260914T0015-teacher-gap-screen-c1-128-r6` 全链通过后生成机器候选；全部产物仍须人工复核。

## 结果

**FAILED CLOSED（canary）。** Slurm `57580365` 于 2026-09-13 18:24:39–18:31:24（CEST）在 `lrdn0374` 使用 4×A100，固定提交 `1a59505`，退出码 `1:0`。full 阶段未启动。

- canary 8 条全部完成，零截断、零 runner error；8/8 都触发 fallback，0/8 被恢复，严格解析率 0，低于 0.75 门槛。
- 最终拒绝原因：`correct_answer_not_reference_alias=4`、`obvious_answer_leakage=3`、`invalid_distractors=1`、`fallback_decision_mismatch=1`（一条可有多个原因）。

逐条审计确认，失败并非单一 prompt 格式问题，而是暴露了两层契约冲突：

1. **答案基数误读。** M3VQA 官方 `answers` 是需要共同预测的答案列表，`answer_evals` 是每个答案的别名集合。现行 candidate 丢弃 `answer_evals`，并把 `answers` 误称为 aliases；validator 又要求 `correct_answer_text` 等于列表中的单个元素。多答案题因而无论输出完整答案集合还是单个元素都无法同时满足题意与当前契约。
2. **guide 与短答案泄漏判定过粗。** 对品牌、犬种等题，必要的中间节点可能与某个 atomic answer 同词；现行词面检测会将其一律视为 final-answer leakage。此处需要先定义完整答案对象和推理节点，再判断 guide 是否把**完整最终结论**提前给出，不能简单删除安全门。

此外，fallback 当前会锁定 primary 中已经被判定无效的 `correct_answer_text`，使重试无法修复该字段；该机制错误应在新版本中改为只锁定经验证的高层决定和由输入确定的 canonical target。

结论：停止 r6 fullpool 扩量。下一版必须先恢复 `answer_evals`，显式区分单答案与多答案，重新运行 C1 screen；旧 screen 的 13 条候选全部降级为无效机器结果，不进入人工审核。
