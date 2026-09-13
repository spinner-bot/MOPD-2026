# 20260914T0015-teacher-gap-screen-c1-128-r6

阶段：W2A C1-128 完整筛选。

目的：验证 r6 的 blind 无锁紧凑回退与 audit decision lock。4×A100、48 小时、固定提交。canary 要求零最终截断；full 允许至多 `floor(N/100)` 个明确 reject，严格解析率仍至少 90%，零 runner error。

## 结果

**结构门控通过，但其语义候选结论已被下游发现的输入答案契约错误作废。** Slurm `57580364` 于 2026-09-13 18:04:33–18:24:30（CEST）在 `lrdn0291` 使用 4×A100，固定提交 `1a59505`，退出码 0。

- blind canary 8/8；full 127/128（99.22%），9 次 fallback 中 8 次恢复，5 个 primary 截断，最终选中输出零截断、零 runner error。唯一 reject 为 `invalid_visible_evidence`。
- audit canary 8/8，6/6 fallback 恢复；full 128/128，77/77 fallback 恢复，零截断、零 runner error。
- audit 机器标签为 `external_required=114`、`ambiguous=1`、`image_only=13`；联结后 13 条进入 W2B，115 条拒绝。

W2B canary 随后暴露出上游语义字段误读：M3VQA 的 `answers` 是**多项正确答案列表**，`answer_evals` 才是逐项别名集合；本轮 candidate 却把 `answers` 写成 `reference_answers` 并在 prompt/validator 中称为 aliases。例如多答案题 `m3vqa-data_0008198` 的 gold 为 `Ford/Acura/Suzuki`，audit 却允许只以 `Ford` 为最终答案。因此本档只能证明 r6 的结构生成、回退与 gate 稳定，不能支持“C1 有 13 条 image-only R 候选”或 10.16% 语义产率估计，也不能据此扩 fullpool。
