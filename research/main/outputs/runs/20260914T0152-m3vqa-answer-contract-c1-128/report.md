# 20260914T0152-m3vqa-answer-contract-c1-128

阶段：W2 输入语义契约修复。

## 结果

**PASSED。** 对固定 C1-128 与官方 M3VQA 原始 JSONL 按 `data_id`、`image_id`、question 三重对齐，128/128 成功、0 reject；双跑字节一致，全部通过新版 candidate validator，且输出不含 evidence、URL、实体名或 posthoc 字段。

M3VQA 的 `answer_evals` 子列表才定义需要分别命中的目标项及其别名。本批真实目标基数为：1 项 54 条、2 项 20 条、3 项 23 条、4 项 16 条、5 项 15 条；即 **74/128（57.81%）是多答案目标**。旧流水线把 `answers` 整体称为 aliases，足以改变 audit 对“参考答案是否由图像蕴含”的判断，因此 r6 screen 的 13 条机器候选全部作废。

新版把每个 `answer_evals` 子列表保存在 `reference_answer_alias_groups`，并以对齐的 canonical items 构造唯一完整选项：单项为原字符串，多项为确定性 JSON 数组字符串。后续 audit 必须判断全部目标项，construct 必须逐字复制 `canonical_answer_text`。本档只修复输入，不产生 image-only/R 语义标签。
