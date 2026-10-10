# 人工输出审计协议

## 1. 审计对象

审计对象是 Vision-OPD 已训练权重在官方/复现推理入口下产生的原始文本输出。必须保留完整 `model_answer`，不能只保留经过 `<think>`、`<answer>` 或 `Answer:` 解析后的答案。

## 2. 每条样本的审计顺序

1. 先查看图像、问题、参考答案和模型原始输出；
2. 标记最终选择项首次明确出现的位置；
3. 标记输出中是否出现视觉事实、位置、对象属性、比较关系或其他可核验内容；
4. 判断这些内容是否与图像直接对应；
5. 判断推理步骤是否产生了新的中间结论，而不是重复问题或套用模板；
6. 判断删除推理/证据文本后，模型最终选择项是否仍已经明确；
7. 记录无法判断、输出截断、解析异常和图像无法查看的情况。

## 3. 人工字段

```text
human_output_class:
  direct_choice
  format_only_prefix
  choice_with_explanation
  reasoning_then_choice
  thinking_block_then_choice
  unclear
  error_or_truncated

human_evidence_present: true/false/unclear
human_evidence_valid: true/false/unclear
human_reasoning_present: true/false/unclear
human_reasoning_valid: true/false/unclear
human_answer_visible_before_reasoning: true/false/unclear
human_reasoning_appears_decision_relevant: true/false/unclear
human_answer_correct: true/false/unclear
human_notes: free text
```

`human_answer_visible_before_reasoning=true` 只能在原始输出中已经出现明确选择项、且之后才出现解释/推理时填写。不能根据模型声称“先思考后回答”推断其内部因果顺序。

## 4. 判定边界

- 模型自己写出的“我看到了……”只是模型声称，不自动算作有效视觉证据；必须由人工对照图像确认。
- “根据图像选择 A”“答案是 A”属于格式性或直接选择输出，不属于推理。
- 仅复述问题、选项或结论，不属于有效推理。
- 输出有推理文本，不代表推理文本影响了选择；该字段只能填写为人工审计后的判断或 `unclear`。
- 不能把单个案例外推为整体比例；整体结论必须基于固定抽样清单和完整回收率。
