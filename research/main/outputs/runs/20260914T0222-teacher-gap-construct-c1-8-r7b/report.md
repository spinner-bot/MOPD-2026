# 20260914T0222-teacher-gap-construct-c1-8-r7b

阶段：W2B C1-8 完整答案候选构造。上游 r7 screen `57593306`，输入 SHA-256 `82f977aafb5c7153aefcfd38e2e005bf38c7d5d5b7ffc6935da2acc19cdb5843`；固定代码 `aa11b60`。

## 结果

Slurm `57595439` 于 2026-09-13 20:25:50–20:33:08 CEST 在 `lrdn2412` 运行，`FAILED (1:0)` 是 canary fail-closed 的预期停止。8 条均无 runner error、无最终截断；仅 2/8 严格有效，低于 75% 门槛，full 未运行。7 条触发 fallback，1 条恢复。最终拒绝原因：`invalid_distractor_cardinality` 5 条、`options_not_distinct` 1 条。有效输出为 1 个 accept proposal 与 1 个 reject proposal，均非人工真值。

逐条 raw 显示两种契约过度约束：模型自然把多答案错误项输出为 JSON 数组，而门禁只接受“数组的字符串编码”；guide 常用于先取得 evidence node，旧对齐器却只准指向 non-evidence node。两者都是表示层问题，不应改变语义内容。后继修复允许等价的结构化数组，并允许 guide 对齐任一现存图节点；仍严格检查目标基数、完整选项去重与 dangling target。另一方面，8 条 screen 候选还暴露大量机器语义假阳性（职业群体唯一性、车型/猫种/菜系知识、把题面本身写成视觉证据），所以即使结构构造通过也必须 100% 人工审核，且不得据此直接扩 fullpool。
