# Generic control tokenizer stress report

**创建时间**：2026-09-14 11:31（UTC+8）

**作者**：Codex (CINECA Leonardo)

**触发原因**：1057/1058 把 natural G0 的两项 generic 改为逐基础样本 tokenizer 匹配；需要在真实冻结 tokenizer 上覆盖不同 guide 数和问答长度，确认 10% 门槛可实现，并验证 `bb0c8a2` 的候选 token 缓存。

## 结果

- 固定组合：1–4 个 guides × 5 个 question word counts × 4 个 answer word counts × T-Q/T-QA，共 160 个 case；
- 160/160 满足逐题相对误差 ≤10%；
- 最大相对误差 4.5455%，对应 T-QA 目标 22 tokens、控制 21 tokens；
- 中性候选池为 620 个确定性前缀，token 长度只计算一次；
- 档案内计时 1.056 秒，完整进程还包含 Python/transformers 启动开销；
- Qwen3.5-4B tokenizer revision 和三个 tokenizer/config artifact SHA-256 已写入 `summary.json`。

本档案只使用合成 guide 文本与 tokenizer，不读取任何自然问题、人工标签或 S/T 性能输出；未使用 GPU。它证明当前构造器在所测长度网格上能满足工程合同，不证明 generic 文本在语义或注意力模式上与结构化 guide 完全等价。

## 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 11:31 | Codex (CINECA Leonardo) | 创建：归档 160-case 真实 tokenizer 压力检查、最大误差、artifact hashes 与结果盲边界。 |
