# 20260914T0258-image-affordance-b1-256-r2

阶段：支路 B1，仅看完整图片的 affordance machine proposal。固定输入与首次提交相同，共 256 图，SHA-256 `e69b6d14fe8da625720af1ee7198213b909c682c10e2313003f43c2c139debf4`。不展示原问题、答案、evidence、URL 或 teacher-gap 结果。

## 结果

Slurm `57598570` 于 2026-09-13 20:59:20–21:05:16 CEST 在 `lrdn1784` 运行。环境预检和 4×A100 模型加载成功，证明首次提交的路径故障已修复。8 条 canary 中只有 1 条严格有效（12.5%），低于 75% 门槛；2 条 fallback 最终截断，full 未运行，作业按设计 `FAILED (1:0)` fail-closed。没有 runner error。

逐条 raw 表明主要问题不在视觉描述能力：8 条 primary 都生成了可读的实体和关系，7 条触发 fallback，其中 6 条的直接原因是模型选择 `candidate`，却同时给出 `r_score=1`（低于候选所需的 2）。这反而表示模型认为图中只有较弱关系，validator 拒绝是正确的。旧 fallback 只看 decision/feasibility/external 三个字段，把这种内部不一致的 `candidate` 错误锁死；compact prompt 又只给空数组骨架，导致模型把对象数组改写成字符串列表、使用 0–1 浮点分数并在两条上重复到截断。0/7 fallback 恢复说明恢复协议有缺陷，不能把 12.5% 解释为图像池可供性。

后继只修恢复协议和候选校准：无效 primary candidate 允许降级为 reject/ambiguous；compact prompt 给出完整对象结构和整数分数；normal prompt 明确 `candidate => r_score>=2`，并排除“识别两个对象后复述简单远近/上下关系”冒充两步 R。不会放宽 validator，也不会自动采纳唯一有效的机器 candidate。所有有效输出仍待独立人工复核。
