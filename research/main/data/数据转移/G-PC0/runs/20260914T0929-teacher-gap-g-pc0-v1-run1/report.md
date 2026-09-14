# 20260914T0929-teacher-gap-g-pc0-v1-run1

**阶段**：G-PC0 v1 首次四条件运行，配置 gate 失败

**创建时间**：2026-09-14 09:29（UTC+8）

**最后修改时间**：2026-09-14 09:36（UTC+8）

**作者**：Codex (CINECA Leonardo)

**触发原因**：按 1043 对结果前冻结的 16 题正控制执行唯一完整运行。

## 结果状态

Slurm `57636033` 使用 4×A100，运行 4 分 38 秒后以 exit code 1 结束。CPU preflight 先通过 manifest profile、16 张图片 checksum、实际 4B tokenizer 与 49/49 token 长度门禁；四个 rank 随后各写出 16 条 raw，固定 64 项调度完整。

64/64 条在 choice scoring 的模型前向中产生同型 `IndexError`，没有一条形成 choice scores、生成字母或 correctness。结果 gate 因此按设计失败，没有创建 parsed 文件或 decision 文件，未读取也不存在可解释的条件性能。

## 根因与修复边界

Qwen3.5 processor 除 `input_ids` 和 `attention_mask` 外，还返回与 prompt 等长的 `mm_token_type_ids`。旧 choice scorer 为候选答案追加一个 token 时，只把 `input_ids` 与 `attention_mask` 从 N 扩为 N+1，`mm_token_type_ids` 仍为 N。模型的 M-RoPE 逻辑使用 N+1 attention mask 索引 N 长度的 modality tensor，所以 64 条均表现为 mask 比 indexed tensor 长 1。

修复仅在每个 choice 前向中给 `mm_token_type_ids` 追加同形零值；零表示新答案 token 是普通文本。新增 CPU 测试要求三者长度同步为 N+1、追加标签为 0，且不得原位修改 processor 输入。图片、问题、选项、guides、Generic-T、模型 revision、调度、评分定义与三分支裁决均不改变。

依据 1043 的配置失败条款，允许以同一 manifest SHA `10b09d12…33bb1c`、新 run ID、修复后的冻结提交完整重跑。run1 永久保留为失败对照，不拼接其 raw，不尝试恢复部分结果。

## 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 09:36 | Codex (CINECA Leonardo) | 创建：归档 Slurm 57636033 的统一 N+1/N modality mask 失败、零可解释性能和同 manifest 配置重跑边界。 |
