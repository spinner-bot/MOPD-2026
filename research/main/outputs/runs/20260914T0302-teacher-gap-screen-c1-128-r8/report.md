# 20260914T0302-teacher-gap-screen-c1-128-r8

阶段：W2A C1-128 语义门禁复核。输入与 r7 完全相同，共 128 条，完整答案契约 `m3vqa-answer-contract-v1`，SHA-256 `656b7eadcf96d33135242a8f7e7beda775558bd8f5a5807913337a4867d18392`。

## 目的

r7 的 8 个机器 construction candidates 中，逐条诊断发现 6 条具有可确定的 audit 语义缺陷。r8 在 prompt 与 CPU validator 同时要求：evidence node 必须是直接视觉观察，answer node 必须覆盖完整 canonical answer 且不得否定必要项，explanation 不得承认参考答案项未在图中呈现却仍判 `reference_entailed=true`。

离线用新版 validator 回放 r7 已归档输出后，8 个候选降为 2 个；该回放只测试拒绝器，无法观察新 prompt 是否会让模型正确改判为 external/ambiguous。因此 r8 在同一 128 条上完整重跑 blind/audit，保持单次固定 commit provenance。机器候选仍不是人工真值，不据此扩大 fullpool。

## 结果

Slurm `57599082` 于 2026-09-13 21:05:47–21:11:47 CEST 在 `lrdn1759` 运行。blind canary 8/8 均在生成前以 runner error 拒绝，full/audit 未运行，作业按设计 `FAILED (1:0)` fail-closed；没有语义输出，因此本次不能评价新版 audit。

根因是迁移后存在两个 M3VQA 图片根。固定 C1 输入的旧绝对路径经 `/leonardo_work/AIFAC_F02_774/sqs/data` 软链接解析到 `/leonardo_scratch/large/userexternal/rqian000/sqs/data/...`；本次提交却沿用了 B1 新池的 `/leonardo_scratch/large/userexternal/rqian000/data/...`。两处均有图片副本，但 runner 的根目录逃逸检查正确地拒绝了不属于指定 root 的解析路径。后继 r8b 只把 `MOPD_IMAGE_ROOT` 对齐到 C1 路径实际解析根，不改输入内容、样本、模型、prompt 或 validator，并在 B1 r3 结束后串行运行。
