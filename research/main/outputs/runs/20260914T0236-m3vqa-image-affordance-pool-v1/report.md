# 20260914T0236-m3vqa-image-affordance-pool-v1

阶段：图像优先支路 B0，问题防火墙后的源图池。

## 结果

从 13,125 条 M3VQA question records 确定性聚合出 10,565 张唯一源图，0 reject。输出只含图像身份、路径、checksum、opaque source record IDs 与防火墙 provenance；不含原问题、答案、hop、evidence、URL、task-family 候选或任何 teacher-gap 结果。

完整 6.2 MiB manifest 留在 Work/scratch，不提交 Git。两次独立全量运行的 `images.jsonl` SHA-256 同为 `d16bc8a…e927b`，字节一致。Git 仅归档固定 seed `20260914`、按 `sha256(seed\0image_candidate_id)` 选出的 256 张 image-only pilot，以及完整哈希和摘要。sample SHA-256 为 `e69b6d14…ebf4`。

这些记录只是 affordance screening 输入，不是 R/PR 标签。机器视觉审计也只能产生候选；必须先人工确认图像结构，再独立生成新问题，最后按具体问题验证闭世界、唯一答案和最小证据路径。
