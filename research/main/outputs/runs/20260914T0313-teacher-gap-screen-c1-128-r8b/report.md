# 20260914T0313-teacher-gap-screen-c1-128-r8b

阶段：W2A C1-128 语义门禁复核。输入、模型、prompt、validator 和阈值与 r8 相同；唯一修正是把 `MOPD_IMAGE_ROOT` 对齐到固定 C1 绝对路径经软链接解析后的实际根 `/leonardo_scratch/large/userexternal/rqian000/sqs/data/m3vqa/materialized/images`。

## 结果

**FAILED CLOSED（audit full 中途存储迁移）。** Slurm `57600241` 于 2026-09-13 21:18:52–21:34:37 CEST 在 `lrdn2738` 使用 4×A100，固定提交 `8c5537f`，退出码 `1:0`。

- blind canary 8/8；blind full 127/128（99.22%），9 次 fallback、8 次恢复，零最终截断、零 runner error；与 r7 的 blind 结果一致。
- audit canary 8/8，5/5 fallback 恢复；机器标签为 external 7、image-only/R 1。
- audit full 共写出 128 条，但只有 95 条严格有效（74.22%）：57 次 fallback、53 次恢复；33 条拒绝中 29 条是 `FileNotFoundError` runner records，另有 4 条未通过新增的完整答案/解释一致性语义门禁。gate 正确阻止了 parse/select-construction。

29 个 runner error 从 input index 97 起出现在全部四个 rank。首批错误是单个图片路径消失，随后 25 条直接报告整个 `/leonardo_scratch/large/userexternal/rqian000/sqs` 不存在。作业结束后核查显示旧根已消失，数据和模型位于新建的 `/leonardo_scratch/large/userexternal/rqian000/ICLR-2027-czy/`；该顶层目录的 birth time 为 21:34:29 CEST，正处于 audit full 运行窗口。因此这是运行中存储根被整体重定位，不是模型输出截断，也不是新版语义 validator 把 29 条合法记录误杀。

残缺的 95 条有效 audit 中出现 external 88、ambiguous 3、image-only/R 4。因为最后 29 条没有推理，其中还包括离线回放保留候选 `m3vqa-data_0000042`，这 4 条不得解释成 C1 的最终候选数，也不补做 construction。支路 A 已因原题低产率冻结，故不再为取得一个无人工真值的精确机器计数提交第四次 4×A100 重跑；后续 GPU 作业必须等待迁移根和软链接稳定，并在提交前重新校验全部图片路径。
