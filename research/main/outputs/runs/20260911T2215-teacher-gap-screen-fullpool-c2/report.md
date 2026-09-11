# 20260911T2215-teacher-gap-screen-fullpool-c2

阶段：W2A（全量预付费筛选 chunk c2/4；admin 指令滚动提交，2026-09-11 22:15 UTC+8）

目的：对剩余 eligible 池的 1/4 执行 blind+audit 语义筛选（冻结规则不变，详见 command.txt 合规说明）

## 输入

chunk_c2.jsonl（确定性生成于 $WORK/sqs/data/m3vqa/processed/exp-fullpool/；manifest.json 待链路恢复后从 Leonardo 同步）

## 结果

PENDING（Slurm 57331299，4×A100，预计 ~3.3h 墙钟）。完成后回填。
