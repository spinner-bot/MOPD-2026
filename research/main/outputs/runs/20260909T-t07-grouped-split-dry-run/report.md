# 20260909T-t07-grouped-split-dry-run

阶段：T07

目的：候选池分组切分干跑：隔离精确与近重复图像，确认正式冻结前的阻塞条件。

## 输入

- `candidate-v0.1/02_candidates/source-normalized.jsonl`：19,366 条候选。
- `candidate-v0.1/00_sources/image-duplicate-clusters.json`：66 个精确重复簇、73 个近重复簇。

## 结果

并查集将每个精确或近重复簇连通分量锁定在同一 split。干跑得到 16,684 个组，且重复簇跨 split 违规为 0。

切分计数：训练 14,020；验证 1,766；测试 2,597；Stage 0 候选 983。

该结果不是正式冻结：PR 候选、引导 QA 与控制项尚未产生，T05/T06 审核也尚未完成。

## 配套准入审计

同次运行的 T01 审计结果保存为 `outputs/t01-stage0-source-audit.json`：当前候选仅具备源图像标识和路径，19,366 条均缺失正式 Stage 0 所需的样本标识、题目、选项、答案、裁剪和控制字段。
