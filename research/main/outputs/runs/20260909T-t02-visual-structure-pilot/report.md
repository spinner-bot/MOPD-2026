# 20260909T-t02-visual-structure-pilot

阶段：T02

目的：Qwen3.5-9B four-GPU visual structure and crop proposals for frozen pilot shards

## 已提交配置

- Slurm 作业：`56995520`，分区 `boost_usr_prod`。
- 资源：1 节点、4 GPU、4 个确定性分片、1,200 个 pilot 候选。
- 模型：本地 Qwen/Qwen3.5-9B，固定 revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 解码：贪婪解码，最多 512 个新 token。

## 当前状态

截至 2026-09-09 15:55 UTC，作业状态为 `PENDING (Priority)`，尚未取得节点或 GPU。提交脚本使用四个 `srun` task，每个 task 绑定一张 GPU；不使用 API 或外传数据。

运行完成后，必须先以 `validate_t02_proposals.py` 验证全部分片，再更新本档案的状态和结构化统计。未通过验证的输出不得进入 T03。
