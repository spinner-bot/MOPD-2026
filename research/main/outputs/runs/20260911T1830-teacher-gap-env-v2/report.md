# 20260911T1830-teacher-gap-env-v2

阶段：INFRA

目的：Install and validate the pinned Stage-0 Python environment from the audited offline wheelhouse

## 结果

2026-09-11 18:30（UTC+8）提交离线环境恢复作业 `57307611`；作业于
00:02:28 后以 `COMPLETED / ExitCode 0:0` 结束。

- 环境：`/leonardo_work/AIFAC_F02_774/sqs/envs/mopd-stage0-py311`
- Python 3.11.16、PyTorch 2.4.1+cu121、Transformers 5.5.0、Pillow 12.0.0
- `pip check`：无破损依赖
- `cuda_available=False`：本次是 CPU-only 基础设施作业，符合预期；GPU 可用性由后续 4×A100 canary 实测
- `pip freeze`、导入检查、Slurm stdout/stderr、`sacct` 与输入哈希已归档

状态：**COMPLETED**。
