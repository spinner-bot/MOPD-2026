# 2174-T：全量数组启动与 checkpoint 续跑验证

## 作业状态快照

数组 `60427687` 已在 Leonardo 的 boost 生产队列实际启动。复核时已有21/30个条件—评测集分片写出 `COMPLETE-<condition>-<track>` 标记；5片仍在运行：S2 的两个 MME 评测集，以及 S3 的 hrbench-8k、mme-realworld、mme-realworld-cn。依赖管理器 `60427690` 仍因 `afterany:60427687_*` 未满足而等待，符合“整组数组结束后再汇总/续跑”的预设依赖。

已有21个标记对应此前 Slurm `COMPLETED` 的任务。4个已触及两小时限制、尚无完成标记的分片为 S0/mme-realworld、S0/mme-realworld-cn、SC/mme-realworld、S1/mme-realworld。Slurm 记录这些任务为 `TIMEOUT`；这只说明分片尚未完成，不代表其已达到完整数据覆盖。

## 部分结果与恢复依据

超时分片的正式输出路径中仍有逐条写入的 JSONL checkpoint。核验时文件行数分别为：S0/mme-realworld 12,759（该评测集全量23,609）；S0/mme-realworld-cn 5,870（全量5,917）；SC/mme-realworld 12,700；S1/mme-realworld 13,190。行数仅作为已落盘进度快照，不等同于唯一、成功请求数或完成证明。

`infer.py` 的恢复逻辑会读取已有输出、按 sample_uid 合并重复记录，并仅对仍待处理/可重试的样本继续请求；每条结果写入后立即 flush。缺少分片完成标记的任务不会进入全量冻结验证。管理器待数组全部终止后会检查30个分片标记，并只重新提交缺失编号。因此本轮超时分片的部分输出得以保留，后续续跑从 checkpoint 接续；本报告不把 checkpoint 行数表述为已完成评测结果。

## 当前判定

生成阶段已从排队等待转为运行，数据正在写入 large 存储区。完整评测仍未完成：数组尚有5片运行、4片待续跑，之后仍需冻结验证、原生评分、模型判别、答案抽取与统计分析。现阶段不启动下游阶段。
