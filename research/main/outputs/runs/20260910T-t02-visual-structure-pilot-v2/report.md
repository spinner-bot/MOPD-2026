# 20260910T-t02-visual-structure-pilot-v2

阶段：T02（pilot 重跑 v2）

目的：修复 v1 pilot（20260909T-t02-visual-structure-pilot）0/1200 通过 gate 的两个结构性缺陷后，重新生成 Qwen3.5-9B 视觉结构与 crop 提案。

## v1 失败诊断（本 v2 的依据）

1. thinking 模式未关闭 + max_new_tokens=512：1200 条中 984 条（82%）在推理链中被截断、从未输出 JSON；127 条（10.6%）JSON 中途截断；仅 89 条（7.4%）完整。宽松重解析后也仅 93 条 accepted。
2. bbox 坐标系错位：模型原生输出 [0,1000] 归一化坐标（95/95 个 bbox 上限恰为 1000，而源图像 1500–2672 px），v1 校验仅查 0<=x1<x2<=w，导致 accepted 样本 bbox 静默错位，crop 不可用。
3. 内容质量评估：v1 thinking 链显示模型理解任务、证据定位合理，属结构/配置问题而非 9B 能力不足。

## v2 修复

- runner：apply_chat_template(..., enable_thinking=False)；max_new_tokens 512→1024；prompt 改为单一 JSON 对象 + 显式 [0,1000] 归一化坐标约定；EOS 截断检测（truncated 标志）；--limit 支持 canary。
- bbox_geometry.py：确定性 norm1000→pixel 变换，保留双坐标系与 provenance 标签；>1000 坐标视为像素直通。
- validator：新增 truncated_output 拒绝原因。
- sbatch v2：同一 allocation 内先跑 96 条 canary（4 GPU × 24），CPU fail-closed 门控（parse_rate>=0.8、accept_rate>=0.4、truncation_rate<=0.1、零 runner 错误），通过后才跑全量 1,200 条，失败即提前退出避免浪费 GPU 时。

## 已提交配置

- Slurm 链：57151182（pilot v2，4 GPU/12h）→ afterany → 57151183（校验，输出 t02-pilot-v2-accepted.jsonl / rejects/t02-pilot-v2.jsonl）→ afterany → 57151184（1,200 条完整性 gate）。
- 模型：本地 Qwen/Qwen3.5-9B，revision c202236235762e1c871ad0ccb60c8ee5ba337b9a，贪婪解码。
- 输入：candidate-v0.1/02_candidates/t02-visual-structure-pilot/shard-{000..003}.jsonl（1,200 条，与 v1 相同，保证可对比）。

## 当前状态

2026-09-10 提交完毕，57151182 PENDING (Priority)。canary 汇总将写入 summaries/canary.json；运行结束后必须经 57151183/57151184 验证并更新本档案，未通过验证的输出不得进入 T03。
