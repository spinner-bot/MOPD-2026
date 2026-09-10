# 20260910T-t02-thinking-ab

阶段：T02

目的：Thinking-mode A/B on the same 96 canary candidates (3072 budget) to measure reject->accept flip rate vs non-thinking

## 结果（作业 57154737，COMPLETED，墙钟 45:50，4 GPU）

### thinking 臂输出健康度（96 条，max_new_tokens=3072）

- decision：accept_candidate 66，reject 1，无法解析 29
- truncated 32/96（33%）：**3072 预算对 thinking 模式仍不足**，长推理触顶
- bbox 坐标空间：70/70 全部 norm1000_to_pixel（与非 thinking 臂一致，几何修复在两种模式下稳定）

### 与非 thinking canary（v2，同 96 条）逐条对比

| 非thinking → thinking | 条数 |
|---|---:|
| reject → accept_candidate | **42** |
| reject → 截断/无法解析（censored） | 27 |
| reject → reject | 1 |
| accept → accept | 23 |
| accept → 截断/无法解析 | 2 |
| none → accept | 1 |

- **翻转率 reject→accept：60.0%（42/70）**；含 censored 的下界—上界：60%–98.6%
- 反向翻转 accept→reject：**0/25**

### 翻转真实性检验（对抗"thinking 自我说服"假设）

用 Vision-OPD 自带 source_bbox_xyxy（真值证据区域，像素空间）校验各臂 accept 提案的定位质量：

| 组 | n | mean IoU | median IoU | IoU>0 | 中心落于真值框×1.5 |
|---|---:|---:|---:|---:|---:|
| thinking 翻转 accept | 42 | 0.528 | 0.710 | 40/42 (95%) | 36/42 (86%) |
| 非thinking 稳定 accept | 25 | 0.715 | 0.883 | 23/25 | 21/25 |
| thinking 全部 accept | 66 | 0.584 | 0.807 | 62/66 | 58/66 |

**结论：翻转是真实感知而非幻觉**——若 thinking 只是自我说服，其 bbox 应随机分布；实际 95% 与真值区域重叠、median IoU 0.71。该结果同时独立验证了 norm1000→pixel 换算正确（坐标空间若错位，IoU 应趋近 0）。非thinking accept 的 IoU 更高（median 0.88）说明其接受集偏向显著大目标，thinking 额外找回的是困难小目标。

### 对生产配置的含义（供决策，不自动执行）

1. **P 层（小目标细粒度）T02 需要 thinking 模式**：非 thinking 对困难小目标存在系统性提前放弃（≥60% 的拒绝是误拒）。
2. **thinking 预算需 ≥4096–6144**：3072 仍有 33% 截断。
3. **吞吐代价**：thinking ~115 s/条 vs 非thinking ~8 s/条（~14×）。全量 19,366 条纯 thinking 约需 123 h/4GPU，超出 48h 上限——需分层策略（如 R 层非thinking、P 层 thinking；或非thinking 先筛 + thinking 仅重试 reject 子集）或在多次 allocation 中分段执行。具体方案与 P 层 crop 相对指代处置（双图/改写/分层）一并留待项目决策。

## 提交说明

对照臂：与 v2 canary 完全相同的 96 条候选（每 shard 前 24 条），thinking 开启、max_new_tokens=3072。产出后与 20260910T-t02-visual-structure-pilot-v2/canary/ 逐条对比，统计 reject→accept 翻转率，为 P 层设计决策（是否保留 thinking、是否改为双图输入、是否改写问题）提供证据。翻转率低 => 非 thinking 拒绝是稳定判断（数据问题）；翻转率高 => 非 thinking 存在 deliberation 不足（配置问题）。
