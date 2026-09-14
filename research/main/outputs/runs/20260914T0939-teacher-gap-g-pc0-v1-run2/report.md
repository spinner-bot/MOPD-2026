# 20260914T0939-teacher-gap-g-pc0-v1-run2

**阶段**：G-PC0 v1 同 manifest 配置修复重跑

**创建时间**：2026-09-14 09:39（UTC+8）

**最后修改时间**：2026-09-14 09:48（UTC+8）

**作者**：Codex (CINECA Leonardo)

**触发原因**：1044 证明 run1 的 64/64 失败来自 choice scorer 未扩展 `mm_token_type_ids`；按 1043 只修配置、保持 `10b09d12…33bb1c` manifest 不变后完整重跑。

## 完整性结果

Slurm `57636861` 使用 4×A100，4 分 36 秒完成，exit code 0。CPU preflight、64 条 raw、64 条 parsed、manifest 反向绑定、固定 schedule、图片 checksum、runtime code hash、continuation、EOS/截断与解析率门禁全部通过。生成字母与四项 choice score argmax 为 64/64 一致，因此结果不是 constrained generation 与 scoring 两条路径互相矛盾。

## 预注册结果

| 条件 | 正确数 / 16 |
|---|---:|
| S | 6 |
| T-Q | 5 |
| T-QA | 4 |
| Generic-T | 5 |

T-QA 相对 S 为 0 个错→对、2 个对→错，净翻转 −2。S 未达到 14/16 ceiling，但 T-QA 也远未达到错→对至少 4、净翻转至少 3，因此 1043 决定器输出 `INGESTION_OR_RECIPE_FAILURE`。这一决定不可被事后诊断改写。

## 事后描述性诊断

四条件均出现输出标签塌缩：S 预测 A 为 14/16；T-Q 为 15/16；T-QA 为 13/16，并另预测 C 3 次；Generic-T 为 15/16。每个条件的最大标签占比均超过结果后定义的 75% 诊断阈值，而 gold 在每条件严格 A/B/C/D 各 4。

T-QA 使正确标签的 full-vocabulary log-prob 均值相对 S 上升 6.359，但四选项内正确答案归一化概率反而下降 0.0199，accuracy 下降 2/16。这说明“任何单字母答案更像合适 continuation”的整体尺度变化不能当成语义 G 增益；choice 内相对排序仍未指向正确答案。S/T-Q/T-QA/Generic-T 的最大标签占比分别为 0.875/0.9375/0.8125/0.9375。

四题族的 T-QA 正确数分别为 directed path 2/4、sequential transform 1/4、grid trace 0/4、set relation 1/4，未出现某一题族的正向翻转。该分母只有 4，仅作描述。

最稳妥的解释是：**当前 4B + letter-choice 接口 + guide 表示没有把正确中间状态转成可靠的选项决策**。强 A/C 标签先验是明确观察到的候选机制，但当前固定映射无法把“标签接口遮蔽”与“模型确实不利用 guide”分开。下一步若继续，必须先预注册同题同条件的标签轮换诊断；不得直接改写 guides 或换题后把新结果当作 v1。

## 边界

本结果不证明自然图片上文本特权无效，不评价 M3VQA 图像可供性，不允许启动自然 G0、全量 T02–T08 或训练。它证明的是，在当前最小正控制上，现有消费链尚未通过健康检查；自然数据继续等待 B2a，人为扩量不会修复这个模型/接口问题。

## 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 09:48 | Codex (CINECA Leonardo) | 创建：归档 57636861 的 gate-pass、预注册机制失败和标签塌缩事后诊断。 |
