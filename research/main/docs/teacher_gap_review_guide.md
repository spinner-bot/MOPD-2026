# Teacher-Gap 人工审核与交付指引

**创建时间**：2026-09-12 14:45（UTC+8）

**创建作者**：claude code (local)

**创建原因**：W2 teacher-gap pilot 完成后需进行人工审核，提前准备指引文档

**修改时间**：2026-09-14 04:40（UTC+8）

**修改作者**：Codex (CINECA Leonardo)

**修改原因**：同步人工 evidence/DAG 集合一致性与 R/PR 困难视觉节点门禁，防止只改标签生成 PR

---

## 1. 审核目的与边界

人工审核是从模型提议候选到冻结实验样本之间的强制质量门。审核者必须同时查看完整图像、最终问题、参考答案、机器生成的 guides 与选项，独立判断：

1. 问题和全部必要中间命题能否仅由图像与题面推出；
2. 问题指代是否对完整图像成立，答案是否唯一；
3. 样本是否确属 R 或 PR，而非单点识别式 P；
4. 至少两个不同视觉证据是否共同进入同一条必要推理链；
5. guides 和选项是否正确、有效，且不泄漏最终答案或注入外部事实。

机器的 blind、audit 和 construct 输出只是待审核提议，不能自动写成 `human_verified=true`。官方 M3VQA evidence 文本只允许用于实验结束后的来源分析，禁止进入 blind solver、entailment auditor、teacher 输入或人工判断的闭世界证据链。

## 2. 审核材料与权威格式

模型构造完成后，先用 `build_teacher_gap_human_review_queue.py` 将通过结构校验的 construct 记录、blind/audit 提议、guides 和选项合并成完整审核队列。`review-*.csv` 便于浏览和分工；`review-*.jsonl` 是提交给 `finalize_teacher_gap_manifests.py` 的权威输入。旧 C2 审核包仅有原问题且字段不完整，不能直接用于冻结；必须从实际 W2B 构造产物生成新队列。

CSV 中的数组、对象和布尔值容易被电子表格转成字符串。若在 CSV 中工作，交付前必须转换成下述 JSONL 类型，并通过终检程序。禁止手工填写 `human_verified`；该字段只在终检通过后由程序写入冻结清单。

## 3. 单条记录的精确字段契约

| 字段 | JSON 类型 | 合格值或要求 |
|---|---|---|
| `human_knowledge_label` | string | 合格样本必须为 `image_only`；`external_required`、`ambiguous` 不能冻结 |
| `human_task_family` | string | 合格样本必须为 `R` 或 `PR` |
| `human_unique_answer` | boolean | 合格样本必须为 `true` |
| `human_full_image_valid` | boolean | 合格样本必须为 `true` |
| `human_required_evidence` | array[string] | 至少两个非空、互不替代的视觉证据 |
| `human_difficult_visual_evidence` | array[string] | R 必须为空；PR 至少一项，且每项逐字来自 `human_required_evidence` |
| `human_reasoning_depth` | integer | 至少为 2，且等于图中从 evidence 到 answer 的最长边数 |
| `human_reasoning_graph` | object | 满足第 4 节的最小有向无环图 |
| `human_guides_ok` | boolean | 合格样本必须为 `true` |
| `human_options_ok` | boolean | 合格样本必须为 `true` |
| `human_answer_leakage` | boolean | 合格样本必须为 `false` |
| `human_external_fact_injected` | boolean | 合格样本必须为 `false` |
| `human_decision` | string | `accept` / `borderline` / `reject`；只有 `accept` 可冻结 |
| `human_reviewer` | string | 审核者的稳定标识，不得为空 |
| `human_reviewed_at` | string | 带时区的 ISO 8601 时间，例如 `2026-09-13T22:29:00+08:00` |
| `human_notes` | string | 可为空；记录拒绝原因、争议点或修订说明 |

JSON 布尔值必须写成不带引号的 `true`/`false`。`"yes"`、`"no"`、`1`、`0` 均不符合审核输入契约。

## 4. 最小推理图契约

边的方向固定为 `[依赖节点, 被推出节点]`。每个合格图必须：

- 恰有一个 `answer` 节点，至少一个 `intermediate` 节点，至少两个 `evidence` 节点；
- 每个节点含唯一非空 `id`、合法 `type` 和非空 `claim`；
- evidence 是根节点，answer 是汇点，非 evidence 节点有上游依赖；
- 无自环、重边、未知端点和有向环；
- 所有节点都位于通向唯一 answer 的祖先子图中，不得加入无用证据；
- answer 的祖先中至少有两个 evidence；
- `human_reasoning_depth` 等于 evidence 到 answer 的最长路径边数。

合格示例：

```json
{
  "human_required_evidence": ["左侧标记连接到上顶点", "右侧标记连接到下顶点"],
  "human_reasoning_depth": 2,
  "human_reasoning_graph": {
    "nodes": [
      {"id": "e1", "type": "evidence", "claim": "左侧标记连接到上顶点"},
      {"id": "e2", "type": "evidence", "claim": "右侧标记连接到下顶点"},
      {"id": "h1", "type": "intermediate", "claim": "两个连接属于同一形状"},
      {"id": "a", "type": "answer", "claim": "正确选项是三角形"}
    ],
    "edges": [["e1", "h1"], ["e2", "h1"], ["h1", "a"]]
  }
}
```

`required_evidence` 与图中 evidence claim 应表达同一组必要观察。若去掉任一证据答案仍可唯一推出，该样本不满足当前 R/PR 门槛，除非重新构造问题使该证据成为必要条件。

终检按忽略大小写和首尾空白后的集合，要求 `human_required_evidence` 与推理图全部 evidence claim 精确一致且互不重复。PR 必须从该集合指出至少一个真实困难局部节点；R 的困难节点列表必须为空。只改 `human_task_family` 而不提供困难必要证据，不能把 R 升级为 PR。

## 5. 判定顺序

1. **闭世界检查**：遮蔽参考答案和官方 evidence 文本，仅凭完整图像与题面判断可答性。需要百科、人物身份、历史地理事实或题面未给出的规则时，标为 `external_required`。
2. **完整图像检查**：排除依赖原 teacher crop 方位的描述。例如原 crop 中的“图像底部”在完整图中不成立时，`human_full_image_valid=false`。
3. **唯一答案检查**：四个选项中必须恰有一个由图像证据支持；合理多解或关键区域不可辨时拒绝。
4. **R/PR 与图检查**：先列必要证据，再画最小 DAG，最后由最长路径填写 depth。单个局部属性读取归 P，不纳入当前 teacher-gap 冻结集。
5. **guides 检查**：每问每答必须可由图像推出，并支持下一中间节点；删除某一步若不影响推理，应精简。
6. **泄漏与注入检查**：guide 不得直接给出最终选项、同义改写答案、逐项排除到唯一答案，也不得引入图中没有的事实。
7. **选项检查**：正确答案文本与参考答案一致，四个选项互异；干扰项应同类型且可被图像证据排除。

任一冻结必需条件失败时填写实际观察结果，并将 `human_decision` 设为 `borderline` 或 `reject`。不要为了通过终检而把失败字段改成合格值。

每个 guide 的 `supports_next` 必须引用人工确认图中存在的 `intermediate` 或 `answer` 节点，且整组 guides 至少支持一个 intermediate 节点。若人工修订了节点 ID，必须同步修订 guide 引用。

## 6. 完整 JSONL 合格示例

```json
{"candidate_id":"m3vqa-example","human_knowledge_label":"image_only","human_task_family":"R","human_unique_answer":true,"human_full_image_valid":true,"human_required_evidence":["左侧连接关系","右侧连接关系"],"human_difficult_visual_evidence":[],"human_reasoning_depth":2,"human_reasoning_graph":{"nodes":[{"id":"e1","type":"evidence","claim":"左侧连接关系"},{"id":"e2","type":"evidence","claim":"右侧连接关系"},{"id":"h1","type":"intermediate","claim":"两条关系指向同一图形"},{"id":"a","type":"answer","claim":"选择三角形"}],"edges":[["e1","h1"],["e2","h1"],["h1","a"]]},"human_guides_ok":true,"human_options_ok":true,"human_answer_leakage":false,"human_external_fact_injected":false,"human_decision":"accept","human_reviewer":"reviewer-01","human_reviewed_at":"2026-09-13T22:29:00+08:00","human_notes":""}
```

## 7. 交付与验收

1. 保留原模型输出和初始空模板；完成稿使用新的文件名，禁止覆盖运行档案中的既有证据。
2. 检查 `candidate_id` 唯一，JSONL 每行均能独立解析。
3. 运行 `finalize_teacher_gap_manifests.py`。其输出中的拒绝原因是修订依据；不得绕过或手工删除拒绝记录。
4. 只有达到 8/32/96 三个互斥分区共 136 个不同近重复簇，并通过标签平衡、哈希和清单校验后，才可将状态记为 `frozen`。
5. 任何机器输出或未经人工查看的记录均不得署名为人工审核。
6. `--output-dir` 必须是尚不存在的新 run ID；终检拒绝复用空目录或覆盖任何既有输出。每条 construction candidate 还必须显式携带真实 `source_dataset`，冻结程序按记录原样传递，不从 ID 推断，也不统一改写为 M3VQA。
7. frozen manifest 保留 `canonical_answer_text` 与 `difficult_visual_evidence`；bundle gate 会重新核对正确选项文本、required evidence 与 DAG、R/PR 困难节点规则，以及 guide 的 `supports_next`。人工终检通过后的语义 provenance 不得在导出时丢弃。

## 8. 关联文档

- 1017：teacher-gap 冻结协议
- 1019：C2 M3VQA 富化与人工审核包
- 1020：W2 滚动作业链清点
- `code/data_pipeline/build_m3vqa_teacher_gap_review_packet.py`：空白审核模板生成器
- `code/data_pipeline/build_teacher_gap_human_review_queue.py`：构造产物与人工字段的完整审核队列生成器
- `code/data_pipeline/finalize_teacher_gap_manifests.py`：人工审核终检与冻结程序
- `code/data_pipeline/reasoning_graph_contract.py`：推理图权威结构校验器

## 9. 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-12 14:45 | claude code (local) | 创建：W2 teacher-gap pilot 后提前准备人工审核和交付指引。 |
| 2026-09-13 22:29 | Codex (CINECA Leonardo) | 修订：统一 1017 冻结协议、审核包字段和终检程序，修复旧指南取值与机器契约不兼容。 |
| 2026-09-14 04:40 | Codex (CINECA Leonardo) | 修订：同步 `586c1d4` 的 evidence/DAG 集合一致性和 R/PR 困难视觉节点字段；补齐历史修订表。 |
| 2026-09-14 04:42 | Codex (CINECA Leonardo) | 修订：终检输出改为新 run ID 排他写入，并要求 construction candidate 显式携带真实图源，避免混合来源被静默写成 M3VQA。 |
| 2026-09-14 05:07 | Codex (CINECA Leonardo) | 修订：冻结清单保留 canonical answer 与困难视觉证据，使 bundle gate 可独立重验答案、证据/DAG、R/PR 和 guide 对齐。 |
