# 图像可供性 B2a 盲审指引

**创建时间**：2026-09-14 03:52（UTC+8）  
**创建作者**：Codex (CINECA Leonardo)  
**创建原因**：1035 停止 9B 自由 affordance 自评分后，需要让人工在不受模型 proposal 和 M3VQA 原题锚定的情况下，对固定 32 图建立首批可校准真值。

---

## 1. 审核目的

B2a 判断的是：**只看这张完整图片，人是否能提出一个自然、答案唯一、无需外部知识，而且真正需要关系推理的具体问题。**

它不直接判断图片“属于 P、R 或 PR”。图片只有潜在可供性；P/R/PR 标签属于具体的图片、问题、答案和最小证据链。审核者必须先写出一个实际目标，才能判 candidate。

第一阶段禁止查看：

- B1 的机器实体、关系、路径、decision 或分数；
- M3VQA 原问题、答案、Wikipedia evidence 和 URL；
- 任一模型在 S/T 条件下的结果；
- 其他审核者对同图的判断。

界面也不得显示 `source_dataset`、`source_image_id`、`image_candidate_id` 或图片路径。审核者只用随机化的 `calibration_record_id` 导航；服务端保留的来源身份仅供哈希校验和第一阶段锁定后的联结，不能进入浏览器 API。

机器可以提供图片浏览界面和字段校验，但不得生成、改写或预填任何 `human_*` 字段。

## 2. 权威材料与文件规则

空白权威包为：

```text
research/main/outputs/runs/20260914T0328-image-affordance-b2a-blind-calibration/outputs/blind-review-32.jsonl
```

它含 32 条，SHA-256 为 `b61839ddb4111f1be5d2d67f165c40dc05cd10daf8d5ca02a202b707a1057a58`。只读取该文件，不覆盖。每位审核者复制一份到新的工作文件，例如 `Work/b2a-review-reviewer01.jsonl`；多人审核必须各自使用独立副本。

`image_path` 只用于打开对应完整图片。若路径或哈希异常，停止该条并报告基础设施问题，不自行换图。

技术预检 `20260914T0445-b2a-image-decode-preflight` 确认 32/32 可解码且哈希一致，但其中 12 张低于 0.1 MP、3 张短边小于 224 px。审核时应能查看 100% 原生像素和当前缩放比例；不得使用超分辨率、AI 增强或平滑放大后新增的伪细节作为证据。分辨率不足以确认必要证据时，按真实观察标为 ambiguous/unusable，而不是猜测。

标准库审核界面可从仓库根启动：

```bash
/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python \
  research/main/code/review/image_affordance_blind_review_app.py \
  --draft Work/b2a-review-reviewer01-v1.jsonl \
  --export-prefix Work/b2a-review-reviewer01-final-v1 \
  --port 8765
```

服务只接受 `127.0.0.1`，打开 `http://127.0.0.1:8765/`。若浏览器不在 Leonardo 同一主机，通过 SSH local port forwarding 访问该 loopback 端口，不能把 host 改成 `0.0.0.0`。`--draft` 与 `--export-prefix` 必须指向审核者自己的新路径；draft 第一次保存后可继续原子更新，final 三个文件只创建一次。

界面载入 draft 时要求每条 JSON 的键集合与冻结原包完全一致；额外 `null` 键和缺键均拒绝。finalizer 对绕过界面的 reviewed 文件执行同一严格 schema 门，图片响应则对实际发送字节重验冻结 SHA-256。

## 3. 判定顺序

### 3.1 图片可用性

先判断图片是否能正常打开，主体是否足够清楚。填写 JSON boolean `human_image_usable`。图片损坏或整体不可辨时设为 `false`，decision 设为 `reject`，并在 notes 写明原因。

### 3.2 提出一个具体目标

在不知道原题的情况下，寻找图片能自然支持的比较、空间组合、连接、顺序、状态变化或图表关系问题。只记录一个最清楚、最自然的目标。

合格目标必须：

- 可从这张完整图与问题文字本身作答；
- 有唯一、可明确写出的完整答案；
- 至少依赖两个互不替代的视觉证据；
- 至少需要两个不可省略的推理变换；
- 不要求认出具体人物、地名、品牌、物种、事件或百科事实。

若能提出合格目标，decision 为 `candidate`。完全提不出时用 `reject`；图片似乎有潜力，但分辨率、遮挡或多解使判断不稳时用 `ambiguous`。reject/ambiguous 必须在 notes 说明原因。

### 3.3 写出完整答案

`human_canonical_answer` 可以是非空字符串，或多项目标的非空字符串数组。它必须回答整个问题，不能只写其中一个组成项。

问题和答案应使用图中可见描述，例如“左侧被阻挡的路线”，避免“巴黎”“某演员”“某汽车品牌”这类需要记忆的身份词。即使审核者认识图中对象，也要问：一个不知道其专名的人能否仅凭图像完成推理。

### 3.4 列出直接视觉证据

`human_evidence_descriptions` 至少写两项直接观察，每项只描述图中事实，例如：

- “左侧路线在中段被路障阻断”；
- “右侧路线连续连接到出口标记”。

下列表述不是直接视觉证据：

- “题目问的是哪条路”；
- “正确答案是右路”；
- “这个地标通常位于某国”；
- “根据常识这种动物会……”；
- “模型认为它是 R”。

### 3.5 写出推理变换

`human_reasoning_steps` 至少写两项从证据走向答案的变换。变换会产生新的中间结论；单纯看见、命名、重复位置词或把一句话拆成两句不计作推理变换。

**假两步，判为非 candidate：**

1. 识别盘子和炉灶；
2. 说盘子在炉灶前面。

这里只判断一个直接关系。把两个物体分别“识别”并不会自然产生两步推理。

**可接受的两步结构：**

1. 分别沿两条可见路线检查连续性，得到左路被阻断、右路连通；
2. 把两项路径状态与“可到达出口”条件结合，选出右路。

这里两侧状态共同进入比较，任何一侧缺失都可能破坏唯一答案。

### 3.6 做反事实必要性检查

在 `human_counterfactual_required_evidence` 中逐字复制 `human_evidence_descriptions` 的全部项目，并逐项问：**若遮住或删除这项证据，剩余信息还能否唯一推出答案？** 两个列表必须表示同一组互不重复的最小证据；finalizer 会校验集合一致。

- 若仍能唯一作答，该证据不是必要证据，应从链中删除；
- 若删到最后只剩一个必要局部读取，当前目标偏 P，不作为 R/PR candidate；
- 若两项或更多证据各自不可替代，且共同经过至少两步变换得到答案，才保留 candidate。

### 3.7 区分 R 与 PR

- **R**：关系推理链成立，所需局部证据在正常查看下清楚可辨；`human_difficult_visual_evidence` 必须为空数组。若其中确有困难必要节点，应改判 PR。
- **PR**：同一关系链成立，而且链上至少一项必要证据存在真实感知困难，例如小字、微小标记、密集遮挡或细粒度局部差异；必须在 `human_difficult_visual_evidence` 中具体写出该证据。

图片复杂、物体很多或画面漂亮本身不构成 PR。困难视觉节点必须是答案链上的必要证据。

### 3.8 外部知识检查与签名

`human_external_knowledge_needed` 必须是 JSON boolean。candidate 必须为 `false`。若目标仍需外部事实，将 decision 改为 reject/ambiguous 或重新设计成闭世界问题。

完成一条后填写稳定 reviewer ID 和带时区 ISO 8601 时间；全部字段由本人核查后，人工设置：

```json
"review_status": "human_blind_affordance_review_complete",
"human_verified": true
```

B2a 的 `human_verified` 是审核者对盲审字段的签名，与 `teacher_gap_review_guide.md` 中由冻结程序写入的下游训练清单状态不是同一阶段。

## 4. 字段类型

| 字段 | 类型 | candidate 要求 |
|---|---|---|
| `human_image_usable` | boolean | `true` |
| `human_affordance_decision` | string | `candidate` |
| `human_target_question` | string | 非空、闭世界、唯一答案 |
| `human_canonical_answer` | string 或 array[string] | 覆盖完整问题 |
| `human_affordance_family` | string | `R` 或 `PR` |
| `human_evidence_descriptions` | array[string] | 至少 2 项直接视觉事实 |
| `human_reasoning_steps` | array[string] | 至少 2 项不可省略的变换 |
| `human_counterfactual_required_evidence` | array[string] | 与直接证据列表为同一集合，至少 2 项 |
| `human_difficult_visual_evidence` | array[string] | 必须来自直接证据列表；PR 至少 1 项，R 必须为空 |
| `human_external_knowledge_needed` | boolean | `false` |
| `human_reviewer` | string | 非空稳定标识 |
| `human_reviewed_at` | string | 带时区 ISO 8601 |
| `human_notes` | string | candidate 可空；其他决定必填 |

布尔值必须使用不带引号的 `true`/`false`；列表必须是 JSON array，不能写成逗号分隔字符串。

## 5. 完成门禁

完成 32 条后运行：

```bash
/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python \
  research/main/code/data_pipeline/finalize_image_affordance_blind_review.py \
  --original research/main/outputs/runs/20260914T0328-image-affordance-b2a-blind-calibration/outputs/blind-review-32.jsonl \
  --reviewed Work/b2a-review-reviewer01.jsonl \
  --accepted Work/b2a-review-reviewer01.valid.jsonl \
  --rejected Work/b2a-review-reviewer01.invalid.jsonl \
  --summary Work/b2a-review-reviewer01.summary.json
```

只有 `valid_signed_reviews=32` 才算这一位审核者完成第一阶段。finalizer 不补字段、不改 decision，也不会把未签名记录升级为已验证。任何图片身份、防火墙字段或其他不可变字段被修改都会拒绝。

finalizer CLI 默认固定原始包 SHA-256 `b61839dd…a1057a58` 和 32 条记录，并将原包与 reviewed 工作文件的实际 SHA-256 写入 summary。只有显式合成测试才能通过参数改期望值；真实 B2a 不得覆盖默认门禁。

`--accepted`、`--rejected` 与 `--summary` 必须是三个互不相同且尚不存在的新路径。finalizer 使用 exclusive create，拒绝覆盖旧审核结果；修订或重试应使用带版本的新文件名，并保留原输出。

若安排两位审核者，两人必须先各自得到 32/32 valid，再计算 candidate/noncandidate 一致性和 R/PR 一致性；分歧通过第三份显式 adjudication 记录解决，不能让后审核者覆盖先审核者文件。

1039 的 B2a 路由分析器 `analyze_image_affordance_blind_calibration.py` 也要求 `--output` 为尚不存在的新路径并排他创建；0、1–7、8–32 三档裁决摘要不能原地覆盖。

## 6. 第二阶段揭示规则

第一阶段文件和哈希锁定前，禁止查看 B1。锁定后按同一 `image_candidate_id` 揭示机器 proposal，仅用于评价：

1. candidate precision 与可计算样本上的漏检率；
2. 机器描述是否忠实；
3. 机器路径是否匹配人工的具体目标和必要证据；
4. 简单关系被拆成假两步的比例。

第二阶段不得回改第一阶段人工判断。B1 r3 只完成 canary，缺失机器输出的图片记为 `machine_output_unavailable`，不能算作机器 reject 或假阴性。

锁定第一阶段文件后，用以下命令生成独立的第二阶段揭示包；输出路径必须是新文件：

```bash
/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/mopd-stage0-py311/bin/python \
  research/main/code/data_pipeline/build_image_affordance_machine_comparison.py \
  --original research/main/outputs/runs/20260914T0328-image-affordance-b2a-blind-calibration/outputs/blind-review-32.jsonl \
  --reviewed Work/b2a-review-reviewer01.jsonl \
  --machine-raw 'research/main/outputs/runs/20260914T0308-image-affordance-b1-256-r3/canary/image-affordance/rank-*.jsonl' \
  --output Work/b2b-review-reviewer01.jsonl \
  --summary Work/b2b-review-reviewer01.summary.json
```

构建器会重新验证原包 SHA、32/32 签名和单条 ID，并记录第一阶段文件 SHA。只有 `parsed_valid` 的机器输出进入 precision/recall 交集；`invalid` 与 `unavailable` 分开计数。第二阶段的 faithful/path-match/false-two-step 等人工比较字段保持空白，不能由程序代填。

B2b 的 `--output` 与 `--summary` 同样必须是互不相同且尚不存在的新路径；构建器排他创建文件，拒绝覆盖既有揭示包或统计摘要。

## 7. 关联材料

- 1034：图像优先支路 B 的最初冻结顺序；
- 1035：任务瓶颈、图像可供性和 effect-first 决策；
- 1036：迁移后二次根变化与兼容路径恢复；
- `build_image_affordance_blind_calibration.py`：空白盲审包构建器；
- `finalize_image_affordance_blind_review.py`：完整性、防火墙与语义字段门禁。

## 8. 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 03:52 | Codex (CINECA Leonardo) | 创建：将 1035 的先人后机设计转成可直接执行的 32 图审核规则、字段类型、反事实检查与完成命令。 |
| 2026-09-14 04:13 | Codex (CINECA Leonardo) | 修订：加入 B2b 锁后揭示命令；明确机器 invalid/unavailable 不得作为负例，比较字段继续由人工填写。 |
| 2026-09-14 04:40 | Codex (CINECA Leonardo) | 修订：与 `586c1d4` 统一 R/PR 语义；R 的困难证据数组必须为空，PR 必须含至少一个属于必要证据集合的困难节点。 |
| 2026-09-14 04:47 | Codex (CINECA Leonardo) | 修订：引用 B2a 技术解码预检；要求核对原生像素和缩放，禁止图像增强产生证据。 |
| 2026-09-14 05:02 | Codex (CINECA Leonardo) | 修订：finalizer 的 accepted/rejected/summary 改为互异新路径和排他写入；重试使用新版本文件名。 |
| 2026-09-14 05:05 | Codex (CINECA Leonardo) | 修订：B2b 锁后揭示包与摘要同步改为互异新路径和排他写入，保留先人后机联结证据。 |
| 2026-09-14 05:08 | Codex (CINECA Leonardo) | 修订：B2a 路由分析摘要改为新路径排他写入，防止三档来源裁决被原地替换。 |
| 2026-09-14 05:17 | Codex (CINECA Leonardo) | 修订：finalizer CLI 自身固定原始 32 包 hash/条数，并在 summary 保存 original/reviewed 两份输入 hash。 |
| 2026-09-14 05:30 | Codex (CINECA Leonardo) | 修订：加入 C4 标准库 loopback 界面启动命令、SSH 本地转发边界，以及 draft 可续写/final 排他创建规则。 |
| 2026-09-14 12:21 | Codex (CINECA Leonardo) | 修订：依据 1069/1070 隐藏可反查原题的来源身份，要求 draft/finalizer 严格 schema，并把图片校验绑定到实际响应字节。 |
