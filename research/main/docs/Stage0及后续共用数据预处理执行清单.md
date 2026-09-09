# Stage 0 及后续实验共用数据预处理执行清单

本清单覆盖 Stage 0、Stage 1、主实验和分层分析的共同数据基础。所有候选、模型生成结果、人工修订和冻结版本都必须保留；Stage 0 只能使用其中通过全部准入门槛的独立 screening 子集。

## 角色与模型分工

| 标记 | 执行者 | 允许的工作 | 不允许的工作 |
|---|---|---|---|
| D | 确定性脚本 | 解压、哈希、解析、去重、切分、模板渲染、统计、格式和路径检查 | 语义标注、选择正确答案、判断推理链有效性 |
| B | Qwen3.5-4B 基础模型 | 批量转写候选格式、抽取已有字段、生成待审清单、按既定规则做机械分类 | 单独决定 crop、最终答案、guide 有效性、P/R/PR 或是否通过 |
| H | Qwen3.5-9B 高性能模型 | 图像/证据结构分析、latent reasoning graph、guide QA、干扰项建议、语义验证、边界样本复审 | 覆盖来源事实、在证据不足时猜测、替代人工抽查 |
| M | 人工审核 | 边界样本和分层抽查的最终判定、歧义/泄漏/版权问题处理 | 无记录地直接修改冻结数据 |

所有 B/H GPU 工作使用 Slurm 单节点的最大可用 GPU 数；同一批候选的生成和验证记录固定模型 revision、prompt hash、解码参数、GPU 数和随机种子。B/H 的输出只能是候选或审计意见，不能绕过 D 和 M 的准入门槛。

## 必须执行的模型调用

下表不是建议，而是数据进入下一阶段前必须完成的实际模型调用。所有调用均读取本地固定 revision 的 Qwen3.5-9B，通过 Slurm 多卡分片运行；每条原始响应保存到对应 run archive 的 `outputs/`，解析后的 JSON 另存，二者不得相互覆盖。

| 调用 ID | 所属任务 | 模型输入 | 模型必须输出 | 调用后确定性处理 |
|---|---|---|---|---|
| H-T02 | T02 | 原图、来源 crop/bbox、最终问题 | evidence bbox、视觉清晰度/依赖度、接受或拒绝 | JSON/bbox/图像边界检查，生成 crop 或 reject |
| H-T03 | T03 | T02 合格图像/crop、最终题、来源证据 | latent graph、P/R/PR、策略、1–4 个 guide QA、深度、拒绝原因 | schema、答案泄漏、guide 完整性与路径检查 |
| H-T04 | T04 | T03 合格样本、自由回答来源的参考答案与证据 | 三个可排除干扰项、唯一答案判断、拒绝原因 | 固定 A/B/C/D 置换，生成 deterministic controls 与 token 审计 |
| H-T05 | T05 | 完整候选和渲染后的九条件文本；不提供生成阶段的思维过程 | answerability、dependency、relevance、leakage、ambiguity、grounding 分数及 accept/borderline/reject | 阈值过滤、边界样本进入人工审核、生成拒绝原因码 |

每个调用采用 `do_sample=false` 的首轮；只有为一致性检查明确登记的复核轮才可更换 seed。H-T03 和 H-T05 即使使用同一 checkpoint，也必须使用不同 prompt、不同输出目录和不同 Slurm run。

## 固定目录和版本

大文件只写入 Leonardo Work：

```text
/leonardo_work/AIFAC_F02_774/rqian000/MOPD-2026/payload/data/processed/<data_version>/
  00_sources/       # 来源记录与不可变索引
  01_assets/        # 已解压图像、crop、缩略图、哈希
  02_candidates/    # 尚未采纳的模型候选
  03_verified/      # 自动验证通过、等待人工抽查
  04_human_review/  # 人工审计表与修订记录
  05_frozen/        # 唯一允许进入实验的 JSONL 和 manifest
  rejects/          # 带拒绝原因的样本，绝不静默删除
```

`research/main/data/processed/<data_version>/` 仅保存指向 Work 的路径映射、版本 manifest、样本 ID 清单和哈希。每个生成或审计步骤创建独立 `run_id`，并按 `docs/运行与留档规范.md` 保存完整 run archive。

版本只允许单向推进：`candidate` → `verified` → `human_reviewed` → `frozen`。冻结后修订必须建立新版本，原版本不可覆盖。

## 任务清单

### T00：来源与资产登记（D，低难度）

输入：下载归档、Vision-OPD、M3VQA 及其他已下载来源。

执行：

1. 记录来源 revision、许可证说明、文件 SHA-256、大小和解压命令。
2. 解压图像到 `01_assets/`；保存原文件名、来源相对路径、尺寸、RGB 模式和内容哈希。
3. 为每个来源建立 `source_index.jsonl`，至少包含 `source_dataset`、`source_record_id`、`source_image_id`、原图路径和哈希。
4. 对同一来源和跨来源计算 SHA-256；对图像计算 perceptual hash。重复或近重复记录进入 `rejects/duplicates.jsonl`，不进入后续切分。

验收：每条 source record 都能解析到存在的图像；重复表和资产 manifest 均有哈希；缺失文件数为零。

### T01：候选池分层（D + B，低难度）

输入：T00 的 `source_index.jsonl`、Vision-OPD 的题目/crop/bbox、M3VQA 的问题/证据。

执行：

1. Vision-OPD 进入 P 候选池，保留其原题、四选项、答案、bbox、教师 crop 和原图。
2. M3VQA 进入 R 候选池，保留问题、参考答案、每跳 evidence、URL 与图片；尚未存在真实 crop 的记录必须标记 `needs_grounded_crop=true`。
3. B 仅将原始字段转换为统一 schema，不补写事实、不生成答案。缺字段即写入 `missing_fields`。
4. 用固定 seed 在 P/R/PR、四种核心策略、depth 1–4+ 的目标配额上抽取候选；每个候选保留抽样理由和 seed。

输出：`02_candidates/source-normalized.jsonl`、`candidate-strata.json`、`rejects/source-invalid.jsonl`。

验收：候选记录可追溯到一条来源记录；任何来源未提供的字段都显式为空或标记待补，不得虚构。

### T02：视觉结构和 crop 候选（H，较高难度）

输入：原图、来源 bbox（如有）、最终问题、来源证据。

对每条候选让 H 输出严格 JSON：

```json
{
  "decision": "accept_candidate|reject",
  "evidence_regions": [{"bbox_xyxy": [0, 0, 0, 0], "role": "required_evidence", "why": "..."}],
  "crop_primary_index": 0,
  "visual_dependency": "low|medium|high",
  "visual_clarity": "low|medium|high",
  "reason": "..."
}
```

约束：bbox 必须在图像边界内；若图像不提供完成最终问题所需的证据，输出 `reject`；不得以全文图冒充 crop。M3VQA 的外部文本 evidence 不能替代图像 evidence；若任务本质依赖外部知识，必须在 R 类元数据中明确，并由后续设计决定是否保留。

随后 D 根据 bbox 生成 PNG crop，记录 bbox、原图哈希、crop 哈希、面积比例和生成命令。B 只检查 JSON、边界、文件存在性和面积比例。

验收：每条待采纳记录都有可打开的原图与 crop；crop 不越界；目标区域明显不相关、空白或等于整图的候选进入 `rejects/crop-invalid.jsonl`。

### T03：latent graph 与 guide QA 候选（H，最高难度）

输入：T02 合格的图像/crop、最终问题、选项和仅允许使用的来源证据。

H 的输出必须是下列 JSON，禁止输出自由文本：

```json
{
  "decision": "accept_candidate|reject",
  "task_family": "P|R|PR",
  "guidance_strategy": "localization|entity_tracking|logical_bridge|evidence_integration",
  "required_evidence": ["E1", "E2"],
  "intermediate_states": ["H1", "H2"],
  "guides": [
    {"question": "...", "answer": "...", "supports_next": "H1 or final"}
  ],
  "final_answer_leaked_by_guide": false,
  "reasoning_depth": 1,
  "rejection_reason": ""
}
```

生成提示必须包含以下不可违反的规则：

1. 先确定 final question，再构造 `E → H → answer` 图；
2. guide 只对应中间状态，任何 guide answer 不得给出最终答案、最终选项文本或其同义改写；
3. 每个 guide 必须可以由图像、crop 或前一个中间状态回答；
4. 不确定、无唯一答案或无真实依赖时必须 `reject`；
5. 每条保留模型原始输出、prompt hash、模型 revision、temperature、seed。

P 允许视觉定位链；R 必须有不可省略的中间推理；PR 同时需要真实视觉瓶颈和推理链。B 不得代替 H 生成此步骤。

验收：每条 `accept_candidate` 有 1–4 个完整 QA 对、明确策略、深度和 evidence graph；其它记录保留原样进入 rejects。

### T04：四选项与随机控制（H + D + B，较高难度）

输入：T03 合格候选。

执行：

1. 已有四选项的来源保留原始选项和顺序；H 只检查唯一性与歧义。
2. 自由回答来源由 H 提出三个语义上合理、但可由证据排除的干扰项；不得使用明显荒谬选项。D 固定 A/B/C/D 随机置换，并以固定 seed 记录正确答案位置。
3. H 再次盲审题干与选项：确认只有一个正确答案；不通过则 reject。
4. D 从同图不相关、已验证的 guide 池按固定 seed 构建 `random_questions`；使问题数与 token 数接近真实 guide。若同图没有足够不相关 guide，候选不得进入正式 Stage 0。
5. D 生成 `T_shuffle`（同一 guide 内容、固定乱序）和 `T_generic`，不允许模型临时改写。

验收：每条有唯一正确答案、冻结 random/shuffle/generic 控制、token 长度审计；所有控制文本和 seed 写入样本记录。

### T05：独立自动验证（H + D，最高难度）

输入：T03–T04 的完整候选，但不给验证 prompt 生成阶段的推理文本以外的信息。

H 针对每条独立输出以下分数：

```json
{
  "answerability": 0,
  "logical_dependency": 0,
  "relevance": 0,
  "answer_leakage": 0,
  "ambiguity": 0,
  "visual_grounding": 0,
  "decision": "accept|borderline|reject",
  "reasons": ["..."]
}
```

固定规则：answerability、relevance、visual_grounding 必须为高；logical_dependency 至少中等；answer_leakage 必须为 0；答案必须唯一。若生成与验证使用同一 9B checkpoint，必须使用独立 prompt、独立 seed、隔离输出文件，并将同模型限制写入 run archive；所有 borderline 样本进入人工审核。

D 还执行：schema/path 检查、选项唯一性、bbox 边界、guide 与答案字符串/规范化同义词泄漏检查、重复图像检查、同一图像 random 对照检查和样本 ID 唯一性检查。

输出：`03_verified/auto-accepted.jsonl`、`04_human_review/borderline.jsonl`、`rejects/verification-failed.jsonl`。

### T06：人工分层审核（M，较高难度）

输入：自动通过样本的随机 5–10%，全部 borderline 样本，以及 P/R/PR、四策略、depth 1–4+ 每层的最低抽样量。

每个审核者逐条记录：最终答案正确性、guide 的可回答性/相关性/依赖性、答案泄漏、crop 覆盖、唯一答案、P/R/PR 分类和深度。记录 `reviewer_id`、时间、准则版本、原判定、新判定和理由。两个审核者不一致时进入仲裁，不得静默覆盖。

验收：所有分层均达到抽查比例；borderline 清零；人工确认的泄漏、歧义和错误记录移入 rejects；输出一致率或 Cohen's kappa（若有双审核）。

### T07：冻结正式数据与无泄漏切分（D，较高难度）

输入：T06 通过样本。

执行：

1. 按 source image、场景和近重复簇分组，再切分为 train 8K、validation 1K、diagnostic/test 2K；Stage 0 从独立 screening 预留，不得与三者共享图像或近重复场景。
2. Stage 0 固定 400–600 条，覆盖 P/R/PR、四种策略、depth 1–4+；筛选 seed 和实际每层计数写入 manifest。
3. 使用 `validate_formal_stage0_dataset`、`split_dataset.py`、泄漏检测和输入渲染验证器；任何错误都阻断冻结。
4. 对每个 JSONL、图像索引、prompt、模型 revision 和控制文本生成 SHA-256；输出 `05_frozen/manifest.json`、`splits.jsonl`、`checksums.sha256`、`rejected_ids.jsonl`。

验收：所有正式 Stage 0 条件可渲染；无图像/近重复泄漏；所有必需层覆盖；哈希、版本和样本列表完整。仅此目录允许被 GPU runner 读取。

### T08：Stage 0 与后续训练的共用消费接口（D，低难度）

输入：T07 的 frozen manifest。

执行：实现统一 loader，按 `sample_id` 返回原图、crop、最终问题、选项、答案、九条件文本和元数据；禁止各阶段重新解析来源或现场生成 controls。Stage 0 读取独立 screening split；Stage 1/主实验读取相应 train/validation/diagnostic split，同一 `data_version` 仅以 manifest 指针传递。

验收：同一 `sample_id` 在不同进程、不同重跑中生成完全相同的九条件输入；所有运行 archive 记录 manifest SHA-256。

## 执行顺序与阻断关系

```text
T00 → T01 → T02 → T03 → T04 → T05 → T06 → T07 → T08
                     │                         │
                     └── 不通过 → rejects ────┘
```

T00/T01 可立即推进；T02/T03/T05 使用 H 并可按候选分片并行；T04 的确定性部分、T06 的审核排队和 T07 的切分准备可与高性能模型批次交错进行。T07 未通过前，不提交正式 Stage 0 GPU screening；T08 完成后，同一冻结版本供全部后续阶段复用。

## GPU 排队期间的并行工作

GPU 作业处于 `PENDING` 时，继续执行下列任务；它们不读取尚未生成的 H 模型结论，也不占用 GPU。每项仍单独建 run archive、提交代码与结果摘要。

| 可先做的工作 | 所属任务 | 执行者 | 输入与产物 | GPU 结果到达后的衔接 |
|---|---|---|---|---|
| 跨来源 SHA-256 与 perceptual-hash 去重 | T00 | D | 固定资产清单 → 重复/近重复簇与拒绝清单 | T07 直接以重复簇为切分组 |
| 来源配额、候选分层和固定 seed 审计 | T01 | D + B | candidate pool → P/R/待判 PR、策略/深度缺口表 | T02/T03 以固定 shard 和缺口优先级处理 |
| T02 原始输出校验与 rejects 分流 | T02 | D | 每个 shard 原始 JSONL → bbox/JSON 校验结果 | 9B 作业完成后立即运行，不等待人工整理 |
| latent graph/guide QA JSON schema、解析器和泄漏检查 | T03 | D | 方案字段与提示模板 → 可执行 schema validator | T02 合格候选作为唯一输入 |
| 选项冻结、置换、random/shuffle/generic 控制与 token 长度审计 | T04 | D + B | 完整 guide 候选的接口和固定 seed 规则 | T03 完成后直接批量执行；H 仅处理语义性干扰项 |
| 自动验证量表、边界样本队列和拒绝原因码 | T05 | D | 验证 JSON schema、阈值和输出格式 | T03/T04 完成后交 H 批量判断，D 汇总 |
| 人工审核表、分层抽样和仲裁记录模板 | T06 | D | `verified` 的预期字段 → 审核表与抽样 seed | T05 输出后立即生成审核批次 |
| 图像级/近重复级无泄漏切分与冻结 manifest 的预演 | T07 | D | source index、重复簇、目标比例 → dry-run split 报告 | 仅在接受样本确定后写入正式 frozen 版本 |
| 统一 loader、九条件输入导出和断点续跑接口 | T08 | D | frozen manifest 的接口契约 → loader/渲染/消费测试 | T07 冻结后仅替换 manifest 路径，不重新解析来源 |

### 滚动提交规则

1. 只有输入、代码、资源和独立输出目录均已冻结的真实 GPU 任务才可提前提交。
2. 有上游依赖的 GPU 任务使用 `afterok` 预提交；上游失败时下游保持不运行。
3. 同一工作保持一个有效作业；取消后必须在 run archive 记录作业 ID、原因和替代作业 ID。
4. 为提高 backfill 机会，墙钟时间必须基于已完成的 smoke test 吞吐量填写，不以虚短时间换取调度位置。
5. 不得使用空作业、重复作业或等待人工输入的作业占位；它们既不构成预约，也可能损害公平份额。

## 每个任务的最低留档

每个任务必须保存：输入 manifest、提交时的 Git commit、模型/脚本版本、实际命令、Slurm 资源、prompt 文件及哈希、原始模型输出、接受/拒绝 ID、拒绝原因、输出哈希、`summary.json` 和 `report.md`。模型输出中含有样本内容时保存在 Work payload，并在 main 的 run archive 保存不可变路径与 SHA-256。
