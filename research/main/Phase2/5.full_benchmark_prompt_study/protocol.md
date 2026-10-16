# 工作 5：全量评测集五种提示模式执行协议与门禁记录

状态：本文件为执行协议及可追溯的方法基线。数据revision、题数、提示词、模型/服务参数、原生评分与文本抽取链已实施并经相应门禁；全量推理与v4统计已完成。人工分层审阅仍未完成，因此需要人工审阅的解释性结论应保持审慎。当前证据状态见第9节及项目进展报告。

## 1. 研究问题与实验设计

在保持模型 checkpoint、benchmark 样本、图像、官方 user query、推理配置和评分流程一致时，逐级增加 system prompt 指令，如何影响 Vision-OPD 主评测集上的任务得分、回答格式、输出长度及类别级表现？

每条 benchmark 样本均在五种条件下各评测一次：S0、SC、S1、S2、S3。设计为逐题配对，不抽样、不重复扩充数据。S0 不添加 system message，保持原评测输入结构；其余条件仅增加 system message。原始 user query、图像及 benchmark 特有字段原样保留。每个结果使用 `(benchmark, stable_sample_uid, condition)` 唯一标识。

## 2. 范围

首要范围为 Vision-OPD 论文报告和官方运行示例明确列出的六个主 benchmark track：V* Bench、ZoomBench、HR-Bench 4K、HR-Bench 8K、MME-RealWorld EN、MME-RealWorld CN。MME-RealWorld-Lite 是独立的加速子集，不与 EN/CN 全量集混算；CV-Bench、MMStar、MMVP、POPE、VisualProbe 等属于仓库支持的额外评测/泛化集合，不纳入本轮“论文主评测集”总数。每个集合均以冻结 revision 的完整可评测记录为全集。

题数以固定版本下载、转换及完整数据门禁为准。服务器逐分片读取固定 revision 的源文件，六个track源行数为：V* 191、ZoomBench 845、HR-Bench 4K 800、HR-Bench 8K 800、MME-RealWorld EN 23,609（70个parquet shards）、CN 5,917（13个parquet shards），合计32,162 QA。已转换数据均通过唯一ID与媒体引用校验；全量完整性门禁重新核验32,162条唯一记录、32,162张全图与ZoomBench 845张crop图的媒体哈希。五条件对应160,810次生成。以上为本项目固定revision及本次prepared输入的口径，不应与其他数据转换版本混算。HR-Bench同一仓库的4K、8K分别计数；MME原始版与LMMs-Eval转换版的口径须区分。

## 3. 官方流程核验原则

以仓库 `reproduction/1. Vision-OPD/code/eval/` 的 `run_eval.sh`、`prepare_data.py`、`infer.py`、`judge_qwenlm.py`、`cal_acc.py` 及论文/官方说明为实现依据。冻结其 Git commit、依赖环境、数据集 revisions、模型 checkpoint 哈希/标识、tokenizer、图像编码路径、采样参数、thinking 设置、最大生成长度、并发/重试策略和 judge 配置。

按 benchmark 原有的数据加载、图像处理、任务答案空间和计分实现执行，不把不同任务统一强制改成 A–D：

- V* 与 HR-Bench 保留各自选择题选项及类别字段；
- ZoomBench 保留 MCQ/OQ 原始混合类型、`question_type`、原始 query/response，以及代码当前支持的 full-image/crop-image 输入语义；须核查 Vision-OPD 实际评测是否只将 full image 送入模型，不能把裁剪图误当作默认模型输入；
- MME EN/CN 保留其原生多选选项、语言、category/l2-category 等信息，并使用对应原生评分路径；逐行审计发现30条六项记录的标签序列为 A,B,C,D,D,E 且参考答案均为 E，作为原始 benchmark 异常样本保留并单独报告，不改写 user query 或标签；
- 官方 judge 若包含规则快速通道、LLM 语义判分或类别专用处理，逐条记录实际 `judge_source`、原始判分文本和失败状态。技术错误不得默认为答错、静默丢弃或在重试时覆盖历史原始记录。

已有研究已确认规则答案提取可能不可靠。因此，本工作既复现 Vision-OPD 原有主评分方法，也保留原始模型回答和评分器中间结论，并设计独立的审计/人工复核层：对每类评分路径及条件分层抽样，完整审阅高风险标签（格式异常、judge 分歧、API/解析错误、空答、类别边界样本）；报告中区分原生 benchmark 分数与审计修正/敏感性分析，未经验证不得把改写后的评分称为官方分数。

## 4. 五种提示条件

条件语义沿用工作2 `protocol.md` 与 `prompts/` 中已有定义；具体文件和 SHA-256 在冻结时归档。SC 为中性角色/任务说明；S1 为简洁表达；S2 要求按当前题型给出答案；S3 要求仅输出任务允许的最小答案。提示约束按任务答案空间适配，但不能更改题目、增加证据或向模型泄漏标准答案：

- 多选题：S2 要求给出所选选项；S3 仅输出题目要求的最小答案形式，不在 system prompt 中硬编码 A–D/A–E，也不根据选项数量推断标签范围；遇到重复/异常标签时沿用原始 user query 和 benchmark 答案空间；
- Yes/No、二元判断等任务：S2 要求明确给出该任务定义的标签；S3 只输出允许的二元标签；
- 开放式问答：S2 要求直接回答问题；S3 只给完成任务所需的最短答案，不强迫输出选项字母；
- ZoomBench 845条中，`question_type=mcq` 为621条，字面类别 `question_type=blank` 为224条；全量交叉审计发现前者参考答案全部为选项字母，后者全部为阿拉伯数字，且抽查原始 query 明确要求数字作答。分层统计保留源类别 `blank`，并可额外标记“数字短答”；不回填或修改源字段。

适配是为避免原有 S2/S3 字面约束与MCQ、数字短答等任务不兼容；五条件仍使用各一个固定 system prompt，约束题目回答形式的具体格式来自不变的官方 user query。条件按 `condition × benchmark/task_type × prompt_version` 报告。不得为不同类别临时改写 S2/S3 文本或为追求答案率调整提示。

## 5. 规模、计算与运行策略

设六个冻结 benchmark 行数为 `N_b`，预计模型生成请求数为 `5 × ΣN_b`；不得以当前网页近似规模确定 GPU/时长。完成数据清点后，按完整规模及原始高分辨率媒体路径估算存储、吞吐、推理时长和调度资源。评测含大量图像推理，优先在服务器运行；先提交真实小批 pilot 和并发/显存/吞吐测试，再以可恢复分片运行全量，不把抽样 pilot 当作正式结果。按 benchmark、条件、连续 shard 组织任务；每个 shard 有输入 manifest、配置哈希、开始/结束时间、Slurm job ID、原始日志、已完成/失败 ID 和 checkpoint。单个 benchmark-condition 失败只补该 shard，禁止静默跳过。

每个 shard 保留独立原始 JSONL；恢复逻辑不得破坏早期尝试记录。完成时逐 benchmark-condition 核验覆盖率为 100%、ID 唯一、输入键集合相同、无未解释错误、字段完整、原始响应可追溯。总请求数与预期数严格对账后方可进入统计。

## 6. 统计分析计划

先按 benchmark、官方子类别/任务、语言、题型报告样本数、覆盖率、官方指标及区间；严禁只给跨 benchmark 的单一准确率。对存在不同指标的任务使用其官方指标（例如二元任务的准确率/精确率/召回率/F1，开放题原生语义评分），不得不加说明地折叠成统一 accuracy。总体摘要同时给出 benchmark 等权宏平均和按 QA 数加权的微平均，且明确两者回答的问题不同。

对五种条件做同题配对比较：报告绝对值、配对差、标准化/实际效应量、95% 置信区间及多重比较校正；多条件总体检验后进行预先定义的成对比较。二元正确性使用 McNemar/配对广义线性模型等适配方法；以图像或上位任务为聚类单位做稳健 bootstrap/区间，避免同图多题时把所有 QA 当成完全独立。连续输出长度使用配对分布、稳健统计和适当非参数检验。分层分析考虑 benchmark 与其类别；探索性分析明确标注，不以显著性替代实际效应。

分析维度至少包括：benchmark 与类别表现、五模式配对转移/错误修正与错误引入、答案有效性与格式遵循、开放题评分、输出字符/token 长度及长尾、长度与得分的条件关联、题型/类别异质性、judge 源及人工审计分歧、跨 benchmark 效应一致性。报告图表与解释并置，全部中英双语；长表仅用于必要的附录/可折叠细节。

## 7. 质量门禁与停止条件

1. 数据门禁：固定 revisions 与哈希、完整行数、唯一键、图像/查询/参考答案/类别字段、题型映射均通过。
2. 流程门禁：S0 与官方原推理接口一致；提示词渲染不改变 user query/media；不同条件除 system message 外配置完全相同；ZoomBench 双视图实际输入行为已查明。
3. 判分门禁：按 benchmark 复现原生 judge；保存 judge 原始响应与来源；注入可预见边界样例检查解析；API/JSON/超时错误与真正错误分开编码。
4. Pilot 门禁：覆盖每种 benchmark × 题型/评分路径 × 五条件；审计输出和评分记录，修复方法问题后重新冻结；pilot 仅验证流程，不混入正式全集或若纳入则对所有条件统一纳入。
5. 全量门禁：五条件样本键完全相同，每个 benchmark-condition 无缺失/重复/未解释错误；失败全补齐；统计输入可从 raw 重建。
6. 报告门禁：源数据/推理配置/评分器/prompt/派生表/统计代码/图形版本一一可追溯；结论明确适用范围及推断限制。

若模型服务认证、checkpoint、存储、调度额度或数据授权不足，先完成所有本地可进行的代码审计、数据/字段清单和轻量校验，记录具体障碍与安全替代路径；不得伪称全量评测已开始/完成。

## 8. 记录与版本管理

所有中间产物按阶段保存，不覆盖原始输出。dataset/model/code/prompt/judge 每次变化均生成新版本并记录差异。Git 按文件/模块频繁提交，只暂存本工作明确生成的路径；保留已有工作区变更，不将其混入工作5提交。任何需要拉取或合并的操作遵循项目规范，必须使用 `--no-ff`；服务器文件仍通过 GitHub 中转，不通过 SSH 传输。报告及日志记录实际发生时间、来源、作者、触发原因和对应提交/作业编号。

## 9. 执行状态与已知边界（2026-10-16）

- **数据集与媒体 / Datasets and media：** 固定revision和文件清单见 `dataset_sources.tsv` 及服务器 `source-audit/`。六个prepared track共32,162条、32,162张full image、845张Zoom crop；完整媒体完整性复核通过。各track行数、prepared清单哈希保存在 `source-audit/prepared-integrity-verification.json`。
- **五条件推理 / Five-condition inference：** 全量输出门禁记录30个 condition×track 分片全部通过，每条件32,162条、总计160,810条；配对输入字段与prompt注册表哈希通过。机器可读结果为 `$DATA_ROOT/full/runs/prompt-registry-v1/output-verification.json`。
- **评分与全量统计 / Scoring and full statistics：** 原生评分、终行诊断及Qwen文本抽取在v4中分列；127个v4源输入哈希复核全部通过。派生CSV、分类统计、配对比较和报告位置见 `README.md` 与 `$DATA_ROOT/derived/full-statistics-v4/`。报告不是对视觉辅助审计或人工审核的替代。
- **分层复核 / Stratified review：** 固定风险样本清单包含六个track各10例、每例五条件共300条视图；60个原图均留存哈希。补充Qwen多模态审计已对v2的300条请求生成结果，2条模型原判与参考答案支持状态不一致，被显式保留原文并在派生集保守记为UNCERTAIN；详情见进展报告2221–2225及 `$DATA_ROOT/judgment/visual-audit-v2/`。该模型与Qwen文本抽取为同一权重，不能视为独立模型复现。研究者人工复核尚未完成，定向样本计数不得外推为总体错误率。
- **操作与仓库 / Operations and repositories：** home路径 `/leonardo/home/userexternal/rqian000/spinner-bot/MOPD-2026` 是服务器端主仓库；`MOPD-2026-work5` 是计算克隆。此次新增记录均按 `--no-ff` 同步，home中的外部数据、模型、Work4运行文件和历史报告保持原样。大体积benchmark原始数据、图片、完整模型结果均存于large scratch，不入Git。
