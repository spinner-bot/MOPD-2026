# Phase2 工作 5：全量评测集五种提示模式研究

本工作在 Vision-OPD 论文主评测集上，对同一模型进行五种条件的全量配对评测：S0（无 system prompt）以及 SC、S1、S2、S3 四种 system-prompt 条件。评测将遵循 Vision-OPD 的数据转换、图像输入、模型推理和各 benchmark 原生评分路径；所有结果保留 benchmark、官方子类别、语言/题型等分层信息。

服务器存储按项目约定分离：脚本/Git 工作树放 `/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5`；原始 benchmark 数据、图像、模型权重、完整推理输出和日志放 `/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data`。`/leonardo_work/AIFAC_F02_774/rqian000` 当前所在 1 TB 挂载已满，不作为本工作写入位置。实际路径在节点启动前再次检查。

## 当前交付与目录约定

- `report/2130-T-261014-0101-Phase2工作5全量五模式评测实施方案.md`：启动阶段方法报告及工作流程；后续所有关键进展、偏差、结果与交付在本工作报告序列中续记。
- `report/2131-T-261014-0116-CINECA工作存储与GitHub连接诊断.md`：记录存储挂载异常及替代路径。
- `report/2132-T-261014-0136-数据源与服务器恢复检查.md`：记录本轮恢复后的服务器、模型、队列和固定 revision 数据审计进度。
- `report/2133-T-261014-0155-固定Revision数据清点与题数更正.md`：记录六个 track 固定 revision 的逐分片下载、源 parquet 行数及初版 HR 统计更正。
- `report/2134-T-261014-0201-固定版本数据清单校验.md`：记录修订后的最终源行数、文件清单和哈希校验结果。
- `report/2135-T-261014-0209-样本键与任务字段审计.md`：记录全样本主键唯一性、题型、类别与答案空间审计。
- `report/2136-T-261014-0216-题型边界复核与五条件提示词冻结.md`：记录Zoom/MME题型边界逐项复核及v1提示词冻结。
- `report/2137-T-261014-0229-转换调度与推理接口验证.md`：记录Slurm CPU账户限制、可用GPU分区替代、转换器与推理接口契约测试。
- `report/2138-T-261014-0242-全量输入转换完成与题型元数据更正.md`：记录32,162条源记录转换及Zoom `blank` 类别语义更正。
- `report/2139-T-261014-0252-五条件全量任务编排与试运行进度.md`：记录pilot真实模型作业的门禁执行状态及五条件单卡数组任务编排。
- `report/2140-T-261014-0305-全量评测数组预提交与pilot状态.md`：记录五条件全量数组的依赖式预提交、资源试算和pilot实时进展。
- `report/2141-T-261014-0322-图像完整性门禁通过与pilot时限续跑.md`：记录全量图像门禁结果、debug QoS超时、长时pilot续跑和全量数组的新依赖关系。
- `report/2142-T-261014-0330-全量答案抽取与评分复核管线.md`：记录覆盖A–F、重复标签和数字短答的文本判别设计、断点续跑与自动化测试。
- `report/2143-T-261014-0340-全量验证冻结与两条评分链预提交.md`：记录全量后处理队列链、Vision-OPD原生评分任务及当前Slurm依赖状态。
- `report/2144-T-261014-0345-原生评分队列提交与三链调度确认.md`：记录官方评分数组提交、远端语法校验及完整后处理依赖图。
- `report/2145-T-261014-0342-推理Token用量留存与接口测试.md`：记录原始推理输出增加服务端token用量留存、接口测试与当前队列状态。
- `protocol.md`：冻结前协议草案；数据版本、题数、提示词模板、模型/推理参数、评分器版本待核验后逐项冻结。
- `dataset_sources.tsv`：固定数据源 repository、revision 和待核验题数。
- `scripts/fetch_pinned_datasets.py`：在服务器直接从 Hugging Face 下载固定 revision 的 parquet，并输出行数、文件大小与 SHA-256 清单；不经 SSH 传输数据。
- `scripts/audit_dataset_schema.py`：只读取 parquet schema 和类别/任务字段（排除图像载荷），输出用于转换方案复核的字段及其分布。
- `scripts/audit_sample_keys_and_answers.py`：扫描主键、参考答案、选项数及题型/类别分布，不读取图像载荷。
- `scripts/audit_task_edge_cases.py`：交叉核对Zoom题型/答案格式并完整导出MME六选项边界题。
- `scripts/prepare_pinned_benchmarks.py`：读取固定revision parquet，逐批提取图像并复现官方query构造；保留稳定UID和类别元数据。
- `scripts/prepare_pinned_benchmarks.sbatch`：在账户允许的GPU分区运行CPU为主的固定版数据转换（1张卡用于满足分区准入）。
- `scripts/test_infer_prompt_conditions.py`：以模拟API做无GPU契约测试，校验五条件消息结构与服务端prompt/completion/total token用量的逐条留存。
- `scripts/make_stratified_pilot.py`：确定性选择覆盖六个track、主类别、Zoom题型及MME六选项异常的小样本pilot。
- `scripts/run_stratified_pilot.sbatch`：单A100启动vLLM，并通过官方infer接口跑五条件pilot。
- `scripts/run_full_conditions.sbatch`：五元素Slurm数组，每个任务独占一张A100运行一个提示条件；四任务并发，输出保存在large scratch并可利用官方推理脚本的逐题检查点恢复。
- `scripts/verify_prepared_benchmarks.py`：对转换后QA覆盖、UID、图片引用与全部媒体SHA-256执行独立完整性门禁。
- `scripts/verify_inference_outputs.py`：核验逐条件完整覆盖、无API失败、消息条件哈希及各条件输入完全配对。
- `scripts/verify_and_freeze_full_outputs.sbatch`：五条件数组完成后执行全量输出门禁，并冻结30个条件×track答案抽取输入分片。
- `scripts/prepare_answer_extraction.py`：生成文本判别输入；模型payload不含图像、gold、条件、UID或类别。
- `prompts/answer_extraction_v1.txt`：初版工作5答案抽取指令，支持A–F选项及数字短答；保留用于历史判别尝试的版本追溯。
- `prompts/answer_extraction_v2.txt`：针对残留无效记录的严格短证据判别指令，要求非答案状态证据为空；补跑结果逐条记录提示词SHA-256、模型配置SHA-256与抽取代码提交。
- `scripts/run_answer_extraction.py`：用本地Qwen3.8-27B作单分片结构化文本答案抽取，记录证据和状态并支持断点续跑。
- `scripts/run_answer_extraction_array.sbatch`：五条件×六track共30个单卡判别任务，最多8个并发。
- `scripts/run_visionopd_native_scorer.sbatch`：在五条件×六track分片上调用未修改的Vision-OPD `judge_qwenlm.py` 与 `cal_acc.py`，保存原生评分作为独立对照。
- `tests/test_answer_extraction_contract.py`：检查选项标签解析、重复D、标签白名单、数字短答证据和无答案状态。
- `audits/`：由服务器生成并经 GitHub 中转的小型字段审计中间产物；原始图片和数据不入 Git。
- `prompts/`：冻结后的条件提示词及按题型适配的模板；修改必须升版本并记录影响。
- `prompts/prompt_registry-v1.json`：首版五条件完整提示词注册表；S0 的 `null` 表示不发送 system message。修改需升版本并重新记录SHA-256。
- `scripts/`：数据清点、完整性门禁、推理编排、汇总统计与报告生成脚本。
- `raw/`：完整原始回答、原生评分器原始输出、模型/API/Slurm 日志、数据下载 revision 和哈希。
- `derived/`：逐题五条件合并表、逐题评分、类别统计、检验与置信区间等可重建派生产物。
- `figures/`：统计图表及其生成源文件。
- `runs/`：运行清单、配置快照、提交/调度记录、校验清单和完成状态。
- `report/`：最终中英双语图文分析报告与必要的中间版本。

## 全量统计与裁判输出解析

统计v3区分三种评分：①按未修改的 Vision-OPD `cal_acc.py` 规则对完整 `judge` 字段执行 `strip().lower() == "yes"` 的官方分数；②仅从裁判输出最终非空行精确解析Yes/No的诊断分数，其他记为 `other` 且不算答对；③本地Qwen纯文本答案抽取分数。分析会逐一核对全部30个条件×评测集的官方 `cal_acc.py` 日志和原始裁判记录，任一不符即失败。完整裁判原文仍保存在逐题合并CSV；报告分别呈现三种评分，不用诊断分数替换官方分数。统计v3输出位于服务器 `derived/full-statistics-v3/`，最终自包含中英双语报告为 `report/full-five-condition-benchmark-report-v3.html`；项目级进展记录位于 `research/main/进展报告/`。

`full-statistics-v1` 和 `full-statistics-v2` 是发现原生裁判字段包含推理文本、与官方精确标签规则不同后生成的历史审计快照，不可引用为最终官方统计结论；派生产物保留在服务器以追溯评分解释更正过程，不覆盖。

未经确认的题数或准确率不作为正式结果。大体积原图/数据按项目既有外部数据管理方式记录获取来源、固定 revision、文件清单和哈希；不得未经检查将许可受限的原始 benchmark 资产复制进 Git。
