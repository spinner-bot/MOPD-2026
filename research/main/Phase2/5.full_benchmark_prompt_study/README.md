# Phase2 工作 5：全量评测集五种提示模式研究

本工作在 Vision-OPD 论文主评测集上，对同一模型进行五种条件的全量配对评测：S0（无 system prompt）以及 SC、S1、S2、S3 四种 system-prompt 条件。评测将遵循 Vision-OPD 的数据转换、图像输入、模型推理和各 benchmark 原生评分路径；所有结果保留 benchmark、官方子类别、语言/题型等分层信息。

服务器存储按项目约定分离：脚本/Git 工作树放 `/leonardo_scratch/fast/AIFAC_F02_774/rqian000/MOPD-2026-work5`；原始 benchmark 数据、图像、模型权重、完整推理输出和日志放 `/leonardo_scratch/large/userexternal/rqian000/MOPD-2026-work5-data`。`/leonardo_work/AIFAC_F02_774/rqian000` 当前所在 1 TB 挂载已满，不作为本工作写入位置。实际路径在节点启动前再次检查。

## 当前交付与目录约定

- `report/2130-T-261014-0101-Phase2工作5全量五模式评测实施方案.md`：启动阶段方法报告及工作流程；后续所有关键进展、偏差、结果与交付在本工作报告序列中续记。
- `report/2131-T-261014-0116-CINECA工作存储与GitHub连接诊断.md`：记录存储挂载异常及替代路径。
- `report/2132-T-261014-0136-数据源与服务器恢复检查.md`：记录本轮恢复后的服务器、模型、队列和固定 revision 数据审计进度。
- `protocol.md`：冻结前协议草案；数据版本、题数、提示词模板、模型/推理参数、评分器版本待核验后逐项冻结。
- `dataset_sources.tsv`：固定数据源 repository、revision 和待核验题数。
- `scripts/fetch_pinned_datasets.py`：在服务器直接从 Hugging Face 下载固定 revision 的 parquet，并输出行数、文件大小与 SHA-256 清单；不经 SSH 传输数据。
- `prompts/`：冻结后的条件提示词及按题型适配的模板；修改必须升版本并记录影响。
- `scripts/`：数据清点、完整性门禁、推理编排、汇总统计与报告生成脚本。
- `raw/`：完整原始回答、原生评分器原始输出、模型/API/Slurm 日志、数据下载 revision 和哈希。
- `derived/`：逐题五条件合并表、逐题评分、类别统计、检验与置信区间等可重建派生产物。
- `figures/`：统计图表及其生成源文件。
- `runs/`：运行清单、配置快照、提交/调度记录、校验清单和完成状态。
- `report/`：最终中英双语图文分析报告与必要的中间版本。

未经确认的题数或准确率不作为正式结果。大体积原图/数据按项目既有外部数据管理方式记录获取来源、固定 revision、文件清单和哈希；不得未经检查将许可受限的原始 benchmark 资产复制进 Git。
