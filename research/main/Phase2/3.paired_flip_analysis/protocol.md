# 配对翻转分析协议

## 1. 固定范围

- 基线：S0；
- 目标条件：S1、S3；
- 排除条件：SC、S2；
- 配对单位：`sample_id`；
- 数据来源：`../2.short_output_prompt_study/` 的正式作业 `60011513`；
- 核心正确性字段：既有 `per_sample.csv` 的 `correct`；
- 长度指标：正式 vLLM `usage.completion_tokens`，字符数作为补充指标。

## 2. 翻转定义

对每个目标条件分别相对 S0 分类：

- `gain`：S0 错误、目标条件正确；
- `loss`：S0 正确、目标条件错误；
- `stable_correct`：两者均正确；
- `stable_wrong`：两者均错误。

交叉分析额外定义：

- S1/S3 共同 gain；
- S1/S3 共同 loss；
- S1-only gain/loss；
- S3-only gain/loss；
- S1 与 S3 方向不一致。

## 3. 分析门禁

1. S0、S1、S3 的 `sample_id` 集合必须完全一致；
2. 每个条件必须有 2000 条记录；
3. 每个 `(sample_id, condition)` 必须唯一；
4. 不重新推断 gold answer 或覆盖既有 `correct`；
5. 每个翻转题目必须能回溯到 S0、S1、S3 的原始输出；
6. SC、S2 不得进入本工作统计结果；
7. 题目属性细分样本量过小时只作描述，不进行不稳定的显著性检验。

## 4. 交付范围

- 全部翻转题目明细；
- S1/S3 分别的 gain/loss 汇总；
- 共同与特异翻转分析；
- 题目属性、输出 token 和原始输出行为分析；
- 图表、可读 HTML 报告和运行记录。
