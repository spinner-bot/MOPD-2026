# 20260913T2320-teacher-gap-construct-c1-128-r4

阶段：W2B guide/option 构造。

目的：仅在 `20260913T2320-teacher-gap-screen-c1-128-r4` 全链通过后，对其严格筛选出的候选运行同一 r4 单次紧凑回退协议，产物仍须 100% 人工审核。

## 配置边界

4×A100、48 小时上限；固定提交；greedy、thinking=false；primary 768 token、fallback 512 token；`afterok` 依赖 screen job。

## 结果

待提交。screen 未通过时本作业应由依赖自动取消且不运行。
