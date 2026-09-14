# G-PC0 资料转移包

本目录是 G-PC0 的可查看资料副本，不依赖服务器运行。总计约 682 KB，包含 16 张程序化视觉任务图片、冻结输入、四条件/四轮换运行结果、运行日志、相关报告和关键代码。

## 快速查看

- [16 张图片预览](图片预览.md)
- 冻结实验材料：`runs/20260914T0916-teacher-gap-g-pc0-v1/outputs/manifest.jsonl`
- 标签轮换逐条结果：`runs/20260914T1003-teacher-gap-g-pc0-label-rotation-v1-run2/parsed/results.jsonl`
- 标签轮换汇总：`runs/20260914T1003-teacher-gap-g-pc0-label-rotation-v1-run2/report.md`
- 原始 G-PC0 报告：`reports/1043-G-PC0机制正控制预注册与冻结-260914.md`、`reports/1045-G-PC0机制失败与文本特权摄取链裁决-260914.md`
- 标签轮换报告：`reports/1046-G-PC0标签轮换诊断预注册与冻结-260914.md`、`reports/1047-G-PC0标签轮换恢复G信号与自然协议影响-260914.md`

## 目录说明

- `runs/20260914T0916-teacher-gap-g-pc0-v1/`：16 组冻结任务、16 张图片、引导 Q/A、测试 Q/A、正确答案和无关引导；
- `runs/20260914T0929-teacher-gap-g-pc0-v1-run1/`：首次运行，因配置错误 fail-closed；
- `runs/20260914T0939-teacher-gap-g-pc0-v1-run2/`：固定字母接口的修复重跑；
- `runs/20260914T0954-teacher-gap-g-pc0-label-rotation-v1/`：四轮答案标签轮换的冻结输入；
- `runs/20260914T1001-teacher-gap-g-pc0-label-rotation-v1-run1/`：短 commit 导致的 fail-closed 首次提交；
- `runs/20260914T1003-teacher-gap-g-pc0-label-rotation-v1-run2/`：最终 256 条轮换结果；
- `runs/20260914T1031-g-pc0-label-rotation-vote-audit/`：生成票和完全平局的描述性审计；
- `reports/`：G-PC0 预注册、运行故障、结果解释和修订报告；
- `code/`：运行 G-PC0 所需的关键代码副本。

## 解释边界

这些图片是程序化正控制，不是真实自然照片。资料包用于查看实验材料和复核结果，不代表自然图片实验，也不包含模型权重。当前自然 G0、G1 和 OPSD 训练仍未启动。

## 转移记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 | Codex (CINECA Leonardo) | 按用户要求将 G-PC0 的图片、输入、运行结果、报告和关键代码复制到 `research/main/data/数据转移/G-PC0`，便于导师离线查看。 |
