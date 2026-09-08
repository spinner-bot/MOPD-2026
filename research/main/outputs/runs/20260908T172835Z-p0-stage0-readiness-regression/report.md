# 20260908T172835Z-p0-stage0-readiness-regression

阶段：P0

目的：修复并验证 Stage 0 的推理深度、样本隔离和 Go/No-Go 判定。

## 结果

6 项合成回归检查全部通过。该结果仅证明 CPU 侧基础逻辑可执行，不能替代真实模型的 Stage 0 screening，也不构成论文实验结论。

## 变更与判断

- `ReasoningDepth` 的枚举继承错误已修复；直接证据到答案的路径为 depth 1。
- Stage 0 在图像级与 train、validation 和 diagnostic test 隔离，避免方法选择污染确认性测试。
- D4 统一记作协议定义的 D4+；覆盖检查只要求 P/R/PR、四种策略和四个深度维度分别出现，不再错误要求所有 48 个交叉组合。
- Go/No-Go 在缺条件、缺 depth≥3 样本、缺 paired significance 或控制组不成立时均返回 No-Go。

## 后续

当前 `data/` 没有正式样本，且 main 内没有已验证的 checkpoint/processor，因此尚不能提交真实 GPU 推理。下一阶段先建立数据与模型的不可变清单、九条件输入构造和推理运行入口。
