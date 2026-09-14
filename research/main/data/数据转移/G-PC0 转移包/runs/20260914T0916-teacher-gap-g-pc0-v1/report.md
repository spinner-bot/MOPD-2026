# 20260914T0916-teacher-gap-g-pc0-v1

**阶段**：G-PC0 程序化机制正控制冻结；对应 1035 §4.1 与 1043。

**创建时间**：2026-09-14 09:16（UTC+8）

**最后修改时间**：2026-09-14 09:23（UTC+8）

**作者**：Codex (CINECA Leonardo)

**触发原因**：B2a 等待独立人工盲审期间，完成一个不读取自然 G0 结果、只检验 4B 与输入实现能否利用正确中间状态的机制健康检查。

## 冻结结果

固定 seed `20260914` 一次生成 16 题，包含有向路径、顺序变换、网格轨迹、集合关系各 4 题。正确标签按样本顺序固定为 A/B/C/D 循环，因此每个题族各含一次 A/B/C/D，全包各标签均为 4。每题包含独立 PNG、四选项、唯一答案、至少三步的 solution trace、required evidence、DAG 与 1–4 个 guides。

最终 manifest SHA-256 为 `10b09d122a5781df564a63cac8449d368a2d785c3fb06ba15c3887db2433bb1c`，generation spec SHA-256 为 `085fca89b3f76d426fbaf060e74b557e16ffdd3e7a0550300af1fbd8104093b0`。spec 内同时冻结生成器、G-PC0 合同、两份字体和 Pillow 版本的哈希或版本；16 张图片各自的哈希见 `outputs/summary.json`。

CPU preflight 使用本地冻结 Qwen3.5-4B tokenizer：A/B/C/D 分别编码为 `[32]`、`[33]`、`[34]`、`[35]`；T-Q 与 Generic-T 的 corpus median 均为 49 tokens，相对差 0，通过 10% 门槛。16/16 记录通过 profile、DAG、标签/题族平衡、图片路径和图片 checksum 检查。测试过程没有读取任何 S/T 性能结果，未使用 GPU。

## 语义审计与修订

正式冻结前逐类查看了代表图片并逐题展开 question、options、guides 和 solution trace。第一次视觉检查发现路径箭头尖端被节点遮挡、顺序变换页脚越界，随后在结果运行前修正并全量重绘。逐题逻辑检查又发现两个集合题中的集合 C 不改变中间结果，使该证据可删除；最终版本改为 C 必须移除至少一个交集成员，且其中一题移除全部成员。旧版本只保留在本机 `/tmp`，从未提交、从未运行条件，也不属于冻结包。

这些检查证明生成合同与图像字节一致，不等于人类语义签名。所有记录明确保持 `human_verified=false`、`procedural_gold_verified=true`。

## 运行边界

当前状态是“设计、16 题包、CPU gate、GPU 脚本均就绪；0 条性能结果”。`submit_g_pc0.sbatch` 已固定相同 4B revision、S/T-Q/T-QA/Generic-T shell、choice scoring、非 thinking 确定性生成和结果绑定，但本档案没有授权或提交该作业。

未来只允许在新的 run ID 下完整运行一次 16×4。冻结裁决为：S≥14/16 得 `CEILING_INCONCLUSIVE`；否则 T-QA 对 S 的错→对≥4 且净翻转≥3 得 `MECHANISM_PASS`；其余为 `INGESTION_OR_RECIPE_FAILURE`。T-Q 和四题族结果只作描述，不做显著性推断。

G-PC0 不能进入自然 G0 的 8/32/96，不能进入训练数据，不能证明 M3VQA 或自然图片上的 teacher gap，也不能替代 B2a/B3–B5 人工验收。它只能帮助解释未来自然 G0 的弱或零效应究竟更像“G 摄取链失败”还是“自然任务缺少稳定瓶颈”。

## 修订记录

| 时间（UTC+8） | 作者 | 触发原因 |
|---|---|---|
| 2026-09-14 09:16 | Codex (CINECA Leonardo) | 创建：生成 16 题 G-PC0 v1 候选冻结包。 |
| 2026-09-14 09:20 | Codex (CINECA Leonardo) | 修订：根据代表图片检查修复箭头方向可见性与页脚越界，加入生成器/合同代码哈希。 |
| 2026-09-14 09:22 | Codex (CINECA Leonardo) | 修订：实际 4B tokenizer 显示 Generic-T 仅 38 vs T-Q 49 tokens；结果前扩展通用文本至 49 tokens 并重跑 preflight。 |
| 2026-09-14 09:23 | Codex (CINECA Leonardo) | 修订：逐题最小性审计发现两条集合题的 C 可删除；改为 C 必然改变中间集合后完成最终冻结。 |
