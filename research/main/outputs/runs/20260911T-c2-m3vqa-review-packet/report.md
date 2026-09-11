# 20260911T-c2-m3vqa-review-packet

阶段：C2（1018 §5 分配给 claude code (local)；spinner-bot 手动启动）

目的：M3VQA raw-field enrichment + nested 128/512/1024 expansion packets + blank human review templates for W2 screening (1018 §5.3, owner: claude code (local)); deterministic, no semantic judgement, evidence-text firewall

## 结果

在 Leonardo（login02，CPU，无 Slurm 作业）以固定输入双跑，**全部产物 byte-identical**；本地按 manifest 复核 14/14 哈希一致。状态：**COMPLETED**（2026-09-11 13:24 UTC 前后）。

关键数字（详见 `outputs/packet-summary.json`）：

- 输入校验：raw `quesions.jsonl` sha256 `019d7bd1…a24a` **与 Codex 在 1015 §7 状态日志记录的下载哈希吻合**；normalized 13,125 条；image-root 模式核验 10,565 张图全部存在。
- 结构性拒绝：仅 `duplicate_source_image_id` 2,560（与 C1 完全一致）；无 join 缺失、无缺图、无路径越界。eligible = 10,565。
- 嵌套扩列（seed=20260911，hop≥2 分层，C1 同款比例分配+固定排列+单调钳制）：128 ⊆ 512 ⊆ 1024 验证通过；hop 分布 1024 档 = 2:329 / 3:353 / 4:198 / 5:137 / 6:7。
- **对齐包**（`--aligned`，按 Codex 冻结名单 `20260911T1910-m3vqa-expanded-1024`）：1,024 条，与 C1-128 重合 50、新增 974——与 Codex 档案 summary 完全一致；每条带 `in_c1_shortlist` 标志。该包使人工审核集合与 W2A GPU 筛选集合严格对齐。
- 人工审核模板（CSV 带 BOM/Excel 友好 + JSONL，每档一套）：10 个 `human_*` 字段**全部留空**（禁止伪造人工审核）。
- evidence 正文防火墙：运行时逐条自检 + 测试断言，泄漏检查 passed；定位元数据仅入 `posthoc_analysis_only`（带 1017 §3.2 警告标签）。
- 合约测试：本地 12/12 通过（含 W2A `validate_candidate` 兼容性——富化记录可原样作 `run_teacher_gap_candidate_stage.py --input`）。

完整分析、两套扩列的口径裁决请求与人工审核指引见 `进展报告/1019-T-2026-09-11-C2-M3VQA富化与人工审核包.md`。
