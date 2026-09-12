# Teacher-Gap 人工审核指引

**创建时间**：2026-09-12 14:45（UTC+8）  
**作者**：claude code (local)  
**触发原因**：W2 teacher-gap pilot 完成后需进行人工审核，提前准备指引文档

---

## 1. 审核目标

验证 W2 teacher-gap pilot 产出的候选集质量，确保：
1. 候选问题仅依赖图像和题面可答（无需外部知识）
2. 问题指代对完整图像成立（非 crop 视角）
3. 引导问题序列（guides）形成有效推理路径
4. 无答案泄漏（guides 不透露 final answer）

---

## 2. 审核材料

### 2.1 C2 aligned 包（1024 条）
位置：`outputs/runs/20260911T-c2-m3vqa-review-packet/outputs/`

| 文件 | 说明 |
|---|---|
| `review-aligned-1024.csv` | 1024 条 aligned 候选，`human_*` 字段待填 |
| `review-tier-128.csv` | C1 短名单 128 条 |
| `review-tier-512.csv` | 512 条扩展层 |
| `review-tier-1024.csv` | 1024 条扩展层 |

### 2.2 W2 筛选产物（待生成）
W2 screen 作业完成后，产物位于：
- `outputs/runs/20260911T*-teacher-gap-screen-*/outputs/`
- `outputs/runs/20260912T*-teacher-gap-screen-*-r2/outputs/`

分类标签：
- `image_only`：仅依赖图像可答（高优先级）
- `external_required`：需要外部知识（低优先级，可能需要重构）
- `ambiguous`：歧义或不确定（中优先级）

---

## 3. 审核优先级

### 第一优先：W2 screen 产物中的 `image_only` 候选
这些已通过机器筛选，确认不依赖外部知识，但仍需验证：
- 问题对完整图像成立
- 无歧义
- guides 有效且无泄漏

### 第二优先：C2 aligned 包的 borderline 样本
机器质检中被评为 borderline 的样本需要人工终审。

### 第三优先：`ambiguous` 候选
需要人工判断是否可 salvage。

---

## 4. 审核流程

### 4.1 打开 CSV 文件
使用 Excel、LibreOffice 或任何 CSV 编辑器打开 `review-*.csv`。

### 4.2 逐条审核
对每条记录，填写以下字段：

| 字段 | 说明 | 取值 |
|---|---|---|
| `human_knowledge_label` | 是否需要外部知识 | `none` / `some` / `heavy` |
| `human_task_family` | 任务类型 | `P` / `R` / `PR` |
| `human_unique_answer` | 答案是否唯一 | `yes` / `no` / `ambiguous` |
| `human_full_image_valid` | 对完整图像是否成立 | `yes` / `no` / `partial` |
| `human_required_evidence` | 必需的证据描述 | 自由文本 |
| `human_reasoning_graph` | 推理图（E→H→A） | 自由文本或 JSON |
| `human_guides_ok` | guides 是否有效 | `yes` / `no` / `partial` |
| `human_options_ok` | 选项是否合理 | `yes` / `no` / `partial` |
| `human_decision` | 最终决定 | `accept` / `borderline` / `reject` |
| `human_notes` | 备注 | 自由文本 |

### 4.3 使用审核表单
详细检查清单见 `docs/human_audit_form.md`。

---

## 5. 常见问题

### Q1: 如何判断"需要外部知识"？
如果回答该问题需要查询 Wikipedia、百科、地理/历史事实、或题面未提供的常识，则标记为 `external_required`。

### Q2: 如何判断"对完整图像成立"？
检查问题的指代是否依赖 crop 视角。例如：
- ❌ "bottom center 的物体是什么？"（依赖 crop 视角）
- ✅ "图中飞行员佩戴的耳机是什么颜色？"（对完整图像成立）

### Q3: 如何判断"答案泄漏"？
检查 guides 是否直接或间接透露 final answer。例如：
- Final: "耳机是什么颜色？"
- ❌ Guide: "它不是蓝色的。"（否定形式泄漏）
- ✅ Guide: "飞行员佩戴了什么头部设备？"（无泄漏）

---

## 6. 审核完成后

1. 保存 CSV 文件
2. 提交到 git：`git add research/main/data/review_templates/*.csv && git commit`
3. 推送：`git push`
4. 通知 admin 和 Codex

---

## 7. 引用

- 1017 号报告：teacher-gap 冻结协议
- 1019 号报告：C2 M3VQA 富化与人工审核包
- 1020 号报告：W2 滚动作业链清点
- `docs/human_audit_form.md`：详细审核表单
- `code/data_pipeline/build_m3vqa_teacher_gap_review_packet.py`：审核包生成脚本
