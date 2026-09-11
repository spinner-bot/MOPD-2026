# Text-OPD 代码目录

> 所有实验代码集中存放于此，确保版本控制和可追溯性

---

## 目录结构

```
code/
├── README.md                    # 本文件
├── data_processing/             # 数据处理
│   ├── generate_guiding_questions.py   # 引导性问题生成
│   └── quality_check.py                # 质量检查
├── training/                    # 训练代码（待创建）
│   ├── run_baseline.sh
│   ├── run_text_opd.sh
│   └── configs/
├── evaluation/                  # 评估代码（待创建）
│   ├── evaluate.py
│   └── analyze_results.py
└── utils/                       # 工具函数（待创建）
    └── ...
```

---

## 使用指南

### 1. 生成引导性问题

```bash
python code/data_processing/generate_guiding_questions.py \
    --input-file data/raw/train.jsonl \
    --output-file data/guiding_questions/raw_generated/batch_001.json \
    --batch-size 100 \
    --api-key sk-xxx \
    --start-idx 0 \
    --end-idx 100
```

**参数说明**：
- `--input-file`: 输入 JSONL 文件（Vision-OPD 格式）
- `--output-file`: 输出 JSON 文件
- `--batch-size`: 批次大小（默认 100）
- `--api-key`: DeepSeek API key
- `--start-idx`: 起始索引（默认 0）
- `--end-idx`: 结束索引（默认 None，表示 batch_size）

### 2. 质量检查

```bash
python code/data_processing/quality_check.py \
    --input-file data/guiding_questions/raw_generated/batch_001.json \
    --output-report data/quality_reports/batch_001_report.json
```

**检查项**：
- 生成成功率（目标 > 95%）
- 平均问题数量（目标 2-4 个）
- 问题长度分布
- 重复检测
- 异常值检测

---

## 开发规范

### 代码风格
- 使用 Python 3.12+
- 遵循 PEP 8
- 函数和类要有 docstring
- 关键逻辑要有注释

### 版本控制
- 每个功能模块独立 commit
- commit message 清晰描述改动
- 大文件使用 .gitignore 排除

### 测试
- 关键功能要有单元测试
- 数据处理要有验证步骤
- 质量检查要自动化

---

## 待完成任务

- [ ] 训练代码（基于 Vision-OPD 修改）
- [ ] 评估代码
- [ ] 可视化工具
- [ ] 数据合并脚本

---

**最后更新**：2026-09-04
