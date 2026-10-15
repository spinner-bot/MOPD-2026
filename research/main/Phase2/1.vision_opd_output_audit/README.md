# Vision-OPD 输出机制审计

本目录用于执行 A101 中导师布置的第一项 Phase2 工作：对已训练 Vision-OPD 模型的真实输出进行确定性抽样和人工审计。

## 当前问题

只回答以下问题：

1. 输出是否包含有效视觉证据或推理内容；
2. 这些内容是有效推理、无效格式性文本，还是直接给出选择项；
3. 最终选择项是否在推理/证据文本之前已经明确；
4. 输出中的推理/证据是否可能影响最终判别。

该审计不直接判断 Vision-OPD 论文是否正确，也不替代完整 benchmark 复现。

## 目录约定

```text
vision_opd_output_audit/
├── README.md
├── sample_and_prepare_audit.py
├── protocol.md
├── inputs/       # 远程运行产生的输入清单或其哈希，不放大型原始数据
├── runs/         # 每次抽样/推理/解析的独立档案
└── review/       # 人工审计表、案例记录和汇总
```

大型模型、benchmark 图像和完整输出保留在服务器指定存储区或独立 run archive；不复制进 Git。

## 使用方式

```bash
python sample_and_prepare_audit.py \
  --input /path/to/model_answer.jsonl \
  --output review/audit-sample.jsonl \
  --n 100 \
  --seed 261010
```

脚本只做确定性抽样、输出结构提取和人工字段初始化，不对证据、推理或答案正确性作自动科学判定。
