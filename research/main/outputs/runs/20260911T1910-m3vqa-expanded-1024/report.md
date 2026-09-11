# 20260911T1910-m3vqa-expanded-1024

阶段：W2A_INPUT

目的：Build a deterministic 1024-record screening-only M3VQA expansion with the frozen C1 rules and seed

## 结果

使用与 C1 相同的 seed=20260911、hop≥2 分层、缺失排除与唯一 `source_image_id` 规则，生成 1,024 条**仅供语义筛选**的核心字段扩列，并以两个临时目录双跑比对。它不是 image-only/R 真值，也不替代 C2 的 raw 字段富化和人工审核模板。

结果：双跑四个原生产物 byte-identical；1,024 条的 candidate/source-image ID 均唯一，hop 分布为 2:329、3:353、4:198、5:137、6:7。由于固定 seed 下 `random.sample(k)` 的不同 `k` 不保证嵌套，它与原 128 池重合 50 条；另输出按 candidate ID 排序的 974 条差集供下一批筛选，避免重复 GPU 推理。全部产物 SHA-256 已记录。状态：**COMPLETED**。
