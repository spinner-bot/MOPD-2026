# 20260911T-c1-m3vqa-field-audit

阶段：C1（1014 §4.2 分配给 claude code (local)）

目的：M3VQA 字段清单、外部知识确定性线索评估、固定 seed 128 条分层候选短名单——为 1016 闭世界筛选（W2）与 locked pilot 提供输入。

## 结果

- 池：13,125 条（仅 M3VQA；Vision-OPD 支路因 payload 授权删除且 P 层待决而空置）；字段缺失 0
- 排除：重复 source_image_id 2,560 → eligible 10,565
- 短名单：128 条（hop2:41/hop3:44/hop4:25/hop5:17/hop6:1/hop7:0），ID 与源图均唯一
- 代表记录：16 条；确定性线索结论：URL 域名线索判别力为零（100% wikimedia），语义筛选不可省略
- 验收：--help ✅ 实跑 ✅ 双跑哈希一致 ✅ 计数/唯一性 ✅
- 产物哈希（seed=20260911）：shortlist 3dcf3aced17e5b5f6dbc19c64ba19fe75dc51db451b3ab539925d9a52c16fd7d；field_audit 313643ffa76c4ab75f8555eacd479f2da0efc7b87153f68ba734e541212973d2；representatives 6818d9a38ab8e2e6065eaaba8f356bd0512d71292a86d70151e8e7cfd793f06e

完整分析、schema 缺口（raw 8 字段被 normalized 丢弃）与对 1016 的先验提示见 `进展报告/1015-T-260911-M3VQA字段与候选抽样审计.md`。
