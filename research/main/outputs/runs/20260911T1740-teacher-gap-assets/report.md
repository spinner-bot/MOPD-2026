# 20260911T1740-teacher-gap-assets

阶段：INFRA

目的：Restore pinned base Qwen3.5-4B and Qwen3.5-9B under current WORK storage for protocol 1016

## 结果

2026-09-11 17:43（UTC+8）完成。

- 当前存储根：`/leonardo_work/AIFAC_F02_774/sqs`；未写入已删除的旧 payload 根。
- base Qwen3.5-4B：revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，目录占用约 8.7 GiB。
- base Qwen3.5-9B：revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，目录占用约 19 GiB。
- 两个目录共登记 30 个非缓存文件；逐文件 SHA-256 见 `outputs/model-files.sha256`。
- 4B/9B `config.json` SHA-256 分别为 `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`、`d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`。
- 下载由固定 revision 驱动；HF CLI 提示未认证速率限制，但下载与哈希步骤均正常完成。

这些目录与已有的 `OPD-V-Qwen3.5-4B` 明确分离；1016 pilot 只允许使用本档案登记的 base 4B。
