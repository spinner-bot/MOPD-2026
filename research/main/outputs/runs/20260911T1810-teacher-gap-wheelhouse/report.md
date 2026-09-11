# 20260911T1810-teacher-gap-wheelhouse

阶段：INFRA

目的：Download the pinned Stage-0 Python wheelhouse on a networked login node for offline compute-node installation

## 结果

2026-09-11 18:10（UTC+8）在可联网登录节点完成固定依赖下载。离线 wheelhouse 位于
`/leonardo_work/AIFAC_F02_774/sqs/cache/wheelhouse-stage0-py311`，共 85 个文件、
3,041,728,082 bytes。

`pip --dry-run --ignore-installed --no-index` 已仅使用该目录成功解析
`requirements-stage0.lock` 的全部 85 个固定版本；完整文件清单、逐文件 SHA-256、
下载日志与离线解析日志均已收入本档案。状态：**COMPLETED**。
