# 20260914T0248-image-affordance-b1-256

阶段：支路 B1，仅看完整图片的 affordance machine proposal。固定输入 256 图，SHA-256 `e69b6d14fe8da625720af1ee7198213b909c682c10e2313003f43c2c139debf4`。不展示原问题、答案、evidence、URL 或 teacher-gap 结果。

## 结果

Slurm `57597609` 于 2026-09-13 20:52:52–20:53:12 CEST 在 `lrdn2923` 启动，并在模型加载前以 `FAILED (2:0)` 终止。预检发现显式传入的 Python 路径 `/leonardo_scratch/large/userexternal/rqian000/envs/mopd-stage0-py311/bin/python` 不存在；模型与图片路径存在。未运行 canary 或 full，生成记录为 0，因此这不是模型能力、数据质量或 B1 门禁结果。

失败原因是新存储布局中环境位于 fast scratch，而提交命令错误地指向 large scratch。后继修复只更正确定性运行时路径，并使用新 run ID 与新固定 commit 重提。所有未来有效输出仍标记为待独立人工复核，不能直接成为 R/PR 或 task-family 真值。
