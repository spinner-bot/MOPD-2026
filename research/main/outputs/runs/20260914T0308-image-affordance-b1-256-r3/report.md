# 20260914T0308-image-affordance-b1-256-r3

阶段：支路 B1，仅看完整图片的 affordance machine proposal。固定输入仍为同一 256 图，SHA-256 `e69b6d14fe8da625720af1ee7198213b909c682c10e2313003f43c2c139debf4`；源问题及答案继续隔离。

## 目的

r2 canary 的 8 条 primary 均能输出可读视觉描述，但 6 条把自身仅评为 `r_score=1` 的弱关系仍标为 candidate；旧 fallback 锁死该无效正例，空数组骨架又诱发字符串列表、浮点分数和重复截断。r3 不放宽 validator，只做三项校正：无效 candidate 可降级为 reject/ambiguous；compact prompt 明示完整嵌套对象和整数分数；normal/compact prompt 排除“识别两个对象后复述一个简单空间关系”作为两步 R。

## 结果

待运行。作业在 r8 结束后再申请 4×A100。所有输出均为机器建议，human_verified=0；full 只有在 canary 结构通过后运行。
