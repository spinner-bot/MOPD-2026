# 20260914T0308-image-affordance-b1-256-r3

阶段：支路 B1，仅看完整图片的 affordance machine proposal。固定输入仍为同一 256 图，SHA-256 `e69b6d14fe8da625720af1ee7198213b909c682c10e2313003f43c2c139debf4`；源问题及答案继续隔离。

## 目的

r2 canary 的 8 条 primary 均能输出可读视觉描述，但 6 条把自身仅评为 `r_score=1` 的弱关系仍标为 candidate；旧 fallback 锁死该无效正例，空数组骨架又诱发字符串列表、浮点分数和重复截断。r3 不放宽 validator，只做三项校正：无效 candidate 可降级为 reject/ambiguous；compact prompt 明示完整嵌套对象和整数分数；normal/compact prompt 排除“识别两个对象后复述一个简单空间关系”作为两步 R。

## 结果

Slurm `57599758` 于 2026-09-13 21:12:00–21:18:36 CEST 在 `lrdn1759` 运行。canary 达到 7/8 严格有效（87.5%）、零最终截断、零 runner error；3 条触发 fallback，2 条恢复。结构 gate 因而通过并启动 full，但在 full 运行 24 秒、四个 rank 仍均为 0 条输出时，由 Codex 主动取消。human_verified=0。

取消依据不是结构失败，而是逐条语义复核发现 7 个有效 proposal 全部选择 candidate。多个路径只是把简单关系拆成两个措辞步骤，例如“识别餐盘和炉灶→判断餐盘在前”“识别两个座椅→判断相邻”“识别神庙和狗→判断前后景”；这不满足两项不可缺的推理变换。prompt 新增的文字排除规则没有改变 9B 的这种评分行为，它只是把 r2 的 `r_score=1` 改成 2。继续全量会得到高召回但几乎无排序信息的机器候选，并增加 256 条人工负担。

因此 r3 证明了嵌套 JSON 和 fallback 结构问题已解决，也同时否定“让同一模型自由描述后自评 R 分数”作为 affordance 主筛选器。后继应把可供性判断绑定到一个具体、可验证的闭世界目标及反事实必要性检查，或先用小规模人类图像审核建立校准集；不再继续用分数阈值反复重提同一配方。
