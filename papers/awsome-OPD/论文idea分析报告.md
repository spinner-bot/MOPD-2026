# 论文Idea分析报告：AwesomeOPD 逐篇深度拆解
> 数据源: https://github.com/thinkwee/AwesomeOPD
> 目标：学习每篇论文的idea提出角度和思路模式

---

## [001] GKD — Agarwal et al. (Google DeepMind, ICLR 2024)

### Idea提出角度：**"框架命名者"角度**

GKD的核心贡献不是发明了on-policy蒸馏（之前已有相关工作），而是：
1. **命名并形式化**了"On-Policy Distillation for LMs"这个范式
2. **系统性地比较**了on-policy vs off-policy、FKL vs RKL vs JSD的多种组合
3. 提出**Generalized JSD**作为统一散度框架

### 思维链
```
知识蒸馏在CV成功 → LM蒸馏还停留在off-policy SFT
    ↓
问题: off-policy蒸馏有exposure bias，学生没见过自己的错误状态
    ↓
方案: 让学生在自生成轨迹上接受教师监督
    ↓
贡献: 不是"发明"而是"命名+系统验证"这个范式
```

### 可借鉴思路
- **"第一个命名者"策略**: 如果你发现一个正在兴起但还没人系统定义的方向，做第一个命名和形式化它的人。ICLR对这种"命名+系统验证"的论文接受度很高。
- **比较空间设计**: GKD的实验设计核心是在{on-policy, off-policy} × {FKL, RKL, JSD}的格点上做系统比较——这种"格点消融"是最有说服力的实验设计。

