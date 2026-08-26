# tmux 使用指南（训练必备）

> 用途：让训练在后台运行，SSH 断开也不会中断。

---

## 一、为什么用 tmux

| 场景 | 不用 tmux | 用 tmux |
|------|----------|---------|
| SSH 断开 | 训练死亡 ❌ | 训练继续 ✅ |
| 笔记本合盖 | 训练死亡 ❌ | 训练继续 ✅ |
| 网络波动 | 训练死亡 ❌ | 训练继续 ✅ |
| 43 小时长任务 | 必死无疑 ❌ | 稳稳运行 ✅ |

**结论**：跑训练必须用 tmux。

---

## 二、基本操作

### 2.1 创建会话
```bash
tmux new -s 会话名
```
- 示例：`tmux new -s vopd`（创建一个叫 vopd 的会话）
- 效果：进入一个"受保护的终端"

### 2.2 在 tmux 里跑程序
```bash
# 和平时一样，该干嘛干嘛
cd ~/vopd
python train.py
```
- 程序在 tmux 里运行，不会因为 SSH 断开而死

### 2.3 脱离 tmux（让程序后台跑）
**按 `Ctrl+B`，松开，再按 `D`**

- `Ctrl+B`：tmux 命令前缀（告诉 tmux "我要下指令了"）
- `D`：detach（脱离）
- 效果：回到普通终端，tmux 里的程序继续跑

### 2.4 查看正在运行的会话
```bash
tmux ls
```
- 输出示例：`vopd: 1 windows (created Wed Aug 26 13:40:00 2026)`

### 2.5 重新进入 tmux
```bash
tmux attach -t 会话名
```
- 示例：`tmux attach -t vopd`
- 效果：回到 tmux，看到程序实时输出

### 2.6 关闭 tmux 会话
```bash
# 在 tmux 里
exit
# 或者
Ctrl+D
```
- 效果：结束 tmux 和里面的程序

---

## 三、快捷键速查

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 脱离 tmux | `Ctrl+B` 然后 `D` | 程序继续跑 |
| 重新进入 | `tmux attach -t 会话名` | 回到会话 |
| 查看所有会话 | `tmux ls` | 列出会话 |
| 关闭会话 | `exit` 或 `Ctrl+D` | 在 tmux 里执行 |

---

## 四、完整示例（跑训练）

```bash
# 1. 创建 tmux 会话
tmux new -s vopd

# 2. 进环境 + 跑训练
cd ~/vopd
conda activate vision-opd
# ... 设环境变量 ...
python train.py 2>&1 | tee train.log

# 3. 脱离 tmux（让训练后台跑）
按 Ctrl+B，松开，再按 D

# 4. 回普通终端，可以关 SSH 了
# 训练在 tmux 里继续跑，43 小时没问题

# 5. 想查看进度
tail -40 ~/vopd/train.log

# 6. 想实时看输出
tmux attach -t vopd
```

---

## 五、常见问题

### Q1: 脱离 tmux 后，程序还在跑吗？
**在跑**。`Ctrl+B D` 只是离开 tmux，程序在后台继续运行。

### Q2: SSH 断了，tmux 里的程序会死吗？
**不会**。tmux 是独立于 SSH 的，SSH 断了 tmux 还在。

### Q3: 怎么知道 tmux 里还有几个会话？
```bash
tmux ls
```

### Q4: 不小心关了 tmux，程序能恢复吗？
**不能**。`exit` 或 `Ctrl+D` 会结束 tmux 和里面的程序。**脱离用 `Ctrl+B D`，别用 exit**。

### Q5: 多个会话怎么切换？
```bash
# 在 tmux 里，按 Ctrl+B 然后按 W
# 会弹出会话列表，用方向键选择
```

---

## 六、一句话总结

- **跑长任务**：`tmux new -s 名字` → 跑程序 → `Ctrl+B D` 脱离 → 关 SSH
- **回来看**：`tmux attach -t 名字`
- **忘脱离了**：`tmux ls` 看会话，`tmux attach` 回去

**记住**：`Ctrl+B D` = 安全离开，`exit` = 结束一切。
