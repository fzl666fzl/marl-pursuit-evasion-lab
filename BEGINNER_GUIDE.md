# 零基础使用说明：多智能体追捕-逃跑实验

这份文档是给第一次接触这个项目的人看的。你不需要先懂强化学习，也不需要先懂很多代码。你只要按顺序做，就能知道这个项目在做什么、怎么运行、结果怎么看。

---

## 1. 这个项目到底在做什么？

这个项目模拟一个简单的追捕游戏：

- 有 3 个追捕者：`adversary_0`、`adversary_1`、`adversary_2`
- 有 1 个逃跑者：`agent_0`
- 追捕者的目标：尽快碰到逃跑者
- 逃跑者的策略：随机乱跑
- 追捕者的策略：用 DQN 算法训练出来

你可以把它理解成：

> 三个小机器人在二维平面里合作抓一个随机逃跑的小机器人。

这个项目要比较三种方法：

1. `random_baseline`：追捕者也随机乱走。
2. `dqn_baseline`：追捕者用 DQN 学习追捕。
3. `dqn_curriculum_team_reward`：在 DQN 基础上加一点改进，让训练更容易、更强调团队协作。

最后我们看哪种方法更容易抓到逃跑者。

---

## 2. 你需要先知道几个文件夹

项目目录是：

```text
D:\marl-pursuit-evasion-lab
```

常用文件夹如下：

```text
configs/     放实验配置文件
src/         放项目代码
tests/       放测试代码
runs/        放实验运行结果
figures/     放生成的图
videos/      放生成的 GIF 动画
```

你主要会用到这些文件：

```text
configs/smoke_test.yaml
configs/random_baseline.yaml
configs/dqn_baseline.yaml
configs/dqn_curriculum_team_reward.yaml

figures/eval_comparison.png
figures/training_curve.png
figures/eval_comparison.csv

videos/pursuit_demo.gif
```

---

## 3. 第一次运行前要做什么？

先打开 PowerShell 或 VS Code 终端。

进入项目目录：

```powershell
cd D:\marl-pursuit-evasion-lab
```

以后所有命令都在这个目录里运行。

这个项目已经有 `.venv` 虚拟环境，所以你可以直接用：

```powershell
.\.venv\Scripts\python
```

你不一定要激活虚拟环境。只要命令前面写 `.\.venv\Scripts\python`，就会使用项目自己的 Python。

---

## 4. 第一步：先检查项目能不能正常运行

运行：

```powershell
.\.venv\Scripts\python -m pytest -q
```

如果看到类似：

```text
10 passed
```

说明项目代码没有明显问题。

如果没有看到 `passed`，先不要继续训练，把错误截图或复制出来再处理。

---

## 5. 第二步：先跑一个很短的测试版实验

不要一上来跑正式实验，因为正式实验会比较久。

先运行这个：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/smoke_test.yaml --episodes 10
```

这条命令的意思是：

- 跑一个随机策略实验
- 跑一个很短的 DQN 测试实验
- 训练轮数只用 10 轮
- 自动生成结果图和表格

运行结束后，你会看到类似输出：

```text
runs\random_baseline
runs\smoke_dqn
figures\training_curve.png
figures\eval_comparison.png
figures\eval_comparison.csv
```

这说明结果已经生成了。

---

## 6. 第三步：打开结果图

打开对比图：

```powershell
start figures\eval_comparison.png
```

打开训练曲线：

```powershell
start figures\training_curve.png
```

打开结果表格：

```powershell
start figures\eval_comparison.csv
```

如果你在 VS Code 里点开图片看不到，不一定是图没生成。可以用上面的 `start` 命令用系统图片查看器打开。

---

## 7. 图应该怎么看？

### `eval_comparison.png`

这张图用来看不同实验的追捕效果。

你重点看两个东西：

```text
capture_rate
mean_steps_to_capture
```

含义：

- `capture_rate`：成功抓到逃跑者的比例，越高越好。
- `mean_steps_to_capture`：平均多少步抓到逃跑者，越低越好。

如果一个方法：

- `capture_rate` 更高
- 或者 `capture_rate` 差不多但 `mean_steps_to_capture` 更低

就说明它表现更好。

### `training_curve.png`

这张图用来看训练过程中追捕者的奖励变化。

简单理解：

- 曲线越来越高：训练可能在变好。
- 曲线一直乱跳：训练不稳定。
- smoke 测试版训练太短，曲线不一定明显。

### `eval_comparison.csv`

这是表格版结果。可以放进报告或 README。

里面一般有这些列：

```text
experiment
episodes
capture_rate
mean_episode_reward
mean_steps_to_capture
success_episode_count
```

---

## 8. 第四步：生成追捕动画

如果你已经跑过 smoke 测试，可以生成一个简单 GIF：

```powershell
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/smoke_dqn/best.pt --episodes 1
```

然后打开：

```powershell
start videos\pursuit_demo.gif
```

这个 GIF 就是追捕者和逃跑者在环境里的运动过程。

GIF 里的颜色含义：

```text
红色 P0/P1/P2 = 追捕者
绿色 E        = 逃跑者
灰色 OBS      = 障碍物
```

注意：障碍物不会动。如果你看到旧版 GIF 里有黑色边缘在动，那是默认渲染产生的深色描边，不是障碍物在动。现在项目已经改成清晰版渲染，障碍物会显示成灰色 `OBS`。

---

## 9. 正式实验怎么跑？

确认 smoke 测试没问题后，再跑正式实验：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments
```

这条命令会自动跑三组实验：

```text
random_baseline
dqn_baseline
dqn_curriculum_team_reward
```

正式实验会比 smoke 测试慢很多，因为默认每个 DQN 实验会训练 3000 个 episode。

如果你只是想先做一个中等长度实验，可以这样：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --episodes 300
```

这会把 DQN 训练轮数改成 300，速度比正式版快。

---

## 10. 三种实验分别是什么意思？

### 1. `random_baseline`

追捕者随机行动。

它的作用是当作最低标准：

> 如果 DQN 连随机策略都比不过，那说明训练失败。

### 2. `dqn_baseline`

追捕者使用 DQN 学习。

DQN 会根据观察到的环境状态，学习每一步应该往哪个方向走。

### 3. `dqn_curriculum_team_reward`

这是改进版 DQN。

它多了两个小改进：

1. 课程学习：先从简单难度开始，再逐渐变难。
2. 团队奖励：不仅看每个追捕者自己的奖励，也看整个追捕团队的平均表现。

它的目标是让训练更稳定，追捕成功率更高。

---

## 11. 常用命令汇总

检查项目：

```powershell
.\.venv\Scripts\python -m pytest -q
```

跑快速测试：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/smoke_test.yaml --episodes 10
```

跑正式三组实验：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments
```

跑中等长度实验：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --episodes 300
```

打开对比图：

```powershell
start figures\eval_comparison.png
```

打开训练曲线：

```powershell
start figures\training_curve.png
```

打开表格：

```powershell
start figures\eval_comparison.csv
```

生成 GIF：

```powershell
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 3
```

打开 GIF：

```powershell
start videos\pursuit_demo.gif
```

---

## 12. 运行后你应该交什么？

如果这是课程项目，最后可以交这些东西：

```text
README.md
BEGINNER_GUIDE.md
figures/eval_comparison.png
figures/training_curve.png
figures/eval_comparison.csv
videos/pursuit_demo.gif
runs/dqn_baseline/eval_summary.json
runs/dqn_curriculum_team_reward/eval_summary.json
```

报告里可以写：

> 本项目基于 MPE2 simple_tag_v3 构建多智能体追逃实验。实验设置 3 个追捕者和 1 个随机逃跑者，比较随机策略、DQN 基线和 DQN 改进策略。改进策略引入课程学习和团队奖励混合机制，并使用 capture_rate、mean_episode_reward、mean_steps_to_capture 等指标进行评估。

---

## 13. 如果你完全不知道下一步干什么

按这个最短流程做：

```powershell
cd D:\marl-pursuit-evasion-lab
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/smoke_test.yaml --episodes 10
start figures\eval_comparison.png
start figures\training_curve.png
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/smoke_dqn/best.pt --episodes 1
start videos\pursuit_demo.gif
```

做完这几步，你就已经完成了：

- 检查项目
- 跑实验
- 生成图
- 生成动画
- 看结果

之后再决定要不要跑正式 3000 episode 实验。
