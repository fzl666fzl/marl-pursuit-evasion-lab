# marl-pursuit-evasion-lab

如果你是第一次接触这个项目，先看零基础说明：[BEGINNER_GUIDE.md](BEGINNER_GUIDE.md)。

轻量、可复现、适合本科生简历展示的多智能体追捕-逃跑实验项目。项目基于 MPE2 `simple_tag_v3`，训练 3 个追捕者协作追捕 1 个随机逃跑者。

核心设定：

- 环境：`mpe2.simple_tag_v3`
- 追捕者：`adversary_0`、`adversary_1`、`adversary_2`
- 逃跑者：`agent_0`，固定随机策略
- 算法：共享参数的 Independent DQN、DQN + 课程学习 + 团队奖励混合、MAPPO baseline
- 改进：课程学习、团队奖励混合、集中式 critic

## Installation

建议使用 Python 3.12。Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[dev]
```

快速检查：

```powershell
python -m pytest
python -m pursuit_lab.train --config configs/smoke_test.yaml
```

也可以不改 YAML，直接覆盖常用训练参数：

```powershell
python -m pursuit_lab.train --config configs/dqn_baseline.yaml --episodes 100 --experiment-name quick_dqn
```

## Experiments

一键跑三组默认实验并生成图表/CSV：

```powershell
python -m pursuit_lab.run_experiments
```

快速 smoke 版也可以用同一个入口覆盖训练轮数：

```powershell
python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/smoke_test.yaml --episodes 10
```

随机基线：

```powershell
python -m pursuit_lab.train --config configs/random_baseline.yaml
```

DQN 基线：

```powershell
python -m pursuit_lab.train --config configs/dqn_baseline.yaml
```

DQN + 课程学习 + 团队奖励混合：

```powershell
python -m pursuit_lab.train --config configs/dqn_curriculum_team_reward.yaml
```

默认训练配置为 CPU 平衡版，每组 3000 episodes。想先确认流程，用 `configs/smoke_test.yaml`。

## Evaluation And Rendering

评估 checkpoint：

```powershell
python -m pursuit_lab.evaluate --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 50
```

批量复评某次完整训练下所有 `best.pt` checkpoint：

```powershell
python -m pursuit_lab.evaluate_runs --runs runs/post_seedfix_full_20260606_164815 --output runs/post_seedfix_best_eval_500 --episodes 500
```

生成 GIF：

```powershell
python -m pursuit_lab.render --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 3
```

GIF 使用清晰版配色：红色 `P0/P1/P2` 是追捕者，绿色 `E` 是逃跑者，灰色 `OBS` 是固定障碍物。

生成图表：

```powershell
python -m pursuit_lab.plot --runs runs/
```

输出文件：

- `runs/<experiment>/metrics.csv`
- `runs/<experiment>/best.pt`
- `runs/<experiment>/config.yaml`
- `runs/<experiment>/eval_summary.json`
- `runs/<experiment>/eval_episodes.csv`
- `figures/training_curve.png`
- `figures/eval_comparison.png`
- `figures/eval_comparison.csv`
- `videos/pursuit_demo.gif`

## Metrics

主要评估指标：

- `capture_rate`：成功追捕比例
- `mean_episode_reward`：每个 episode 的追捕者原始奖励总和均值
- `mean_steps_to_capture`：成功追捕 episode 的平均步数；无成功时记为 `max_cycles`
- `success_episode_count`：成功追捕 episode 数

注意：团队奖励混合只用于训练 replay transition。评估始终使用 MPE2 原始奖励，避免指标作弊。

## Expected Report Table

完成默认训练后，可以把 `eval_summary.json` 汇总成类似表格：
`python -m pursuit_lab.plot --runs runs/` 会同时生成 `figures/eval_comparison.csv`，可直接复制到报告或 README。

| Experiment | capture_rate | mean_episode_reward | mean_steps_to_capture | success_episode_count |
| --- | ---: | ---: | ---: | ---: |
| Random baseline | run result | run result | run result | run result |
| DQN baseline | run result | run result | run result | run result |
| DQN + curriculum + team reward | run result | run result | run result | run result |
| MAPPO baseline | run result | run result | run result | run result |

验收目标：学习策略应优于随机基线；改进版 DQN 如与普通 DQN 成功率持平，则比较 `mean_steps_to_capture`；MAPPO 用于观察集中式 critic 是否提升协作追捕成功率。

## Visual Outputs

训练和评估后，README 中引用的演示资源路径如下：

![Training curve](figures/training_curve.png)

![Evaluation comparison](figures/eval_comparison.png)

![Pursuit demo](videos/pursuit_demo.gif)

## Resume Bullet

基于 MPE2/PettingZoo 构建多智能体追逃实验平台，实现 Independent DQN、DQN + 课程学习 + 团队奖励混合和 MAPPO baseline，对比追捕成功率、平均奖励与平均抓捕步数，并验证集中式 critic 对协作追捕的提升。

## Notes

- MPE2 已接替旧版 PettingZoo MPE 环境；本项目直接使用 `from mpe2 import simple_tag_v3`。
- v1 只训练追捕者，不训练逃跑者。
- 本项目默认 CPU 可运行，不依赖 Isaac Sim、ROS 或真实无人机。
