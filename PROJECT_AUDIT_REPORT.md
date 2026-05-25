# MARL 追逃项目审计报告

生成日期：2026-05-21

## 1. 项目一句话介绍

本项目基于 MPE2 / PettingZoo `simple_tag_v3` 构建了一个轻量级多智能体追逃实验平台，用 3 个追捕者学习围捕 1 个随机逃跑者，并比较随机基线、普通 DQN、DQN + 团队奖励混合 + 课程学习，以及 MAPPO。

## 2. 当前环境设置

项目环境不是离散网格地图，而是二维连续空间：

- 环境：`mpe2.simple_tag_v3`
- 追捕者：3 个，`adversary_0`、`adversary_1`、`adversary_2`
- 逃跑者：1 个，`agent_0`
- 障碍物：2 个
- 动作空间：离散 5 动作
- 每局上限：`max_cycles=50`
- 初始坐标：智能体大致采样在 `[-1, 1] x [-1, 1]`
- 评估指标：`capture_rate`、`mean_episode_reward`、`mean_steps_to_capture`、`success_episode_count`

`max_cycles=50` 对这个小尺度 `simple_tag` 环境可以用于快速实验，但对严谨对比来说偏紧。它会让未抓捕成功的 episode 很快截断，因此更适合简历展示和 smoke 实验；如果要写成论文式实验，建议补充 `50/75/100` 步上限的对照。

## 3. 算法实现梳理

### 普通 DQN

普通 DQN 使用共享参数的追捕者 Q 网络。每个追捕者根据自己的 16 维观测输出 5 个动作的 Q 值，训练中使用：

- epsilon-greedy 探索
- replay buffer 经验回放
- target network 稳定 TD target
- Huber loss / smooth L1 loss
- gradient clipping

训练时逃跑者不学习，每一步随机行动。

### 改进 DQN

改进 DQN 在普通 DQN 上加入两类机制：

- 团队奖励混合：训练 transition 中使用 `raw_reward + team_weight * team_mean_reward`，当前权重为 `0.3`
- 课程学习：训练早期降低逃跑者速度，之后逐步恢复到完整难度

本次审计发现课程学习原实现没有真正改变环境难度：项目传入了 `curriculum_stage`，但运行时 MPE2 环境没有实际切换 stage。因此已做最小修复：在每局 `reset()` 后手动按 stage 缩放逃跑者速度和加速度。

当前课程阶段为：

| Stage | 逃跑者速度比例 | `max_speed` | `accel` |
|---|---:|---:|---:|
| 0 | 50% | 0.65 | 2.0 |
| 1 | 75% | 0.975 | 3.0 |
| 2 | 100% | 1.3 | 4.0 |

注意：已有 `runs/dqn_curriculum_team_reward/best.pt` 是修复前训练得到的 checkpoint，因此它不能完全代表“修复后真实课程学习”的训练效果。当前 100 局评估只能说明已有 checkpoint 的表现。

### MAPPO

MAPPO 是策略梯度类多智能体算法，项目中实现为：

- actor：每个追捕者根据本地观测输出动作分布
- critic：使用拼接后的全局观测估计状态价值
- GAE：计算 advantage 和 return
- PPO clip objective：限制策略更新幅度
- entropy bonus：鼓励探索

相比 DQN，MAPPO 更自然地适合多智能体协作，因为它训练时可以借助集中式 critic 评价整体局势，执行时仍然由每个追捕者分散决策。

## 4. 100 局复核评估

本次使用现有 checkpoint 做了 100 局评估，输出目录：

`runs/audit_eval_100_20260521_003147`

| Experiment | Episodes | Capture Rate | Mean Reward | Mean Steps To Capture | Success Count |
|---|---:|---:|---:|---:|---:|
| random_baseline | 100 | 0.10 | 3.0 | 13.50 | 10 |
| dqn_baseline | 100 | 0.33 | 9.9 | 18.39 | 33 |
| dqn_curriculum_team_reward | 100 | 0.33 | 10.5 | 15.03 | 33 |
| mappo_baseline | 100 | 0.75 | 22.8 | 19.20 | 75 |

解读：

- DQN 明显优于随机策略，说明追捕者确实学到了有效策略。
- 已有改进 DQN 与普通 DQN 的成功率相同，但成功时平均步数更低，说明其策略在成功 episode 中更快完成抓捕。
- MAPPO 在现有结果中成功率最高，符合其更适合多智能体协作任务的预期。
- 改进 DQN 的结果需要谨慎表述，因为历史 checkpoint 是课程学习修复前训练得到的。

## 5. 当前优点

- 项目结构清晰，训练、评估、绘图、渲染路径完整。
- DQN 实现包含 replay buffer、target network、epsilon 衰减等关键机制。
- MAPPO 实现包含 actor、critic、GAE、clip loss，具备基本 CTDE 思路。
- 团队奖励只用于训练 transition，评估仍使用原始环境奖励，避免评估指标作弊。
- 测试覆盖了核心模块、CLI smoke、MAPPO smoke、渲染和指标逻辑。

## 6. 当前局限

- 训练只使用单个训练 seed，统计稳定性不足。
- `max_cycles=50` 偏短，可能影响 DQN 与 MAPPO 的公平比较。
- 已有改进 DQN checkpoint 是课程学习修复前训练的，需要重新训练才能证明修复后的课程学习收益。
- DQN 是 Independent DQN，共享参数但没有显式通信或集中式价值分解。
- MAPPO 当前实现适合展示原理，但还不是高度调参后的强基线。

## 7. 最值得做的 3 个后续改进

1. 重新训练修复后的 `dqn_curriculum_team_reward`，至少跑完整 3000 episodes，再用 100 或 500 局评估。
2. 做多训练 seed 对照，例如 seed 0、1、2，各自训练普通 DQN、改进 DQN、MAPPO，再汇总均值和方差。
3. 做 episode length 消融实验，对比 `max_cycles=50/75/100`，判断当前 50 步是否限制了策略表现。

## 8. 本次修改与验证

修改文件：

- `src/pursuit_lab/envs.py`
- `src/pursuit_lab/train.py`
- `tests/test_env.py`

验证命令：

```powershell
.\.venv\Scripts\python -m pytest tests\test_env.py -q
.\.venv\Scripts\python -m pytest -q
```

验证结果：

- `tests/test_env.py`：5 passed
- 全量测试：18 passed

额外 smoke：

- random + DQN smoke：通过，写入临时目录
- curriculum DQN smoke：通过，写入临时目录
- MAPPO smoke：通过，写入临时目录

未执行事项：

- 未重新训练完整 3000 episode 的修复后改进 DQN。
- 未覆盖已有 `runs/`、`figures/`、`videos/` 中的历史结果。
