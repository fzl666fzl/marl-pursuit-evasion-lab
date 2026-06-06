# MARL 追逃项目审计报告

生成日期：2026-05-21
更新日期：2026-06-07

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

注意：这一组是早期历史结果，发生在 `b60ca94 Fix independent random action seeding` 之前。修复后随机动作空间不再锁步采样，因此下面结果只能作为历史参考，当前答辩和简历更建议引用第 5 节的 post-seedfix 完整重跑结果。

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

## 5. Post-seedfix 完整重跑

本轮在 action-space seeding 修复后重新跑了四组配置，未覆盖旧的 `runs/`、`figures/`、`videos/` 产物。

命令：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/dqn_baseline.yaml configs/dqn_curriculum_team_reward.yaml configs/mappo_baseline.yaml --output-dir runs\post_seedfix_full_20260606_164815 --figures figures\post_seedfix_full_20260606_164815
```

输出：

- `runs/post_seedfix_full_20260606_164815/`
- `figures/post_seedfix_full_20260606_164815/eval_comparison.csv`
- `figures/post_seedfix_full_20260606_164815/eval_comparison.png`
- `figures/post_seedfix_full_20260606_164815/training_curve.png`

运行时间：约从 2026-06-06 16:48 到 2026-06-07 02:30，接近 9 小时 42 分钟。

最终评估结果：

| Experiment | Eval Episodes | Capture Rate | Mean Reward | Mean Steps To Capture | Success Count |
|---|---:|---:|---:|---:|---:|
| random_baseline | 50 | 0.34 | 10.2 | 17.94 | 17 |
| dqn_baseline | 10 | 0.10 | 3.0 | 13.00 | 1 |
| dqn_curriculum_team_reward | 10 | 0.10 | 3.0 | 11.00 | 1 |
| mappo_baseline | 10 | 0.70 | 21.0 | 18.71 | 7 |

补充观察：

- 随机基线从历史 0.10 变为 0.34，原因是随机动作空间 seeding 修复后不再出现所有 agent 锁步采样的异常随机行为。
- 普通 DQN 和改进 DQN 的最终网络在 10 局评估中都只有 0.10，但训练中途并非一直差：普通 DQN 中途最高 eval capture rate 到 0.60，改进 DQN 在课程 stage 0 到过 0.70。
- 改进 DQN 的课程切换已在日志中验证：约第 1000 局后进入 stage 1，约第 2000 局后进入 stage 2。
- MAPPO 最终 10 局评估达到 0.70，是本轮完整重跑里最终表现最好的方法。
- 当前 `eval_summary.json` 记录的是最终网络的表现，而不是 `best.pt` 的表现；因此后续更严谨的做法是同时评估 final checkpoint 和 best checkpoint。

## 6. 当前优点

- 项目结构清晰，训练、评估、绘图、渲染路径完整。
- DQN 实现包含 replay buffer、target network、epsilon 衰减等关键机制。
- MAPPO 实现包含 actor、critic、GAE、clip loss，具备基本 CTDE 思路。
- 团队奖励只用于训练 transition，评估仍使用原始环境奖励，避免评估指标作弊。
- 测试覆盖了核心模块、CLI smoke、MAPPO smoke、渲染和指标逻辑。

## 7. 当前局限

- 训练只使用单个训练 seed，统计稳定性不足。
- `max_cycles=50` 偏短，可能影响 DQN 与 MAPPO 的公平比较。
- 当前 trainable 配置最终评估只有 10 局，方差较大；随机基线是 50 局，评估 episode 数不完全一致。
- 当前 `eval_summary.json` 评估最终网络，不评估 `best.pt`，容易低估中途表现更好的 DQN checkpoint。
- 修复后的改进 DQN 在课程早期有更高 eval 表现，但迁移到完整环境后最终回落，课程学习收益还需要更严格消融验证。
- DQN 是 Independent DQN，共享参数但没有显式通信或集中式价值分解。
- MAPPO 当前实现适合展示原理，但还不是高度调参后的强基线。

## 8. 最值得做的 3 个后续改进

1. 对 final checkpoint 和 `best.pt` 都做 100 或 500 局评估，避免只看最终网络导致误判。
2. 做多训练 seed 对照，例如 seed 0、1、2，各自训练普通 DQN、改进 DQN、MAPPO，再汇总均值和方差。
3. 做 episode length 消融实验，对比 `max_cycles=50/75/100`，判断当前 50 步是否限制了策略表现。

## 9. 本次修改与验证

修改文件：

- `src/pursuit_lab/envs.py`
- `src/pursuit_lab/mappo.py`
- `src/pursuit_lab/metrics.py`
- `src/pursuit_lab/train.py`
- `tests/test_env.py`
- `tests/test_rewards_and_metrics.py`
- `PROJECT_AUDIT_REPORT.md`
- `PROJECT_INTERVIEW_GUIDE.md`

验证命令：

```powershell
.\.venv\Scripts\python -m pytest tests\test_env.py -q
.\.venv\Scripts\python -m pytest tests\test_rewards_and_metrics.py::test_append_csv_row_writes_header_once_and_appends_rows -q
.\.venv\Scripts\python -m pytest tests\test_cli_smoke.py::test_train_evaluate_render_and_plot_smoke -q
.\.venv\Scripts\python -m pytest tests\test_mappo_cli.py -q
.\.venv\Scripts\python -m pytest -q
```

验证结果：

- `tests/test_env.py`：7 passed
- `test_append_csv_row_writes_header_once_and_appends_rows`：1 passed
- DQN / evaluate / render / plot smoke：1 passed
- MAPPO CLI smoke：1 passed
- 全量测试：21 passed in 211.11s

额外 smoke：

- random + DQN smoke：通过，写入临时目录
- curriculum DQN smoke：通过，写入临时目录
- MAPPO smoke：通过，写入临时目录

未执行事项：

- 未做多训练 seed 统计。
- 未做 100/500 局 post-seedfix best checkpoint 复评。
- 未覆盖已有 `runs/`、`figures/`、`videos/` 中的历史结果。
