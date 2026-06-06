# marl-pursuit-evasion-lab 面试复习文档

这份文档的目标不是替代代码，而是帮你在面试时能把项目讲清楚：项目解决什么问题，为什么这么设计，训练流程怎么跑，核心算法怎么实现，实验结果怎么看，以及面试官追问时应该怎么回答。

当前项目路径：

```text
D:\marl-pursuit-evasion-lab
```

## 1. 一句话介绍项目

本项目基于 MPE2 / PettingZoo 的 `simple_tag_v3` 追逃环境，构建了一个轻量级多智能体强化学习实验平台。实验中有 3 个追捕者和 1 个逃跑者，逃跑者采用随机策略，追捕者分别使用 Independent DQN、DQN 改进版和 MAPPO 学习追捕策略。项目比较随机基线、普通 DQN、课程学习 + 团队奖励 DQN、以及集中式 critic 的 MAPPO baseline，并用追捕成功率、平均追捕步数和平均奖励评估效果。

面试时可以这样说：

> 我做的是一个多智能体追逃强化学习实验。环境里有三个追捕者和一个随机逃跑者，追捕者先用共享参数的 Independent DQN 学习追捕，再升级到 MAPPO，用集中式 critic 学习团队状态价值。项目重点是比较普通 DQN、奖励和课程改进版 DQN、以及 MAPPO 在追捕成功率和抓捕效率上的差异。

## 2. 项目到底在模拟什么

环境来自 `mpe2.simple_tag_v3`。这是一个二维连续空间里的追逃任务，但动作空间被设置成离散动作。

项目默认设定在 `src/pursuit_lab/constants.py`：

```python
PURSUER_AGENTS = ("adversary_0", "adversary_1", "adversary_2")
PREY_AGENT = "agent_0"

DEFAULT_ENV_KWARGS = {
    "num_good": 1,
    "num_adversaries": 3,
    "num_obstacles": 2,
    "max_cycles": 50,
    "continuous_actions": False,
    "terminate_on_success": True,
}
```

含义如下：

| 项目 | 含义 |
|---|---|
| `num_adversaries: 3` | 三个追捕者 |
| `num_good: 1` | 一个逃跑者 |
| `num_obstacles: 2` | 两个障碍物 |
| `max_cycles: 50` | 每局最多 50 步 |
| `continuous_actions: false` | 使用离散动作，不使用连续控制 |
| `terminate_on_success: true` | 抓到后这一局直接结束 |

运行检查后，实际空间是：

```text
adversary_0 obs_shape=(16,) action_space=Discrete(5)
adversary_1 obs_shape=(16,) action_space=Discrete(5)
adversary_2 obs_shape=(16,) action_space=Discrete(5)
agent_0     obs_shape=(14,) action_space=Discrete(5)
```

所以对追捕者策略网络来说：

```text
输入：16 维 observation
输出：5 个离散动作的分数
```

## 3. 谁在学习，谁不学习

这个项目里只有追捕者学习，逃跑者不学习。

训练时，代码在 `src/pursuit_lab/train.py` 中这样选动作：

```python
if agent in PURSUER_AGENTS:
    actions[agent] = select_epsilon_greedy_action(...)
else:
    actions[agent] = int(action_space.sample())
```

也就是说：

| 智能体 | 策略 |
|---|---|
| `adversary_0` | 学习策略：DQN 或 MAPPO actor |
| `adversary_1` | 学习策略：DQN 或 MAPPO actor |
| `adversary_2` | 学习策略：DQN 或 MAPPO actor |
| `agent_0` | 随机策略 |

面试官如果问“逃跑者怎么逃”，回答：

> 逃跑者不是训练出来的智能策略，而是每一步从动作空间随机采样动作。如果 50 步内没有被追捕者抓到，就算逃跑成功。这个设置让实验重点集中在追捕者能否通过强化学习学会协作追捕，而不是双方同时博弈学习。

## 4. 初始位置怎么定

初始位置由 MPE2 环境在 `env.reset(seed=...)` 时随机生成。项目没有手写初始坐标。

训练中每个 episode 使用：

```python
observations, _ = env.reset(seed=seed + episode_index)
```

所以：

- 每一局初始位置不同。
- 追捕者、逃跑者、障碍物位置都由环境随机初始化。
- 因为传了 seed，所以随机过程可复现。

面试官如果问“是不是完全随机不可复现”，回答：

> 不是。它是随机初始化，但有随机种子控制。同样的 seed 和配置下可以复现实验。

## 5. 多智能体体现在哪里

项目属于多智能体强化学习，因为环境里同时存在多个智能体，每一步所有 agent 都要同时行动。

项目现在包含两类多智能体训练思路。

第一类是共享参数的 Independent DQN。三个追捕者共用一个 DQN，每个追捕者只根据自己的 observation 独立选动作：

可以理解成：

```text
三个追捕者各自看到自己的 observation
       ↓
都调用同一个 DQN 网络
       ↓
各自选一个离散动作
       ↓
环境同时执行这些动作
       ↓
把三个追捕者产生的经验都放进同一个 replay buffer
       ↓
用这些经验训练同一个 DQN 网络
```

第二类是 MAPPO。MAPPO 采用 centralized training decentralized execution：

```text
执行时：
每个追捕者只看自己的 16 维 observation
       ↓
共享 actor 输出 5 个动作 logits
       ↓
每个追捕者独立采样/选择动作

训练时：
critic 看到拼接后的全局 observation
       ↓
估计团队状态价值
       ↓
用 PPO clipped objective + GAE 更新 actor 和 critic
```

为什么 DQN 要共享参数：

- 三个追捕者角色相同，学习目标相同。
- 共享参数能减少模型数量，训练更轻量。
- 三个追捕者的数据都能用于更新同一个策略，样本利用率更高。

DQN 的局限：

- 每个追捕者独立根据自己的 observation 决策，没有显式通信。
- 没有 centralized critic，不能像 MAPPO 那样直接利用全局联合状态学习队友配合。
- 其他智能体的策略变化会让环境对单个智能体看起来更不稳定，这是 Independent RL 的常见问题。

面试时可以说：

> 项目最初用 Independent DQN 是为了建立轻量、可复现的基线。后来我发现 DQN 在多智能体协作上不够稳定：历史复评里 DQN 明显弱于 MAPPO，post-seedfix 重跑里 DQN 最终网络也没有稳定超过随机基线。所以我加入了 MAPPO。MAPPO 的 actor 执行时仍然只用本地 observation，但训练时 critic 可以看到全局 observation，更适合多智能体协作追捕。

## 6. DQN 原理

DQN 的核心思想是学习一个函数：

```text
Q(s, a) = 在状态 s 下执行动作 a，未来能获得的累计回报估计
```

对于这个项目：

```text
s：追捕者当前 observation，16 维向量
a：离散动作，共 5 个
Q(s, a)：这个动作对追捕任务有多好
```

网络结构在 `src/pursuit_lab/dqn.py`：

```python
class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(128, 128)):
        ...
        layers.append(nn.Linear(last_dim, action_dim))
```

默认配置：

```yaml
dqn:
  hidden_sizes: [128, 128]
  learning_rate: 0.001
  gamma: 0.95
```

也就是一个 MLP：

```text
16 维 observation
  → Linear(16, 128)
  → ReLU
  → Linear(128, 128)
  → ReLU
  → Linear(128, 5)
  → 输出 5 个动作的 Q 值
```

动作选择使用 epsilon-greedy：

```python
if rng.random() < epsilon:
    return random_action
else:
    return argmax_q_action
```

含义：

- 训练初期 `epsilon` 高，多随机探索。
- 训练后期 `epsilon` 低，更多选择 DQN 认为最好的动作。
- 默认从 `1.0` 线性下降到 `0.05`，衰减 2500 个 episode。

公式上，DQN 希望让当前 Q 值接近目标值：

```text
target = reward + gamma * max_a' Q_target(next_state, a')
```

如果 episode 已结束，则没有未来项：

```text
target = reward
```

代码在 `src/pursuit_lab/dqn.py`：

```python
q_values = policy_net(observations).gather(1, actions).squeeze(1)

with torch.no_grad():
    next_q_values = target_net(next_observations).max(dim=1).values
    targets = rewards + gamma * next_q_values * (1.0 - dones)

loss = nn.functional.smooth_l1_loss(q_values, targets)
```

这里用了：

- `policy_net`：当前正在训练的网络。
- `target_net`：目标网络，用于计算较稳定的 target。
- `smooth_l1_loss`：Huber loss，比普通 MSE 对异常值更稳。
- `clip_grad_norm_`：梯度裁剪，降低训练爆炸风险。

## 6.1 MAPPO 原理

MAPPO 是 Multi-Agent PPO。它比 Independent DQN 更适合这个项目的核心原因是：它允许训练时使用全局信息，但执行时仍然保持每个追捕者独立行动。

本项目的 MAPPO 实现在 `src/pursuit_lab/mappo.py`。

### Actor

actor 是共享策略网络：

```text
输入：单个追捕者自己的 16 维 observation
输出：5 个离散动作的 logits
```

也就是：

```text
16 维 observation
  → Linear(16, 128)
  → Tanh
  → Linear(128, 128)
  → Tanh
  → Linear(128, 5)
  → Categorical distribution
```

执行时每个追捕者都调用同一个 actor，但只输入自己的 observation。这符合 decentralized execution。

### Critic

critic 是集中式价值网络：

```text
输入：全局 observation
输出：当前全局状态的 value
```

全局 observation 是固定顺序拼接：

```text
adversary_0 obs 16维
+ adversary_1 obs 16维
+ adversary_2 obs 16维
+ agent_0 obs 14维
= 62维
```

所以 MAPPO critic 输入维度是：

```text
3 * 16 + 14 = 62
```

这就是 centralized training：训练时 critic 能看到全局状态，能更好地判断“这个团队局面好不好”。

### PPO clipped objective

PPO 的核心是限制策略更新幅度，避免一次更新把策略推得太远。它比较新旧策略概率比：

```text
ratio = pi_new(a|s) / pi_old(a|s)
```

然后用 clip 限制 ratio：

```text
clip(ratio, 1 - clip_coef, 1 + clip_coef)
```

本项目默认：

```yaml
clip_coef: 0.2
```

### GAE

MAPPO 使用 GAE，也就是 Generalized Advantage Estimation，用来估计 advantage：

```text
advantage = 实际回报相对 critic 估计 value 的优势
```

默认参数：

```yaml
gamma: 0.95
gae_lambda: 0.95
```

`gamma` 控制未来奖励折扣，`gae_lambda` 控制 advantage 估计在偏差和方差之间的折中。

### 为什么 MAPPO 更适合这个任务

Independent DQN 中，每个追捕者只从自己的经验学习，队友行为会让环境显得不稳定。MAPPO 的 critic 能看到全局 observation，因此更容易学习到团队追捕局面的价值。actor 执行时仍然只用本地 observation，所以部署方式没有变复杂。

面试时可以这样说：

> DQN 是 independent learning，协作主要靠共享参数和奖励设计；MAPPO 则引入 centralized critic，让训练时可以利用全局状态学习团队价值。这个改动直接对应多智能体协作追捕的难点。post-seedfix 重跑里，MAPPO 最终 10 局捕获率达到 70%，明显高于两个 DQN 最终网络。

## 7. Replay Buffer 的作用

Replay Buffer 在 `src/pursuit_lab/replay.py`。

强化学习中，如果直接用刚发生的连续经验训练，样本高度相关，训练容易不稳定。Replay Buffer 的做法是：

1. 把每一步经验存起来。
2. 训练时随机抽一批旧经验。
3. 打破样本之间的时间相关性。
4. 提高样本复用率。

每条经验包括：

```text
observation
action
reward
next_observation
done
```

默认配置：

```yaml
training:
  buffer_capacity: 50000
  min_buffer_size: 1000
  batch_size: 64
```

含义：

- 最多存 50000 条 transition。
- 缓冲区少于 1000 条时不训练，先积累经验。
- 每次更新随机采样 64 条。

面试官如果问“为什么不一边走一边马上训练”，回答：

> 一方面连续样本相关性强，直接训练容易不稳定；另一方面 replay buffer 可以复用过去经验，提高样本效率。DQN 的经典做法就是经验回放加目标网络。

## 8. 训练流程

训练主入口是：

```powershell
.\.venv\Scripts\python -m pursuit_lab.train --config configs/dqn_baseline.yaml
```

多组实验入口是：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments
```

DQN 训练主循环在 `src/pursuit_lab/train.py` 的 `train_dqn(config)`。

整体流程：

```text
读取 YAML 配置
设置随机种子
创建输出目录 runs/<experiment_name>
创建环境，读取 obs_dim 和 action_dim
初始化 policy_net 和 target_net
初始化 Adam optimizer
初始化 ReplayBuffer

for episode in total_episodes:
    根据是否开启课程学习设置 curriculum_stage
    创建并 reset 环境
    计算当前 epsilon

    for step in max_cycles:
        追捕者用 epsilon-greedy DQN 选动作
        逃跑者随机选动作
        env.step(actions)
        计算训练用 reward
        写入 replay buffer
        如果 buffer 足够大：
            采样 batch
            优化 DQN
            定期同步 target_net
        如果抓到或达到终止条件：
            break

    记录 metrics.csv
    到 eval_interval 时做评估
    如果评估更好，保存 best.pt

最终再评估一次，写 eval_summary.json 和 eval_episodes.csv
```

关键训练参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `total_episodes` | 3000 | 每个 DQN 实验训练 3000 局 |
| `max_cycles` | 50 | 每局最多 50 步 |
| `learning_rate` | 0.001 | Adam 学习率 |
| `gamma` | 0.95 | 折扣因子 |
| `batch_size` | 64 | 每次训练采样 64 条 transition |
| `target_update_interval` | 500 | 每 500 个全局 step 同步目标网络 |
| `epsilon_start` | 1.0 | 初始完全探索 |
| `epsilon_end` | 0.05 | 最终保留 5% 随机探索 |
| `epsilon_decay_episodes` | 2500 | 2500 局内线性下降 |

MAPPO 训练入口仍然走同一个命令行模块：

```powershell
.\.venv\Scripts\python -m pursuit_lab.train --config configs/mappo_baseline.yaml
```

MAPPO 的核心训练流程在 `src/pursuit_lab/mappo.py`：

```text
读取 MAPPO 配置
创建共享 actor 和集中式 critic

for episode in total_episodes:
    reset 环境
    for step in max_cycles:
        拼接全局 observation 给 critic
        每个追捕者用共享 actor 采样动作
        逃跑者随机采样动作
        env.step(actions)
        记录 observation、global_observation、action、log_prob、value、reward、done

    每 rollout_episodes 局：
        用 GAE 计算 advantages 和 returns
        用 PPO clipped objective 更新 actor
        用 value loss 更新 critic

    定期评估
    如果评估分数更好，保存 best.pt
```

MAPPO 默认参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `total_episodes` | 3000 | 训练 3000 局 |
| `actor_hidden_sizes` | [128, 128] | actor MLP |
| `critic_hidden_sizes` | [128, 128] | critic MLP |
| `learning_rate` | 0.001 | Adam 学习率 |
| `gamma` | 0.95 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 参数 |
| `clip_coef` | 0.2 | PPO 裁剪范围 |
| `entropy_coef` | 0.01 | 探索熵奖励 |
| `value_coef` | 0.5 | value loss 权重 |
| `update_epochs` | 4 | 每批 rollout 更新轮数 |
| `rollout_episodes` | 10 | 每 10 局更新一次 |

## 9. 四组实验分别是什么

默认 `run_experiments` 会跑四组：

```python
DEFAULT_CONFIGS = (
    "configs/random_baseline.yaml",
    "configs/dqn_baseline.yaml",
    "configs/dqn_curriculum_team_reward.yaml",
    "configs/mappo_baseline.yaml",
)
```

### 9.1 random_baseline

配置文件：`configs/random_baseline.yaml`

所有智能体随机行动，不训练模型。

作用：

- 给出最低基线。
- 判断 DQN 是否真的学到东西。

面试时可以说：

> 随机基线是 sanity check。如果训练后的 DQN 长期比随机还差，就说明训练策略、奖励设计或评估方式可能有问题。

### 9.2 dqn_baseline

配置文件：`configs/dqn_baseline.yaml`

普通 DQN：

```yaml
team_reward_weight: 0.0
curriculum: false
```

含义：

- 不用课程学习。
- 不额外混合团队奖励。
- 追捕者只使用环境原始奖励训练。

### 9.3 dqn_curriculum_team_reward

配置文件：`configs/dqn_curriculum_team_reward.yaml`

改进版 DQN：

```yaml
team_reward_weight: 0.3
curriculum: true
```

多了两点：

1. 课程学习。
2. 团队奖励混合。

目标是让训练更容易、更稳定，并鼓励追捕者团队协作。

### 9.4 mappo_baseline

配置文件：`configs/mappo_baseline.yaml`

纯 MAPPO baseline：

```yaml
algorithm: mappo
total_episodes: 3000
mappo:
  gamma: 0.95
  gae_lambda: 0.95
  clip_coef: 0.2
  entropy_coef: 0.01
  update_epochs: 4
  rollout_episodes: 10
```

它不叠加课程学习，也不使用团队奖励混合。训练 reward 使用追捕者原始个体奖励，这样和 DQN baseline 的对比更公平。

MAPPO 的关键差异：

- DQN 是 value-based，学习 Q 值。
- MAPPO 是 policy-gradient，直接学习 actor 策略。
- DQN 的 critic/target 只看单个追捕者 transition。
- MAPPO 的 critic 训练时看全局 observation，更适合团队协作。

## 10. 课程学习是什么

课程学习的思想是：

```text
先学简单任务，再逐步提高难度
```

项目中课程阶段由 `src/pursuit_lab/envs.py` 控制：

```python
def curriculum_stage_for_episode(episode_index, total_episodes):
    first_cut = total_episodes // 3
    second_cut = (2 * total_episodes) // 3
    if episode_index < first_cut:
        return 0
    if episode_index < second_cut:
        return 1
    return 2
```

3000 episodes 时：

| episode 范围 | 阶段 |
|---|---:|
| 0 - 999 | stage 0 |
| 1000 - 1999 | stage 1 |
| 2000 - 2999 | stage 2 |

训练时会通过 `make_env(..., curriculum_stage=stage)` 和 `apply_curriculum_stage` 应用难度：

```python
CURRICULUM_SPEED_FACTORS = (0.5, 0.75, 1.0)
```

也就是说 stage 0 / 1 / 2 会把逃跑者的最大速度和加速度分别缩放到 50% / 75% / 100%。这样课程学习的难度变化由项目代码显式控制。

面试官如果问“为什么要课程学习”，回答：

> 因为一开始就让智能体面对完整难度，DQN 可能很难获得有效奖励，探索效率低。课程学习让它先在相对简单的阶段学到基本追捕行为，再过渡到更难阶段，有助于提高训练稳定性。

## 11. 团队奖励混合是什么

普通训练时，每个追捕者只看自己的环境奖励。改进版中，训练用奖励变成：

```text
mixed_reward_i = raw_reward_i + team_weight * mean(raw_reward_all_pursuers)
```

代码在 `src/pursuit_lab/rewards.py`：

```python
pursuer_rewards = [raw_rewards[agent] for agent in PURSUER_AGENTS]
team_mean = sum(pursuer_rewards) / len(pursuer_rewards)

return {
    agent: raw_rewards[agent] + team_weight * team_mean
    for agent in PURSUER_AGENTS
}
```

默认改进版：

```yaml
team_reward_weight: 0.3
```

例子：

```text
三个追捕者原始奖励：1, 2, 3
团队平均奖励：2
team_weight = 0.3

新奖励：
adversary_0 = 1 + 0.3 * 2 = 1.6
adversary_1 = 2 + 0.3 * 2 = 2.6
adversary_2 = 3 + 0.3 * 2 = 3.6
```

这个例子也被测试覆盖在 `tests/test_rewards_and_metrics.py`。

重要细节：

> 团队奖励混合只用于训练 replay transition，评估仍然使用环境原始奖励。

为什么这样设计：

- 训练时希望鼓励团队表现。
- 评估时要公平比较不同方法，不能把人为混合后的奖励拿来当最终指标。

面试时可以说：

> 我没有直接改评估指标，而是只改训练信号。这样能用团队奖励引导学习，同时评估仍基于原始环境返回，避免指标被奖励 shaping 污染。

## 12. 评估逻辑

评估代码在 `src/pursuit_lab/evaluation.py`。

评估时：

```python
epsilon=0.0
```

也就是说追捕者不再探索，而是完全选择 Q 值最大的动作。逃跑者仍然随机行动。

记录的每个 episode 信息：

```text
episode
seed
captured
steps
episode_reward
```

汇总指标在 `src/pursuit_lab/metrics.py`：

| 指标 | 含义 | 越大/越小 |
|---|---|---|
| `capture_rate` | 成功抓到的比例 | 越大越好 |
| `mean_episode_reward` | 每局追捕者原始总奖励均值 | 通常越大越好 |
| `mean_steps_to_capture` | 成功抓捕时平均用了多少步 | 越小越好 |
| `success_episode_count` | 成功抓捕局数 | 越大越好 |

如果一组评估没有任何成功抓捕：

```python
mean_steps_to_capture = max_cycles
```

默认 DQN 训练中每 250 个 episode 评估一次：

```yaml
evaluation:
  eval_interval: 250
  eval_episodes: 10
  seeds: [0, 1, 2, 3, 4]
```

保存最佳模型时的评分：

```python
score = capture_rate * 1000.0 - mean_steps_to_capture
```

这表示优先选择追捕成功率更高的模型；如果成功率相同，再偏向更快抓到的模型。

## 13. 当前 3000-episode 实验结果

你已经跑完了两轮需要区分的结果：

1. 历史 100 局复评：发生在随机动作空间 seeding 修复前，只适合作为历史参考。
2. post-seedfix 完整重跑：发生在 `b60ca94 Fix independent random action seeding` 之后，是当前更推荐引用的结果。

### 13.1 当前可引用的 post-seedfix 结果

本轮命令：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/dqn_baseline.yaml configs/dqn_curriculum_team_reward.yaml configs/mappo_baseline.yaml --output-dir runs\post_seedfix_full_20260606_164815 --figures figures\post_seedfix_full_20260606_164815
```

输出目录：

```text
runs/post_seedfix_full_20260606_164815/
figures/post_seedfix_full_20260606_164815/
```

运行时间约从 2026-06-06 16:48 到 2026-06-07 02:30，接近 9 小时 42 分钟。

最终评估结果：

| experiment | eval episodes | capture_rate | mean_episode_reward | mean_steps_to_capture | success_episode_count |
|---|---:|---:|---:|---:|---:|
| `random_baseline` | 50 | 0.34 | 10.2 | 17.94 | 17 |
| `dqn_baseline` | 10 | 0.10 | 3.0 | 13.00 | 1 |
| `dqn_curriculum_team_reward` | 10 | 0.10 | 3.0 | 11.00 | 1 |
| `mappo_baseline` | 10 | 0.70 | 21.0 | 18.71 | 7 |

怎么解读：

- action-space seeding 修复后，随机基线不再锁步随机，50 局捕获率为 0.34；所以旧的随机 0.10 不应再当成当前结论。
- DQN 和改进 DQN 的最终网络在 10 局评估中都是 0.10，说明最终策略没有稳定超过随机基线；但训练中途评估曾明显更高，普通 DQN 最高到 0.60，改进 DQN 在课程早期最高到 0.70。
- 改进 DQN 的课程切换确实发生了：约第 1000 局进入 stage 1，约第 2000 局进入 stage 2；但迁移到完整难度后最终表现回落。
- MAPPO 最终 10 局 capture_rate 为 0.70，明显高于两个 DQN 的最终网络，是当前 post-seedfix 重跑中表现最好的方法。
- 当前 trainable 配置每次最终评估只有 10 局，方差较大；后续应对 `best.pt` 和 final checkpoint 都做 100/500 局复评。

面试时推荐结论：

> 修复随机动作 seeding 后，我重新跑了四组完整实验。随机基线 50 局捕获率是 34%；两个 DQN 最终网络在 10 局评估里都是 10%，说明最终策略不够稳定，但训练中途曾达到更高成功率。MAPPO 最终 10 局达到 70%，是当前最强结果。这说明 centralized critic 对多智能体协作更有帮助，同时也暴露出 DQN 需要 best checkpoint 复评和更多评估局数。

### 13.2 历史 100 局复评结果

下面这组是 seeding 修复前的历史 checkpoint 复评，不能和 post-seedfix 结果直接混用。

你当时已经跑完了 DQN、DQN 改进版和 MAPPO 的 3000 episode 长训练：

已确认：

- `runs/dqn_baseline/metrics.csv` 有 3001 行：表头 + 3000 episodes。
- `runs/dqn_curriculum_team_reward/metrics.csv` 有 3001 行：表头 + 3000 episodes。
- `runs/mappo_baseline/metrics.csv` 有 3001 行：表头 + 3000 episodes。
- `runs/mappo_retrain_20260521_103249/metrics.csv` 也有 3001 行，用于重复验证 MAPPO 稳定性。
- 100 局复评图表已生成到 `figures/eval_mappo_retrain/`。

最终 100 局复评结果：

| experiment | eval episodes | capture_rate | mean_episode_reward | mean_steps_to_capture | success_episode_count |
|---|---:|---:|---:|---:|---:|
| `random_baseline` | 100 | 0.10 | 3.0 | 13.50 | 10 |
| `dqn_baseline` | 100 | 0.33 | 9.9 | 18.39 | 33 |
| `dqn_curriculum_team_reward` | 100 | 0.33 | 10.5 | 15.03 | 33 |
| `mappo_baseline_seed0` | 100 | 0.75 | 22.8 | 19.20 | 75 |
| `mappo_retrain_seed0_repeat` | 100 | 0.75 | 22.8 | 19.20 | 75 |

对应文件：

```text
runs/eval_mappo_retrain/mappo_retrain_20260521_103249/eval_summary.json
runs/mappo_baseline/best.pt
runs/mappo_retrain_20260521_103249/best.pt
figures/eval_mappo_retrain/eval_comparison.png
figures/eval_mappo_retrain/eval_comparison.csv
```

怎么解读：

- 随机基线只有 10% 成功率，说明任务本身不是随便乱走就能稳定抓到。
- DQN baseline 和 DQN 改进版都是 33%，说明 DQN 学到了一定追捕能力，但成功率仍偏低。
- DQN 改进版没有提高成功率，但成功时平均步数更低：18.39 降到 15.03，说明它主要提升了抓捕效率。
- MAPPO 成功率达到 75%，明显超过 DQN 的 33%，说明 centralized critic 对多智能体协作追捕非常有帮助。
- MAPPO 平均抓捕步数 19.20 比 DQN 改进版高，不代表 MAPPO 更差，因为 MAPPO 成功局更多，包含了更多“难抓但最终抓到”的 episode。

面试时推荐结论：

> 早期历史复评中，Independent DQN 能把成功率从随机基线的 10% 提到 33%，MAPPO 达到 75%。但这组结果发生在随机动作 seeding 修复前，所以现在只作为历史参考；当前更应该引用 post-seedfix 完整重跑结果。

## 14. 可视化和图表

### 14.1 训练曲线

生成命令：

```powershell
.\.venv\Scripts\python -m pursuit_lab.plot --runs runs/
```

输出：

```text
figures/training_curve.png
figures/eval_comparison.png
figures/eval_comparison.csv
```

MAPPO 四组对比图现在推荐看：

```text
figures/eval_mappo_retrain/eval_comparison.png
figures/eval_mappo_retrain/eval_comparison.csv
```

训练曲线逻辑在 `src/pursuit_lab/plot.py`：

```python
rewards = metrics["episode_reward"].rolling(window=25, min_periods=1).mean()
```

也就是对 `episode_reward` 做 25 局 rolling average，让曲线更平滑。

### 14.2 GIF 追逃动画

生成命令：

```powershell
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 3
```

如果要展示 MAPPO 策略：

```powershell
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/mappo_baseline/best.pt --episodes 3 --output videos/mappo_demo.gif
```

默认输出：

```text
videos/pursuit_demo.gif
```

渲染链路：

```text
render.py
  → load_checkpoint
  → build_policy_from_checkpoint
  → make_env
  → 每步选动作
  → rendering.render_world_frame
  → imageio.mimsave
```

`src/pursuit_lab/rendering.py` 会手动画出：

- 红色：追捕者。
- 绿色：逃跑者。
- 灰色：障碍物。
- 顶部 legend。

## 15. 输出目录含义

一次训练结束后，每个实验目录通常包含：

```text
runs/<experiment>/
  config.yaml
  metrics.csv
  best.pt
  eval_episodes.csv
  eval_summary.json
```

含义：

| 文件 | 作用 |
|---|---|
| `config.yaml` | 本次实验实际使用的配置 |
| `metrics.csv` | 每个训练 episode 的记录 |
| `best.pt` | 评估表现最好的模型 checkpoint |
| `eval_episodes.csv` | 最终评估每局详情 |
| `eval_summary.json` | 最终评估汇总指标 |

`best.pt` 中保存：

DQN checkpoint：

```python
{
    "model_state_dict": policy_net.state_dict(),
    "obs_dim": obs_dim,
    "action_dim": action_dim,
    "config": config,
    "episode": episode,
    "metrics": metrics,
    "epsilon": epsilon,
}
```

MAPPO checkpoint：

```python
{
    "algorithm": "mappo",
    "actor_state_dict": actor.state_dict(),
    "critic_state_dict": critic.state_dict(),
    "obs_dim": obs_dim,
    "prey_obs_dim": prey_obs_dim,
    "global_obs_dim": global_obs_dim,
    "action_dim": action_dim,
    "config": config,
    "episode": episode,
    "metrics": metrics,
}
```

所以评估和渲染时可以根据 checkpoint 自动恢复 DQN 或 MAPPO actor。

## 16. 常用命令

进入项目目录：

```powershell
cd D:\marl-pursuit-evasion-lab
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

不激活也可以直接用：

```powershell
.\.venv\Scripts\python
```

运行测试：

```powershell
.\.venv\Scripts\python -m pytest -q
```

快速 smoke 实验：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments --configs configs/random_baseline.yaml configs/smoke_test.yaml --episodes 10
```

正式默认实验：

```powershell
.\.venv\Scripts\python -m pursuit_lab.run_experiments
```

只训练普通 DQN：

```powershell
.\.venv\Scripts\python -m pursuit_lab.train --config configs/dqn_baseline.yaml
```

只训练改进版 DQN：

```powershell
.\.venv\Scripts\python -m pursuit_lab.train --config configs/dqn_curriculum_team_reward.yaml
```

只训练 MAPPO：

```powershell
.\.venv\Scripts\python -m pursuit_lab.train --config configs/mappo_baseline.yaml
```

评估 checkpoint：

```powershell
.\.venv\Scripts\python -m pursuit_lab.evaluate --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 50
```

评估 MAPPO checkpoint：

```powershell
.\.venv\Scripts\python -m pursuit_lab.evaluate --checkpoint runs/mappo_baseline/best.pt --episodes 100
```

生成 GIF：

```powershell
.\.venv\Scripts\python -m pursuit_lab.render --checkpoint runs/dqn_curriculum_team_reward/best.pt --episodes 3
```

重新生成图表：

```powershell
.\.venv\Scripts\python -m pursuit_lab.plot --runs runs/
```

## 17. 代码结构

```text
src/pursuit_lab/
  constants.py        常量：agent 名称、默认环境参数
  envs.py             创建 MPE2 环境、课程阶段、随机动作
  dqn.py              DQN 网络、epsilon-greedy、DQN 优化
  mappo.py            MAPPO actor/critic、GAE、PPO 更新、MAPPO 训练
  replay.py           经验回放缓冲区
  rewards.py          团队奖励混合
  train.py            训练主流程、随机基线、checkpoint 保存
  evaluation.py       checkpoint 加载、评估 episode、指标汇总调用
  evaluate.py         命令行评估入口
  plot.py             训练曲线和评估对比图
  render.py           GIF 生成入口
  rendering.py        自定义环境帧绘制
  run_experiments.py  一键运行多组实验
  config.py           默认配置、YAML 加载与深度合并
  metrics.py          指标汇总、CSV/JSON 写出
  utils.py            创建目录、设置随机种子
```

测试目录：

```text
tests/
  test_env.py                  环境 agent、随机策略、课程阶段测试
  test_dqn.py                  DQN 输出维度、动作选择测试
  test_mappo.py                MAPPO actor/critic/GAE 测试
  test_mappo_cli.py            MAPPO train/evaluate/render smoke 测试
  test_replay.py               replay buffer 测试
  test_rewards_and_metrics.py  团队奖励和指标测试
  test_cli_smoke.py            train/evaluate/render/plot 命令链路测试
  test_run_experiments.py      多配置实验入口测试
  test_rendering.py            自定义渲染帧测试
```

## 18. 面试官可能追问的问题

### Q1：这个项目为什么算多智能体强化学习？

因为环境中同时有多个 agent：三个追捕者和一个逃跑者。每一步环境接收所有 agent 的动作，并返回每个 agent 的 observation、reward、done 信息。虽然逃跑者不学习，但追捕者之间的行为会互相影响，因此训练仍处在多智能体环境中。

### Q2：你用的是哪种 MARL 算法？

项目包含两类算法。第一类是共享参数的 Independent DQN，每个追捕者独立根据自己的 observation 选动作，但共用同一个 DQN 网络和 replay buffer。第二类是 MAPPO，使用共享 actor 和集中式 critic，训练时 critic 使用全局 observation，执行时 actor 只使用本地 observation。

### Q3：为什么后来要加入 MAPPO？

最初 DQN 是为了做轻量基线，因为动作空间是 `Discrete(5)`，DQN 很适合快速验证。后来实验发现 Independent DQN 的最终策略不够稳定，尤其在 post-seedfix 重跑中两个 DQN 最终网络都只有 10% 捕获率。MAPPO 引入 centralized critic，可以在训练时利用全局 observation 学习团队价值，更符合多智能体协作追捕任务。

### Q4：逃跑者是智能的吗？

不是。逃跑者每一步随机采样动作，不训练模型。项目重点是训练追捕者学会抓随机逃跑目标。

### Q5：初始位置随机吗？

随机，但由 seed 控制。训练中每局使用 `seed + episode_index`，所以开局不同，但实验可复现。

### Q5.1：随机动作会不会所有智能体同步？

不会。项目给每个 agent 的 action space 使用由全局 seed 派生出来的不同 seed，所以随机基线和随机逃跑者都是可复现的，但不会出现所有 agent 每一步都采样到同一个动作的锁步问题。

### Q6：DQN 的输入输出是什么？

对追捕者来说，输入是 16 维 observation，输出是 5 个离散动作对应的 Q 值。选动作时取 Q 值最大的动作，训练时用 epsilon-greedy 保留探索。

### Q6.1：MAPPO 的 actor 和 critic 输入输出是什么？

MAPPO actor 输入单个追捕者自己的 16 维 observation，输出 5 个离散动作 logits。MAPPO critic 输入全局 observation，也就是三个追捕者 observation 和逃跑者 observation 拼接后的 62 维向量，输出一个 state value。

### Q7：为什么需要 target network？

如果用同一个网络同时估计当前 Q 和目标 Q，目标会随着训练不断变化，容易不稳定。target network 延迟同步，可以让 TD target 更稳定。

### Q8：为什么需要 replay buffer？

它能打乱样本相关性，并复用过去经验，提高训练稳定性和样本效率。

### Q9：课程学习具体怎么做？

把 3000 个 episode 分成三段。前 1000 局 stage 0，中间 1000 局 stage 1，最后 1000 局 stage 2。项目代码按 stage 显式缩放逃跑者速度和加速度：50%、75%、100%。

### Q10：团队奖励混合是否改变了最终评估指标？

没有。团队奖励混合只改变训练时存入 replay buffer 的 reward。最终评估仍然使用环境原始 reward 和 capture rate 等指标。

### Q11：怎么判断训练效果好不好？

主要看 `capture_rate` 和 `mean_steps_to_capture`。成功率越高越好；成功率相近时，平均抓捕步数越低越好。训练曲线只辅助观察，最终以评估结果为准。

### Q12：为什么 DQN baseline 3000 局后评估可能还不如随机？

可能原因包括：

- 早期 10 局评估样本量小，随机性较强。
- Independent DQN 在多智能体环境中本来就可能不稳定。
- 逃跑者和初始位置随机，评估 seed 对结果有影响。
- 奖励稀疏或探索效率不足时，DQN 可能学得不稳定。
- 普通 DQN 没有团队奖励和课程学习，可能更难形成协作。

好的回答方式不是硬说它一定更好，而是说：

> post-seedfix 重跑里，DQN 最终网络 10 局评估确实不如随机基线稳定，但训练中途 eval 最高到过 60%。所以我不会硬说最终 DQN 一定更好，而是把它作为轻量基线，并说明后续需要评估 best checkpoint、扩大评估局数和做多 seed 验证。

### Q13：为什么改进版更好？

从 post-seedfix 重跑看，不能说改进版最终一定更好，因为普通 DQN 和改进 DQN 最终网络 10 局捕获率都是 10%。它的价值主要在机制和中途表现：课程阶段切换已经验证，改进 DQN 在课程早期 eval 最高到过 70%，但迁移到完整难度后回落。

但要谨慎：

> 我不会把 DQN 改进版包装成全面成功。它证明了课程学习和团队奖励机制可以接入训练流程，但最终收益还需要 best checkpoint 复评、多 seed 和消融实验来确认。当前真正显著提升最终成功率的是 MAPPO。

### Q13.1：为什么 MAPPO 明显更好？

post-seedfix 完整重跑中，MAPPO 最终 10 局捕获率是 70%，而普通 DQN 和改进 DQN 最终网络都是 10%。原因是 MAPPO 训练时 critic 能看到全局 observation，能更好地评估团队追捕局面；DQN 每个追捕者主要从自己的 transition 学习，协作信息弱一些。

可以这样回答：

> MAPPO 的优势来自 centralized training decentralized execution。执行时每个追捕者仍然只看本地 observation，但训练时 critic 可以看全局状态，这对多智能体协作追捕很关键。因此在 post-seedfix 重跑中，MAPPO 最终捕获率明显高于两个 DQN 最终网络。

### Q14：项目有哪些不足？

可以主动说：

- 逃跑者是随机策略，不是对抗训练出来的智能逃跑者。
- 当前 trainable 配置最终评估只有 10 局，方差较大，应补 100/500 局复评。
- 当前 `eval_summary.json` 评估 final network，不评估 `best.pt`，可能低估中途更好的 checkpoint。
- MAPPO 结果还需要更多不同 seed 训练。
- Independent DQN 没有显式通信或 centralized critic，所以表现弱于 MAPPO。
- 没有系统调参，比如学习率、网络规模、reward weight。
- 当前还没有和 MAPPO 之外的强 MARL 方法比较，例如 MADDPG、QMIX、IPPO。

### Q15：如果继续改进，你会怎么做？

可以回答：

1. 增加多随机种子训练，报告均值和标准差。
2. 扩大评估 episodes，比如 200 或 500。
3. 增加更强基线，如 IPPO、QMIX、MADDPG。
4. 训练逃跑者，做真正的对抗学习。
5. 在 MAPPO 上加入课程学习或团队奖励，做消融实验。
6. 系统调参：学习率、gamma、buffer size、team reward weight。
7. 记录更多诊断指标，比如平均距离、碰撞次数、每个追捕者贡献。

## 19. 面试时的 30 秒版本

> 这个项目是一个基于 MPE2 simple_tag 的多智能体追逃实验。环境里有 3 个追捕者、1 个随机逃跑者和 2 个障碍物，追捕者分别用 Independent DQN、加入课程学习和团队奖励的 DQN，以及 MAPPO 学习追捕。DQN 使用 replay buffer、target network 和 epsilon-greedy，MAPPO 使用共享 actor 和集中式 critic，训练时 critic 输入 62 维全局 observation。修复随机动作 seeding 后完整重跑，MAPPO 最终 10 局捕获率达到 70%，明显高于两个 DQN 最终网络的 10%，说明 centralized critic 更适合这个协作追捕任务。

## 20. 面试时的 2 分钟版本

> 我这个项目做的是一个多智能体追逃强化学习实验，环境用的是 MPE2 / PettingZoo 的 simple_tag_v3。里面有 3 个追捕者、1 个逃跑者和 2 个障碍物。逃跑者不训练，每一步随机运动；追捕者先使用共享参数的 Independent DQN 学习追捕，后来又新增了 MAPPO baseline。
>
> 对每个追捕者来说，observation 是 16 维，动作空间是 5 个离散动作。DQN 是一个两层 MLP，输出 5 个动作的 Q 值。训练时使用 epsilon-greedy 做探索，初始 epsilon 是 1.0，逐渐降到 0.05。每一步三个追捕者产生的 transition 都会进入同一个 replay buffer，缓冲区足够大后随机采样 batch，用 Bellman target 更新网络。为了稳定训练，我用了 target network、Huber loss 和梯度裁剪。
>
> DQN 改进版加入了课程学习和团队奖励混合。课程学习把 3000 个 episode 分成三个难度阶段，让智能体先学简单任务；团队奖励混合是在每个追捕者自己的 reward 上加入一部分团队平均 reward，引导协作。post-seedfix 重跑里，课程切换在第 1000 和 2000 局附近按预期发生，改进 DQN 中途 eval 最高到 70%，但最终网络在完整难度下回落到 10%。这说明机制已经跑通，但还需要 best checkpoint 复评和多 seed 验证。
>
> 后来我加入 MAPPO，采用 centralized training decentralized execution。actor 执行时只看每个追捕者自己的 16 维 observation，critic 训练时看 62 维全局 observation。MAPPO 训练 3000 episode 后，post-seedfix 最终 10 局捕获率达到 70%，明显高于两个 DQN 最终网络。这说明在这个协作追捕任务里，集中式 critic 比 independent DQN 更适合学习团队配合。

## 21. 你最应该记住的代码入口

| 你要讲的点 | 看哪个文件 |
|---|---|
| 环境怎么创建 | `src/pursuit_lab/envs.py` |
| agent 名称和默认环境参数 | `src/pursuit_lab/constants.py` |
| DQN 网络和优化 | `src/pursuit_lab/dqn.py` |
| MAPPO actor/critic 和训练 | `src/pursuit_lab/mappo.py` |
| 训练主循环 | `src/pursuit_lab/train.py` |
| replay buffer | `src/pursuit_lab/replay.py` |
| 团队奖励 | `src/pursuit_lab/rewards.py` |
| 评估 | `src/pursuit_lab/evaluation.py` |
| 一键跑实验 | `src/pursuit_lab/run_experiments.py` |
| 画图 | `src/pursuit_lab/plot.py` |
| GIF 渲染 | `src/pursuit_lab/render.py` 和 `src/pursuit_lab/rendering.py` |
| 配置 | `configs/*.yaml` |
| 测试 | `tests/*.py` |

## 22. 面试前自查清单

面试前建议你能不看文档回答这些问题：

- 这个项目一句话是什么？
- 为什么逃跑者是随机的？
- 追捕者 observation 和 action space 是多少？
- DQN 的输入输出是什么？
- MAPPO 的 actor 和 critic 分别看什么？
- epsilon-greedy 是什么？
- replay buffer 为什么有用？
- target network 为什么有用？
- PPO clipping 和 GAE 大概是什么？
- 课程学习怎么分阶段？
- 团队奖励混合公式是什么？
- 训练指标和评估指标分别是什么？
- `best.pt` 保存了什么？
- `metrics.csv` 和 `eval_summary.json` 分别代表什么？
- 当前实验结果是什么？
- 为什么 MAPPO 成功率比 DQN 高？
- 这个项目最大的局限是什么？
- 如果继续做，你会怎么改？

如果这些都能讲清楚，面试官基本很难把你问崩。真正高级的回答不是“我的结果一定最好”，而是能说清楚设计选择、实验限制和下一步改进。

