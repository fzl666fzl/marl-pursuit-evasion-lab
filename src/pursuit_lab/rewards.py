from __future__ import annotations

from pursuit_lab.constants import PURSUER_AGENTS


def mix_team_rewards(raw_rewards: dict[str, float], team_weight: float) -> dict[str, float]:
    pursuer_rewards = [float(raw_rewards.get(agent, 0.0)) for agent in PURSUER_AGENTS]
    team_mean = sum(pursuer_rewards) / len(pursuer_rewards)
    return {
        agent: float(raw_rewards.get(agent, 0.0)) + float(team_weight) * team_mean
        for agent in PURSUER_AGENTS
    }
