from __future__ import annotations

from pursuit_lab.metrics import summarize_episodes
from pursuit_lab.rewards import mix_team_rewards


def test_team_reward_mixing_changes_training_rewards_only() -> None:
    raw_rewards = {"adversary_0": 1.0, "adversary_1": 2.0, "adversary_2": 3.0, "agent_0": -6.0}
    mixed = mix_team_rewards(raw_rewards, team_weight=0.3)

    assert mixed == {
        "adversary_0": 1.6,
        "adversary_1": 2.6,
        "adversary_2": 3.6,
    }
    assert raw_rewards["adversary_0"] == 1.0


def test_eval_summary_uses_raw_episode_rewards() -> None:
    summary = summarize_episodes(
        [
            {"captured": True, "steps": 7, "episode_reward": 4.0},
            {"captured": False, "steps": 50, "episode_reward": -1.0},
        ],
        max_cycles=50,
    )

    assert summary["capture_rate"] == 0.5
    assert summary["success_episode_count"] == 1
    assert summary["mean_steps_to_capture"] == 7.0
    assert summary["mean_episode_reward"] == 1.5
