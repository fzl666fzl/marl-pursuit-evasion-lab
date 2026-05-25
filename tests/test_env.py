from __future__ import annotations

import pytest

from pursuit_lab.constants import ALL_AGENTS, PREY_AGENT, PURSUER_AGENTS
from pursuit_lab import envs
from pursuit_lab.envs import curriculum_stage_for_episode, make_env, random_actions


def test_simple_tag_env_has_expected_agents() -> None:
    env = make_env(seed=0)
    try:
        observations, _ = env.reset(seed=0)

        assert tuple(env.agents) == ALL_AGENTS
        assert PURSUER_AGENTS == ("adversary_0", "adversary_1", "adversary_2")
        assert PREY_AGENT == "agent_0"
        assert set(observations) == set(ALL_AGENTS)
    finally:
        env.close()


def test_random_policy_runs_five_complete_episodes() -> None:
    env = make_env(seed=1)
    completed = 0
    try:
        for episode in range(5):
            env.reset(seed=episode)
            for _ in range(60):
                actions = random_actions(env)
                _, _, terminations, truncations, _ = env.step(actions)
                if all(terminations.values()) or all(truncations.values()) or not env.agents:
                    completed += 1
                    break

        assert completed == 5
    finally:
        env.close()


def test_curriculum_stage_uses_episode_thirds() -> None:
    total = 3000

    assert curriculum_stage_for_episode(0, total) == 0
    assert curriculum_stage_for_episode(999, total) == 0
    assert curriculum_stage_for_episode(1000, total) == 1
    assert curriculum_stage_for_episode(1999, total) == 1
    assert curriculum_stage_for_episode(2000, total) == 2
    assert curriculum_stage_for_episode(2999, total) == 2


def test_apply_curriculum_stage_scales_prey_difficulty() -> None:
    apply_curriculum_stage = getattr(envs, "apply_curriculum_stage", None)
    assert apply_curriculum_stage is not None

    env = make_env(seed=0)
    try:
        env.reset(seed=0)
        prey = next(agent for agent in env.unwrapped.world.agents if agent.name == PREY_AGENT)

        apply_curriculum_stage(env, 0)
        assert prey.max_speed == pytest.approx(0.65)
        assert prey.accel == pytest.approx(2.0)

        apply_curriculum_stage(env, 1)
        assert prey.max_speed == pytest.approx(0.975)
        assert prey.accel == pytest.approx(3.0)

        apply_curriculum_stage(env, 99)
        assert prey.max_speed == pytest.approx(1.3)
        assert prey.accel == pytest.approx(4.0)
    finally:
        env.close()


def test_make_env_curriculum_stage_scales_prey_after_reset() -> None:
    env = make_env(seed=0, curriculum_stage=0)
    try:
        env.reset(seed=0)
        prey = next(agent for agent in env.unwrapped.world.agents if agent.name == PREY_AGENT)

        assert prey.max_speed == pytest.approx(0.65)
        assert prey.accel == pytest.approx(2.0)
    finally:
        env.close()
