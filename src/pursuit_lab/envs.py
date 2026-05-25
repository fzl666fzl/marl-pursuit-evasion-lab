from __future__ import annotations

from typing import Any

from mpe2 import simple_tag_v3

from pursuit_lab.constants import DEFAULT_ENV_KWARGS

PREY_BASE_MAX_SPEED = 1.3
PREY_BASE_ACCEL = 4.0
CURRICULUM_SPEED_FACTORS = (0.5, 0.75, 1.0)


def make_env(
    *,
    seed: int | None = None,
    render_mode: str | None = None,
    curriculum_stage: int | None = None,
    env_config: dict[str, Any] | None = None,
    **overrides: Any,
):
    kwargs = dict(DEFAULT_ENV_KWARGS)
    if env_config:
        kwargs.update(env_config)
    kwargs.update(overrides)
    if curriculum_stage is not None:
        kwargs["curriculum"] = False
    kwargs["render_mode"] = render_mode

    env = simple_tag_v3.parallel_env(**kwargs)
    if curriculum_stage is not None:
        apply_curriculum_stage(env, int(curriculum_stage))
    if seed is not None:
        for agent in env.possible_agents:
            env.action_space(agent).seed(seed)
    return env


def apply_curriculum_stage(env, stage: int) -> None:
    clamped_stage = max(0, min(int(stage), len(CURRICULUM_SPEED_FACTORS) - 1))
    speed_factor = CURRICULUM_SPEED_FACTORS[clamped_stage]
    for agent in env.unwrapped.world.agents:
        if not agent.adversary:
            agent.max_speed = PREY_BASE_MAX_SPEED * speed_factor
            agent.accel = PREY_BASE_ACCEL * speed_factor


def curriculum_stage_for_episode(episode_index: int, total_episodes: int) -> int:
    if total_episodes <= 0:
        return 2
    first_cut = total_episodes // 3
    second_cut = (2 * total_episodes) // 3
    if episode_index < first_cut:
        return 0
    if episode_index < second_cut:
        return 1
    return 2


def random_actions(env) -> dict[str, int]:
    return {agent: int(env.action_space(agent).sample()) for agent in env.agents}
