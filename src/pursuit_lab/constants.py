"""Shared constants for the simple_tag_v3 pursuit setup."""

PURSUER_AGENTS = ("adversary_0", "adversary_1", "adversary_2")
PREY_AGENT = "agent_0"
ALL_AGENTS = (*PURSUER_AGENTS, PREY_AGENT)

DEFAULT_ENV_KWARGS = {
    "num_good": 1,
    "num_adversaries": 3,
    "num_obstacles": 2,
    "max_cycles": 50,
    "continuous_actions": False,
    "terminate_on_success": True,
}
