from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pursuit_lab.constants import PURSUER_AGENTS
from pursuit_lab.dqn import select_epsilon_greedy_action
from pursuit_lab.envs import make_env
from pursuit_lab.evaluation import build_policy_from_checkpoint, episode_seed, load_checkpoint
from pursuit_lab.rendering import render_world_frame


def render_policy_frames(*, checkpoint: dict, episodes: int) -> list[np.ndarray]:
    policy = build_policy_from_checkpoint(checkpoint)
    config = checkpoint["config"]
    frames: list[np.ndarray] = []
    max_cycles = int(config["env"].get("max_cycles", 50))
    seeds = list(config["evaluation"]["seeds"])

    for episode in range(episodes):
        seed = episode_seed(seeds, episode)
        rng = np.random.default_rng(seed)
        env = make_env(seed=seed, env_config=config["env"])
        try:
            observations, _ = env.reset(seed=seed)
            frames.append(render_world_frame(env))
            for _ in range(max_cycles):
                actions: dict[str, int] = {}
                for agent in env.agents:
                    action_space = env.action_space(agent)
                    if agent in PURSUER_AGENTS:
                        actions[agent] = select_epsilon_greedy_action(
                            network=policy,
                            observation=observations[agent],
                            epsilon=0.0,
                            action_dim=int(action_space.n),
                            rng=rng,
                        )
                    else:
                        actions[agent] = int(action_space.sample())

                observations, _, terminations, truncations, _ = env.step(actions)
                frames.append(render_world_frame(env))
                if all(terminations.values()) or all(truncations.values()) or not env.agents:
                    break
        finally:
            env.close()

    return frames


def render_checkpoint(
    *,
    checkpoint_path: str | Path,
    episodes: int,
    output_path: str | Path,
    fps: int = 12,
) -> Path:
    checkpoint = load_checkpoint(checkpoint_path)
    frames = render_policy_frames(checkpoint=checkpoint, episodes=episodes)
    if not frames:
        raise RuntimeError("No render frames were produced by the environment.")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, frames, fps=fps)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render a trained pursuit policy to GIF.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render.")
    parser.add_argument("--output", default="videos/pursuit_demo.gif", help="Output GIF path.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frame rate.")
    args = parser.parse_args(argv)

    print(render_checkpoint(checkpoint_path=args.checkpoint, episodes=args.episodes, output_path=args.output, fps=args.fps))


if __name__ == "__main__":
    main()
