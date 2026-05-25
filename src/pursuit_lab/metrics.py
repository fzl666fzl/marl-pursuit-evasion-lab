from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def summarize_episodes(episodes: list[dict[str, Any]], *, max_cycles: int) -> dict[str, float | int]:
    if not episodes:
        return {
            "episodes": 0,
            "capture_rate": 0.0,
            "mean_episode_reward": 0.0,
            "mean_steps_to_capture": float(max_cycles),
            "success_episode_count": 0,
        }

    successes = [episode for episode in episodes if episode["captured"]]
    return {
        "episodes": len(episodes),
        "capture_rate": len(successes) / len(episodes),
        "mean_episode_reward": mean(float(episode["episode_reward"]) for episode in episodes),
        "mean_steps_to_capture": (
            mean(float(episode["steps"]) for episode in successes) if successes else float(max_cycles)
        ),
        "success_episode_count": len(successes),
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
