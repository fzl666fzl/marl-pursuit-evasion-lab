from __future__ import annotations

import csv
import json
from pathlib import Path

from pursuit_lab.evaluate_runs import evaluate_run_checkpoints, find_checkpoint_runs


def test_find_checkpoint_runs_returns_runs_with_best_checkpoint(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    (runs_dir / "dqn_baseline").mkdir(parents=True)
    (runs_dir / "dqn_baseline" / "best.pt").write_bytes(b"checkpoint")
    (runs_dir / "random_baseline").mkdir()
    (runs_dir / "random_baseline" / "eval_summary.json").write_text("{}", encoding="utf-8")

    runs = find_checkpoint_runs(runs_dir)

    assert runs == [runs_dir / "dqn_baseline"]


def test_evaluate_run_checkpoints_writes_summaries_and_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "eval_best"
    for name in ["dqn_baseline", "mappo_baseline"]:
        (runs_dir / name).mkdir(parents=True)
        (runs_dir / name / "best.pt").write_bytes(b"checkpoint")

    def fake_evaluate_checkpoint(*, checkpoint_path, episodes, output_dir, seeds=None):
        experiment = Path(checkpoint_path).parent.name
        summary = {
            "episodes": episodes,
            "capture_rate": 0.5 if experiment == "dqn_baseline" else 0.8,
            "mean_episode_reward": 15.0 if experiment == "dqn_baseline" else 24.0,
            "mean_steps_to_capture": 12.0 if experiment == "dqn_baseline" else 9.0,
            "success_episode_count": 5 if experiment == "dqn_baseline" else 8,
        }
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "eval_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (target / "eval_episodes.csv").write_text("episode,captured\n1,True\n", encoding="utf-8")
        return summary

    monkeypatch.setattr("pursuit_lab.evaluate_runs.evaluate_checkpoint", fake_evaluate_checkpoint)

    rows = evaluate_run_checkpoints(runs_dir=runs_dir, output_dir=output_dir, episodes=10)

    assert [row["experiment"] for row in rows] == ["dqn_baseline", "mappo_baseline"]
    assert (output_dir / "dqn_baseline" / "eval_summary.json").is_file()
    assert (output_dir / "mappo_baseline" / "eval_episodes.csv").is_file()
    with (output_dir / "eval_comparison.csv").open("r", encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    assert comparison_rows[0]["capture_rate"] == "0.5"
    assert comparison_rows[1]["capture_rate"] == "0.8"
