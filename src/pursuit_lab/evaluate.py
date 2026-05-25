from __future__ import annotations

import argparse
import json
from pathlib import Path

from pursuit_lab.evaluation import evaluate_checkpoint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained pursuit_lab checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--output", default=None, help="Directory for eval_summary.json and eval_episodes.csv.")
    args = parser.parse_args(argv)

    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.checkpoint).resolve().parent)
    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
