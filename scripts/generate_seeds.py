#!/usr/bin/env python3
"""Generate a JSONL dataset of game seeds for SLIME GRPO training.

Usage:
    python scripts/generate_seeds.py --num-seeds 5000 --output data/curvytron_seeds.jsonl
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Generate curvytron seed dataset")
    parser.add_argument("--num-seeds", type=int, default=5000, help="Number of seeds")
    parser.add_argument("--output", "-o", default="data/curvytron_seeds.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--prefix", default="rl-seed", help="Seed prefix")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        for i in range(args.num_seeds):
            record = {
                "prompt": f"{args.prefix}-{i}",
                "label": "",
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {args.num_seeds} seeds to {args.output}")


if __name__ == "__main__":
    main()
