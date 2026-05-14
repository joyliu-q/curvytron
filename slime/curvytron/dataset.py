"""Curvytron seed dataset for self-play GRPO training.

Generates simple seed prompts ("rl-seed-0", "rl-seed-1", ...) that the
custom generate function uses as game seeds for self-play episodes.
"""

from modal_training_gym import DatasetConfig


class CurvytronSeedDataset(DatasetConfig):
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False

    num_seeds: int = 5000

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None):
        import os

        from datasets import Dataset

        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [{"prompt": f"rl-seed-{i}", "label": ""} for i in range(self.num_seeds)]
        Dataset.from_list(rows).to_parquet(path)

        if eval_paths:
            eval_rows = rows[:100]
            for eval_path in eval_paths.values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                Dataset.from_list(eval_rows).to_parquet(eval_path)
