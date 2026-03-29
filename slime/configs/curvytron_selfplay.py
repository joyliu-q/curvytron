"""Configuration for Qwen3-4B curvytron multi-agent self-play GRPO training."""

from .base import (
    RLConfig,
    QWEN3_4B_MODEL_ARGS,
    DEFAULT_TRAINING_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-4B",
        model_id="Qwen/Qwen3-4B",

        # Modal settings
        n_nodes=1,
        gpu="H100:8",
        app_name="slime-curvytron",
        sync=True,

        # Wandb
        wandb_project="slime-curvytron",
        wandb_run_name_prefix="qwen3-4b-curvytron-selfplay",

        slime_args=f"""
            # Model architecture
            {QWEN3_4B_MODEL_ARGS}

            # Training parallelism and optimization
            {DEFAULT_TRAINING_ARGS}

            # Optimizer
            {DEFAULT_OPTIMIZER_ARGS}

            # GRPO algorithm
            {DEFAULT_GRPO_ARGS}

            # Custom generate function — multi-agent curvytron rollout
            --custom-generate-function-path curvytron.multi_agent_rollout.generate_curvytron_multiagent

            # Data — seeds dataset (prompt is the game seed, label is unused)
            --prompt-data {{data_path}}/curvytron_seeds.jsonl
            --input-key prompt
            --label-key label
            --rollout-shuffle

            # Passthrough RM — rewards are already computed inside the custom
            # generate function (multi-agent self-play). This prevents SLIME
            # from trying to run its own RM and overwriting our game rewards.
            --custom-rm-path curvytron.passthrough_rm.passthrough_rm

            # Rollout settings
            --num-rollout 1000
            --rollout-batch-size 16
            --n-samples-per-prompt 1
            --rollout-max-context-len 4096
            --rollout-max-response-len 50
            --rollout-temperature 1

            --global-batch-size 16
            --balance-data

            # SGLang
            --rollout-num-gpus-per-engine 2
            --sglang-mem-fraction-static 0.7

            # Orchestration
            --actor-num-nodes 1
            --actor-num-gpus-per-node 8
            --colocate

            # Eval (disabled — use scripts/eval_llm.py separately)
            --n-samples-per-eval-prompt 1
            --eval-max-response-len 50
            --eval-top-p 1
        """,
    )
