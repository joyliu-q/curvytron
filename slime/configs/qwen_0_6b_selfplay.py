"""Configuration for Qwen3-0.6B curvytron multi-agent self-play GRPO training.

Scaled down from the 4B self-play config:
- TP=1 (model fits on a single GPU)
- 1 GPU per SGLang engine (8 engines on 8 GPUs for higher throughput)
- Larger rollout batch size (model is ~6x smaller than 4B)
"""

from .base import (
    RLConfig,
    QWEN3_0_6B_MODEL_ARGS,
    DEFAULT_OPTIMIZER_ARGS,
    DEFAULT_GRPO_ARGS,
)


def get_config() -> RLConfig:
    return RLConfig(
        model_name="Qwen3-0.6B",
        model_id="Qwen/Qwen3-0.6B",

        # Modal settings
        n_nodes=1,
        gpu="H100:8",
        app_name="slime-curvytron-0.6b",
        sync=True,

        # Wandb
        wandb_project="slime-curvytron",
        wandb_run_name_prefix="qwen3-0.6b-curvytron-selfplay",

        slime_args=f"""
            # Model architecture
            {QWEN3_0_6B_MODEL_ARGS}

            # Training parallelism and optimization (TP=1 for 0.6B)
            --tensor-model-parallel-size 1
            --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
            --use-dynamic-batch-size --max-tokens-per-gpu 16384
            --megatron-to-hf-mode bridge
            --attention-dropout 0.0 --hidden-dropout 0.0
            --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32

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

            # Rollout settings (larger batches — 0.6B is much smaller)
            --num-rollout 2000
            --rollout-batch-size 128
            --n-samples-per-prompt 1
            --rollout-max-context-len 4096
            --rollout-max-response-len 50
            --rollout-temperature 1

            --global-batch-size 128
            --balance-data

            # SGLang (1 GPU per engine — model easily fits on 1 GPU)
            --rollout-num-gpus-per-engine 1
            --sglang-mem-fraction-static 0.85

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
