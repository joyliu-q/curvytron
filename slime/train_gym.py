"""Curvytron GRPO training using modal-training-gym.

Replaces the hand-rolled Modal app in modal_train.py with training-gym's
TrainConfig + SlimeRecipe, which handles cluster topology, Ray bring-up,
volume mounts, checkpointing, and image building.

Usage:
    # Train with default config (Qwen3-4B self-play)
    python slime/train_gym.py

    # Train with 0.6B model
    python slime/train_gym.py --model 0.6b

    # Train with 8B model on 4 nodes
    python slime/train_gym.py --model 8b
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import modal
import requests

from modal_training_gym import (
    Qwen3_0_6B,
    Qwen3_4B,
    Qwen3_8B,
    SlimeRecipe,
    TrainConfig,
    WandbConfig,
)

from curvytron.dataset import CurvytronSeedDataset
from curvytron.multi_agent_rollout import generate_curvytron_multiagent


# ── Image overlay ────────────────────────────────────────────────────────────
# Adds the curvytron package to the training container so the custom
# generate/RM functions can import their sibling modules (game_client, etc.)

def curvytron_image_overlay(image: modal.Image) -> modal.Image:
    return (
        image
        .add_local_dir(
            str(Path(__file__).parent / "curvytron"),
            remote_path="/root/curvytron",
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc"],
        )
        .run_commands(
            "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",
            # Fix rope_theta access for transformers 5.x (moved to rope_parameters dict)
            r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/glm/glm45_bridge.py""",
            r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
        )
    )


# ── Model configs ───────────────────────────────────────────────────────────

MODELS = {
    "4b": Qwen3_4B,
    "0.6b": Qwen3_0_6B,
    "8b": Qwen3_8B,
}


# ── Recipe builders ─────────────────────────────────────────────────────────

def _base_recipe(**overrides) -> SlimeRecipe:
    """Shared recipe defaults for all curvytron self-play configs."""
    defaults = dict(
        # Passthrough RM — rewards are set inside the custom generate function
        custom_rm_path="curvytron.passthrough_rm.passthrough_rm",

        # Cluster
        gpu_type="H100",
        colocate=True,
        sequence_parallel=False,

        # RL algorithm
        advantage_estimator="grpo",
        use_kl_loss=True,
        kl_loss_coef=0.01,
        kl_loss_type="low_var_kl",
        entropy_coef=0.01,
        eps_clip=0.2,
        eps_clip_high=0.28,

        # Rollout
        n_samples_per_prompt=1,
        rollout_shuffle=True,

        # Training
        lr=1e-6,
        lr_decay_style="constant",
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.98,

        # Memory
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,

        # Eval (disabled — use eval_selfplay.py separately)
        n_samples_per_eval_prompt=1,
        eval_max_response_len=50,
        eval_top_p=1.0,

        # Image
        image_overlay=curvytron_image_overlay,
    )
    defaults.update(overrides)
    return SlimeRecipe(**defaults)


def recipe_4b() -> SlimeRecipe:
    return _base_recipe(
        ref_load="Qwen/Qwen3-4B",
        tensor_model_parallel_size=2,
        rollout_num_gpus_per_engine=2,
        sglang_mem_fraction_static=0.7,

        num_rollout=1000,
        rollout_batch_size=64,
        rollout_max_response_len=50,
        rollout_temperature=2.0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=9216,

        global_batch_size=64,
        save_interval=10,

        actor_num_nodes=1,
        actor_num_gpus_per_node=8,

        custom_generate_function=generate_curvytron_multiagent,

        extra_config={
            "rollout_max_context_len": 4096,
            "balance_data": True,
        },

        wandb=WandbConfig(
            project="slime-curvytron",
            group="qwen3-4b-curvytron-selfplay",
            modal_wandb_secret_name="wandb-secret",
        ),
    )


def recipe_0_6b() -> SlimeRecipe:
    return _base_recipe(
        ref_load="Qwen/Qwen3-0.6B",
        tensor_model_parallel_size=1,
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.85,

        num_rollout=2000,
        rollout_batch_size=128,
        rollout_max_response_len=50,
        rollout_temperature=1.0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=16384,

        global_batch_size=128,
        save_interval=10,

        actor_num_nodes=1,
        actor_num_gpus_per_node=8,

        custom_generate_function=generate_curvytron_multiagent,

        extra_config={
            "rollout_max_context_len": 4096,
            "balance_data": True,
        },

        wandb=WandbConfig(
            project="slime-curvytron",
            group="qwen3-0.6b-curvytron-selfplay",
            modal_wandb_secret_name="wandb-secret",
        ),
    )


def recipe_8b() -> SlimeRecipe:
    return _base_recipe(
        ref_load="Qwen/Qwen3-8B",
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        rollout_num_gpus_per_engine=2,
        sglang_mem_fraction_static=0.7,

        num_rollout=1000,
        rollout_batch_size=128,
        rollout_max_response_len=50,
        rollout_temperature=1.0,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=9216,

        global_batch_size=128,
        save_interval=10,

        actor_num_nodes=4,
        actor_num_gpus_per_node=8,

        custom_generate_function=generate_curvytron_multiagent,

        extra_config={
            "rollout_max_context_len": 4096,
            "balance_data": True,
        },

        wandb=WandbConfig(
            project="slime-curvytron",
            group="qwen3-8b-curvytron-selfplay",
            modal_wandb_secret_name="wandb-secret",
        ),
    )


RECIPES = {
    "4b": recipe_4b,
    "0.6b": recipe_0_6b,
    "8b": recipe_8b,
}


# ── Game server auto-deploy ──────────────────────────────────────────────────

CURVYTRON_URL = "https://modal-labs-joy-dev--curvytron-curvytron.us-east.modal.direct"
DEPLOY_SCRIPT = str(Path(__file__).parent.parent / "modal" / "deploy_modal.py")


def _game_server_healthy() -> bool:
    try:
        resp = requests.get(f"{CURVYTRON_URL}/api/rl/sessions", timeout=5)
        return resp.status_code not in (502, 503, 504)
    except (requests.ConnectionError, requests.Timeout, OSError):
        return False


def ensure_game_server():
    """Check if the curvytron game server is up; deploy it if not."""
    if _game_server_healthy():
        print(f"Game server healthy: {CURVYTRON_URL}")
        return

    print(f"Game server not responding at {CURVYTRON_URL}, deploying...")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", DEPLOY_SCRIPT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Deploy stdout: {result.stdout}")
        print(f"Deploy stderr: {result.stderr}")
        raise RuntimeError(f"Failed to deploy game server (exit {result.returncode})")

    print("Deploy complete, waiting for server to become healthy...")
    for i in range(120):
        if _game_server_healthy():
            print(f"Game server ready: {CURVYTRON_URL}")
            return
        if i % 12 == 0 and i > 0:
            print(f"  Still waiting... ({i * 5}s)")
        time.sleep(5)
    print(f"WARNING: Game server still not healthy after 10 min — proceeding anyway."
          f" Rollouts will fail until the server is up.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curvytron GRPO training via training-gym")
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="4b",
        help="Model size (default: 4b)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5000,
        help="Number of game seeds in the dataset (default: 5000)",
    )
    args = parser.parse_args()

    try:
        modal.Secret.from_name("huggingface-secret").hydrate()
    except modal.exception.NotFoundError as e:
        raise RuntimeError(
            "Missing Modal Secret 'huggingface-secret'. Create one at "
            "https://modal.com/secrets with an HF_TOKEN entry."
        ) from e

    ensure_game_server()

    model = MODELS[args.model]()
    dataset = CurvytronSeedDataset(num_seeds=args.num_seeds)
    recipe = RECIPES[args.model]()

    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=recipe,
    )

    print(f"Starting curvytron training: {model.model_name}")
    print(f"  Recipe: {args.model}")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Nodes: {recipe.total_nodes}")
    result = training_run.train()
    print(f"Training complete: {result.training_run_id}")


if __name__ == "__main__":
    main()
