"""
Unified SLIME GRPO training script for Modal.

Usage:
    modal run modal_train.py::train

    USE_LOCAL_SLIME=/path/to/slime modal run modal_train.py::train

    # Download model
    modal run slime/modal_train.py::download_model --config qwen-4b-sync

    # Prepare dataset
    modal run slime/modal_train.py::prepare_curvytron_dataset

Environment variables:
    USE_LOCAL_SLIME=/path     Path to local slime repo for development
    SLIME_APP_NAME=...        Override Modal app name

Available configs (main):
    - qwen-4b, glm-4-7, glm-4-7-flash

Available configs (test-configs):
    - qwen-4b-lora (LoRA training test config)
"""

import os
import subprocess
from pathlib import Path
from typing import Optional
import time

import modal
import modal.experimental

from configs.base import RLConfig
from configs import get_config as _get_config


DEFAULT_CONFIG = "curvytron-selfplay"


def get_config(run_name: str = DEFAULT_CONFIG) -> RLConfig:
    """Load a training config by name."""
    return _get_config(run_name)


# =============================================================================
# Modal Image & Volumes
# =============================================================================

# Path to local slime repo for development (e.g., USE_LOCAL_SLIME=/path/to/slime)
# Set to a directory path to overlay local slime code, or leave unset to use registry image
LOCAL_SLIME_PATH = os.environ.get("USE_LOCAL_SLIME", "")

image = (
    modal.Image.from_registry("slimerl/slime:nightly-dev-20260126a")
    .run_commands(
        "uv pip install --system git+https://github.com/huggingface/transformers.git@eebf856",  # 4.54.1
        """sed -i 's/AutoImageProcessor.register(config, None, image_processor, None, exist_ok=True)/AutoImageProcessor.register(config, slow_image_processor_class=image_processor, exist_ok=True)/g' /sgl-workspace/sglang/python/sglang/srt/configs/utils.py""",
        # Fix rope_theta access for transformers 5.x (moved to rope_parameters dict)
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/glm/glm45_bridge.py""",
        r"""sed -i 's/hf_config\.rope_theta/hf_config.rope_parameters["rope_theta"]/g' /usr/local/lib/python3.12/dist-packages/megatron/bridge/models/qwen/qwen3_bridge.py""",
    )
    .entrypoint([])
    .add_local_python_source("configs", copy=True)
    .add_local_dir("slime/curvytron", remote_path="/root/curvytron", copy=True)
)

# Overlay local slime code for development
# Install slime to /opt/slime-dev (not /root/slime) to avoid sys.path conflicts when Ray runs scripts
SLIME_DEV_PATH = "/opt/slime-dev"
if LOCAL_SLIME_PATH:
    # Copy the entire slime repo (has pyproject.toml) and install it
    image = image.add_local_dir(LOCAL_SLIME_PATH, remote_path=SLIME_DEV_PATH, copy=True, ignore=["**/__pycache__", "**/*.pyc", "**/.git", "**/.venv", "**/modal"]).run_commands(f"uv pip install --system -e {SLIME_DEV_PATH}")
else:
    SLIME_DEV_PATH = None

with image.imports():
    import ray
    from ray.job_submission import JobSubmissionClient

# Paths
HF_CACHE_PATH = "/root/.cache/huggingface"
DATA_PATH: Path = Path("/data")
CHECKPOINTS_PATH: Path = Path("/checkpoints")

# Volumes
data_volume: modal.Volume = modal.Volume.from_name(
    "curvytron-data", create_if_missing=True
)
hf_cache_vol: modal.Volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
checkpoints_volume: modal.Volume = modal.Volume.from_name("curvytron-checkpoints", create_if_missing=True)

# Ray configuration
RAY_PORT = 6379
RAY_DASHBOARD_PORT = 8265
SINGLE_NODE_MASTER_ADDR = "127.0.0.1"

app = modal.App("curvytron")


# =============================================================================
# Ray Initialization
# =============================================================================


def _init_ray(rank: int, main_node_addr: str, node_ip_addr: str, n_nodes: int):
    """Initialize Ray cluster across Modal containers.

    Rank 0 starts the head node, opens a tunnel to the Ray dashboard, and waits
    for all worker nodes to connect. Other ranks start as workers and connect
    to the head node address.
    """
    os.environ["SLIME_HOST_IP"] = node_ip_addr

    if rank == 0:
        print(f"Starting Ray head node at {node_ip_addr}")
        subprocess.Popen(
            [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={node_ip_addr}",
                "--dashboard-host=0.0.0.0",
            ]
        )

        for _ in range(30):
            try:
                ray.init(address="auto")
            except ConnectionError:
                time.sleep(1)
                continue
            print("Connected to Ray")
            break
        else:
            raise Exception("Failed to connect to Ray")

        for _ in range(60):
            print("Waiting for worker nodes to connect...")
            alive_nodes = [n for n in ray.nodes() if n["Alive"]]
            print(f"Alive nodes: {len(alive_nodes)}/{n_nodes}")

            if len(alive_nodes) == n_nodes:
                print("All worker nodes connected")
                break
            time.sleep(1)
        else:
            raise Exception("Failed to connect to all worker nodes")
    else:
        print(f"Starting Ray worker node at {node_ip_addr}, connecting to {main_node_addr}")
        subprocess.Popen(
            [
                "ray",
                "start",
                f"--node-ip-address={node_ip_addr}",
                "--address",
                f"{main_node_addr}:{RAY_PORT}",
            ]
        )


# =============================================================================
# Training Command Generation
# =============================================================================


def generate_slime_cmd(
    config: RLConfig,
    master_addr: str,
    experiment_name: str,
) -> tuple[str, dict]:
    """Generate the slime training command and runtime environment."""
    import datetime
    import random

    from huggingface_hub import snapshot_download

    hf_model_path = snapshot_download(repo_id=config.model_id, local_files_only=True)
    checkpoint_dir = CHECKPOINTS_PATH / experiment_name
    train_args = config.generate_train_args(hf_model_path, checkpoint_dir, DATA_PATH, is_infinite_run=False)

    train_args += f" --save {checkpoint_dir} --save-interval {config.save_steps if hasattr(config, 'save_steps') else 10}"

    # Add wandb args if API key is available
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d-%H%M%S") + f"-{random.randint(0, 999):03d}"
        wandb_run_name = f"{config.wandb_run_name_prefix}_{run_id}" if config.wandb_run_name_prefix else run_id
        train_args += f" --use-wandb --wandb-project {config.wandb_project} --wandb-group {wandb_run_name} --wandb-key '{wandb_key}' --disable-wandb-random-suffix"

    # Build PYTHONPATH by appending to existing (don't clobber)
    import os as _os
    existing_pythonpath = _os.environ.get("PYTHONPATH", "")
    megatron_path = "/root/Megatron-LM/"
    pythonpath = f"{megatron_path}:{existing_pythonpath}" if existing_pythonpath else megatron_path

    runtime_env = {
        "env_vars": {
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "no_proxy": master_addr,
            "MASTER_ADDR": master_addr,
            # Megatron-LM requires PYTHONPATH (pip install doesn't work due to package name mismatch)
            # slime is pip installed so doesn't need to be on PYTHONPATH
            "PYTHONPATH": pythonpath,
        }
    }

    # Use full path when local slime is installed
    # Note: config.train_script returns "slime/train.py" for base image,
    # but local repo has train.py at root level
    # Check at runtime if dev path exists (USE_LOCAL_SLIME is only set during image build)
    dev_path = "/opt/slime-dev"
    if os.path.exists(dev_path):
        train_script = f"{dev_path}/train.py"
    else:
        train_script = "slime/train.py"

    return f"python3 {train_script} {train_args}", runtime_env


async def run_training(
    config: RLConfig,
    n_nodes: int,
    master_addr: str,
    experiment_name: str, 
):
    """Submit SLIME training job to Ray cluster and stream logs."""
    client = JobSubmissionClient("http://127.0.0.1:8265")

    slime_cmd, runtime_env = generate_slime_cmd(config, master_addr, experiment_name)

    print("Submitting training job...")
    print(f"  Model: {config.model_name}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Checkpoint dir: {CHECKPOINTS_PATH / experiment_name}")

    job_id = client.submit_job(entrypoint=slime_cmd, runtime_env=runtime_env)
    print(f"Job submitted with ID: {job_id}")

    async for line in client.tail_job_logs(job_id):
        print(line, end="", flush=True)

    await checkpoints_volume.commit.aio()
    print("Checkpoints saved and committed to volume")

        


# =============================================================================
# Modal Functions
# =============================================================================


@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=24 * 60 * 60,
)
def download_model(
    run_name: str = DEFAULT_CONFIG,
    revision: Optional[str] = None,
):
    """Download model from HuggingFace."""
    from huggingface_hub import snapshot_download

    cfg = get_config(run_name)

    path = snapshot_download(
        repo_id=cfg.model_id,
        revision=revision,
    )
    print(f"Model downloaded to {path}")

    hf_cache_vol.commit()



@app.function(
    image=image,
    volumes={DATA_PATH.as_posix(): data_volume},
    timeout=24 * 60 * 60,
)
def prepare_curvytron_dataset(num_seeds: int = 5000, force: bool = False):
    """Generate curvytron seed dataset for self-play GRPO training."""
    import json

    data_volume.reload()
    seeds_path = DATA_PATH / "curvytron_seeds.jsonl"

    if seeds_path.exists() and not force:
        lines = sum(1 for _ in open(seeds_path))
        print(f"Dataset already exists: {seeds_path} ({lines} seeds). Use --force to regenerate.")
        return

    with open(seeds_path, "w") as f:
        for i in range(num_seeds):
            f.write(json.dumps({"prompt": f"rl-seed-{i}", "label": ""}) + "\n")

    data_volume.commit()
    print(f"Wrote {num_seeds} seeds to {seeds_path}")


@app.local_entrypoint()
def list_available_configs():
    """List all available training configs."""
    from configs import list_configs

    print("Available configs:")
    for name in list_configs():
        print(f"  - {name}")



@app.function(
    image=image,
    gpu="H100:1",
    timeout=24 * 60 * 60,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
    volumes={CHECKPOINTS_PATH.as_posix(): checkpoints_volume},
)
async def convert_checkpoint(
    model_path: str,
    iter_dir: str,
    origin_hf_dir: str
):
    """Convert Megatron checkpoint to HuggingFace format."""
    from huggingface_hub import snapshot_download
    import glob
    import os
    import subprocess

    await checkpoints_volume.reload.aio()

    local_hf_dir = CHECKPOINTS_PATH / origin_hf_dir

    if not local_hf_dir.exists():
        snapshot_download(repo_id=f"Qwen/{origin_hf_dir}", local_dir=local_hf_dir)
    else:
        print(f"Model {origin_hf_dir} already downloaded.")

    megatron_checkpoint_path = CHECKPOINTS_PATH / model_path / iter_dir
    output_hf_path = CHECKPOINTS_PATH / model_path / f"{iter_dir}_hf"

    # Find the conversion script — check known locations
    candidates = [
        "/root/tools/convert_torch_dist_to_hf.py",
        *glob.glob("/root/**/convert_torch_dist_to_hf.py", recursive=True),
        *glob.glob("/usr/**/convert_torch_dist_to_hf.py", recursive=True),
        *glob.glob("/opt/**/convert_torch_dist_to_hf.py", recursive=True),
    ]
    script_path = None
    for c in candidates:
        if os.path.exists(c):
            script_path = c
            break
    if script_path is None:
        # List what's available for debugging
        tools_contents = os.listdir("/root") if os.path.isdir("/root") else []
        raise RuntimeError(
            f"Conversion script not found. Searched: {candidates[:5]}\n"
            f"/root contents: {tools_contents}"
        )
    print(f"Found conversion script at: {script_path}")

    cmd = [
        "python",
        script_path,
        "--input-dir",
        str(megatron_checkpoint_path),
        "--output-dir",
        str(output_hf_path),
        "--origin-hf-dir",
        str(local_hf_dir),
        "--force",
    ]
    print(f"Running conversion: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd="/root",
        env={**os.environ, "PYTHONPATH": "/root/Megatron-LM:/root"},
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            "Checkpoint conversion failed.\n"
            f"input-dir={megatron_checkpoint_path}\n"
            f"output-dir={output_hf_path}\n"
            f"origin-hf-dir={local_hf_dir}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    await checkpoints_volume.commit.aio()
    print(f"Converted checkpoint saved to {output_hf_path}")


# =============================================================================
# CLI Entry Points
# =============================================================================


@app.function(
    image=image,
    gpu="H100:8",
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        DATA_PATH.as_posix(): data_volume,
        CHECKPOINTS_PATH.as_posix(): checkpoints_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=24 * 60 * 60,
    experimental_options={
        "efa_enabled": True,
    },
)
async def train(
    run_name: str = "curvytron-selfplay",
    eval_games: int = 20,
):
    """Single-node GRPO training on Modal.

    After training, converts the latest checkpoint to HF format and
    evaluates it against previous checkpoints via curvytron self-play.
    """
    from datetime import datetime

    cfg = get_config(run_name=run_name)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_short = cfg.model_name.split("/")[-1]
    experiment_name = f"{run_name}-{model_short}-{timestamp}"

    await hf_cache_vol.reload.aio()
    await data_volume.reload.aio()
    await checkpoints_volume.reload.aio()

    _init_ray(0, SINGLE_NODE_MASTER_ADDR, SINGLE_NODE_MASTER_ADDR, 1)

    async with modal.forward(RAY_DASHBOARD_PORT) as tunnel:
        print(f"Dashboard URL: {tunnel.url}")
        print(f"Experiment: {experiment_name}")
        await run_training(cfg, 1, SINGLE_NODE_MASTER_ADDR, experiment_name)

    # ── Post-training eval: latest checkpoint vs base model ────────────
    if eval_games > 0:
        print(f"\n{'='*60}")
        print("Post-training eval: converting checkpoints and running games")
        print(f"{'='*60}")

        checkpoint_dir = CHECKPOINTS_PATH / experiment_name

        # Find the latest iteration checkpoint
        latest_iter_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
        if latest_iter_file.exists():
            latest_iter = int(latest_iter_file.read_text().strip())
        else:
            # Find highest iter_* directory
            import glob
            iters = sorted(glob.glob(str(checkpoint_dir / "iter_*")))
            if not iters:
                print("No checkpoints found — skipping eval")
                return
            latest_iter = int(Path(iters[-1]).name.split("_")[1])

        iter_dir = f"iter_{latest_iter:07d}"
        hf_dir = f"{iter_dir}_hf"
        hf_path = checkpoint_dir / hf_dir

        # Convert latest checkpoint to HF if not already done
        if not hf_path.exists():
            print(f"Converting {experiment_name}/{iter_dir} to HF format...")
            await convert_checkpoint.remote.aio(
                model_path=experiment_name,
                iter_dir=iter_dir,
                origin_hf_dir=model_short,
            )
            await checkpoints_volume.reload.aio()

        # Also convert an earlier checkpoint for comparison (if available)
        # Find the earliest iter checkpoint
        import glob
        all_iters = sorted(glob.glob(str(checkpoint_dir / "iter_*")))
        megatron_iters = [p for p in all_iters if not p.endswith("_hf")]

        eval_pairs = []
        # Always eval latest vs base model
        eval_pairs.append((f"{experiment_name}/{hf_dir}", cfg.model_id))

        # If we have an earlier checkpoint, eval latest vs earliest
        if len(megatron_iters) >= 2:
            early_iter_dir = Path(megatron_iters[0]).name
            early_hf_dir = f"{early_iter_dir}_hf"
            early_hf_path = checkpoint_dir / early_hf_dir
            if not early_hf_path.exists():
                print(f"Converting {experiment_name}/{early_iter_dir} to HF format...")
                await convert_checkpoint.remote.aio(
                    model_path=experiment_name,
                    iter_dir=early_iter_dir,
                    origin_hf_dir=model_short,
                )
                await checkpoints_volume.reload.aio()
            eval_pairs.append((f"{experiment_name}/{hf_dir}", f"{experiment_name}/{early_hf_dir}"))

        # Print eval commands the user can run separately
        print(f"\n{'='*60}")
        print("Run these eval commands manually:")
        for current, baseline in eval_pairs:
            print(f"  modal run slime/eval_selfplay.py::eval_vs_baseline "
                  f"--current-path '{current}' --baseline-path '{baseline}' --num-games {eval_games}")
        print(f"{'='*60}")
