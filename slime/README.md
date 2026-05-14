# Curvytron RL Training

GRPO self-play training for curvytron using [modal-training-gym](https://github.com/modal-projects/training-gym) on Modal.

## Prerequisites

1. **Modal account** — sign up at [modal.com](https://modal.com)
2. **Modal secrets** — create these at [modal.com/secrets](https://modal.com/secrets):
   - `huggingface-secret` with your `HF_TOKEN`
   - `wandb-secret` with your `WANDB_API_KEY` (optional, for logging)
3. **Install dependencies:**

```bash
uv sync
```

## Train

```bash
# Qwen3-4B self-play (default, 1 node × 8 H100s)
python slime/train_gym.py

# Qwen3-0.6B (smaller/faster, 1 node × 8 H100s)
python slime/train_gym.py --model 0.6b

# Qwen3-8B (4 nodes × 8 H100s)
python slime/train_gym.py --model 8b
```

Training-gym handles everything automatically: model download, dataset
preparation, Ray cluster bring-up, checkpoint saving, and HF conversion.
The script also auto-deploys the curvytron game server if it's not running.

The `TrainResult` returned at the end contains the `training_run_id` and
checkpoint paths.

## Evaluate

Compare a trained checkpoint against a baseline by playing games on the
curvytron server:

```bash
# Eval latest checkpoint from a training run vs base model
python slime/eval_gym.py \
    --training-run-id "Qwen/Qwen3-0.6B.546ca7ad-..." \
    --num-games 10

# Eval a specific checkpoint (0 = earliest, -1 = latest)
python slime/eval_gym.py \
    --training-run-id "..." \
    --checkpoint-idx 0 \
    --num-games 20

# Custom baseline
python slime/eval_gym.py \
    --training-run-id "..." \
    --baseline Qwen/Qwen3-4B
```

This uses training-gym's `DeploymentConfig.serve()` to deploy both models
(handling Megatron→HF conversion automatically) and `EvalConfig` with a
custom `eval_fn` to play games and report win rates.

## Serve a trained bot

Deploy a trained checkpoint as a live SGLang endpoint:

```bash
modal deploy slime/serve_bot.py
```

Edit `MODEL_PATH` in `serve_bot.py` to point at your checkpoint.

## Watch replays

During training, every 10th game is recorded to `/data/game_traces/`.
View them with the replay viewer:

```bash
modal serve slime/game_viewer.py
```

## How it works

```
train_gym.py
  └─ TrainConfig(model, dataset, recipe).train()
       ├─ CurvytronSeedDataset.prepare()     → writes game seeds to volume
       ├─ SlimeRecipe                         → GRPO hyperparams + custom hooks
       │    ├─ custom_generate_function_path  → curvytron.multi_agent_rollout
       │    └─ custom_rm_path                 → curvytron.passthrough_rm
       └─ training-gym launcher              → Modal app, Ray cluster, SLIME
```

For each seed in the dataset, the custom generate function:

1. Creates a game session on the curvytron server
2. Two agents (both the training model) play against each other
3. Each turn, SGLang generates an action with constrained decoding (`left|straight|right`)
4. Per-step reward = escalating survival bonus + reachable-space fraction
5. All per-turn Samples are returned to SLIME for GRPO training

The passthrough RM ensures SLIME uses the rewards computed during the game
rather than running its own reward model.

## Files

| File | Purpose |
|------|---------|
| `train_gym.py` | Training entry point (replaces old `modal_train.py`) |
| `curvytron/dataset.py` | `CurvytronSeedDataset` — generates game seed dataset |
| `curvytron/multi_agent_rollout.py` | SLIME custom generate function entry point |
| `curvytron/multi_agent_system.py` | Self-play game loop, prompts, rewards |
| `curvytron/passthrough_rm.py` | No-op RM (rewards set by generate function) |
| `curvytron/game_client.py` | Async HTTP client for the curvytron game server |
| `curvytron/prompts.py` | System prompt for the game |
| `eval_gym.py` | Checkpoint vs baseline evaluation (training-gym) |
| `eval_selfplay.py` | Legacy eval (deprecated) |
| `serve_bot.py` | Deploy trained model as live SGLang endpoint |
| `game_viewer.py` | Replay viewer for recorded game traces |

## Configuration

The game server URL is configured in `curvytron/game_client.py` via the
`CURVYTRON_URL` environment variable (defaults to the Modal-deployed instance).
