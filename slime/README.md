# slime

## GRPO training with SLIME on Modal

### Setup (one-time)

```bash
modal run slime/modal_train.py::prepare_curvytron_dataset
modal run slime/modal_train.py::download_model --config curvytron-selfplay
```

### Train

```bash
modal run slime/modal_train.py::train_multi_node --config curvytron-selfplay
```

The game server URL is configured in `curvytron/game_client.py` (defaults to the
Modal-deployed curvytron instance).

### How it works

The `curvytron-selfplay` config uses `--custom-generate-function-path` to replace
SLIME's default generation with a self-play game loop (`curvytron.rollout.generate_curvytron_selfplay`).

For each seed in the dataset:

1. Creates a game session on the curvytron server
2. Two agents (both the training model) play against each other
3. Each turn, SGLang generates an action for both players (with logprobs)
4. After the game ends, rewards are assigned retroactively:
   `reward = survival_seconds + 10 if won`
5. All per-turn Samples are returned to SLIME for GRPO training

### Other configs

```bash
# Math baseline configs
modal run slime/modal_train.py::prepare_dataset
modal run slime/modal_train.py::train_multi_node --config qwen-4b
modal run slime/modal_train.py::train_multi_node --config qwen-8b-multi
```
