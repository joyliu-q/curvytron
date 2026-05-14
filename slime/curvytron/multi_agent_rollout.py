"""SLIME custom generate function for curvytron multi-agent self-play.

This is the entry point that SLIME calls via --custom-generate-function-path.
It plays a full self-play episode (two agents, both the training model) and
returns per-turn Samples with per-step survival rewards.
"""

import asyncio
import random

from transformers import AutoTokenizer

from slime.utils.types import Sample

from .multi_agent_system import run_selfplay_game

_tokenizer_cache: dict[str, AutoTokenizer] = {}

# Limit concurrent game sessions to avoid overwhelming the game server
_game_semaphore = asyncio.Semaphore(32)


def _get_tokenizer(hf_checkpoint: str) -> AutoTokenizer:
    if hf_checkpoint not in _tokenizer_cache:
        _tokenizer_cache[hf_checkpoint] = AutoTokenizer.from_pretrained(
            hf_checkpoint, trust_remote_code=True,
        )
    return _tokenizer_cache[hf_checkpoint]


async def generate_curvytron_multiagent(
    args, sample: Sample, sampling_params, evaluation=False,
) -> list[Sample]:
    """Custom generate function for SLIME GRPO.

    Called once per prompt (game seed). Plays a full self-play episode and
    returns all per-turn Samples for both players, each with per-step
    survival reward (1.0 if alive, 0.0 if dead).
    """
    tokenizer = _get_tokenizer(args.hf_checkpoint)
    max_context_length = (
        args.rollout_max_context_len if not evaluation else args.eval_max_context_len
    )

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    async with _game_semaphore:
        samples = await run_selfplay_game(args, sample)

    # SLIME crashes on empty groups (IndexError at sglang_rollout.py:375)
    # and Megatron crashes with narrow() on zero-length sequences.
    # Return a properly-formed dummy sample if the game failed.
    if not samples:
        from copy import deepcopy

        tokenizer = _get_tokenizer(args.hf_checkpoint)
        dummy_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": "game"}, {"role": "user", "content": "choose"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        dummy_ids = tokenizer(dummy_prompt, add_special_tokens=False)["input_ids"]
        action_ids = tokenizer("straight", add_special_tokens=False)["input_ids"]

        fallback = deepcopy(sample)
        fallback.prompt = dummy_prompt
        fallback.tokens = dummy_ids + action_ids
        fallback.response = "straight"
        fallback.response_length = len(action_ids)
        fallback.reward = 0.0
        fallback.status = Sample.Status.COMPLETED
        return [fallback]

    # Debug: log reward state before returning to SLIME
    rewards = [s.reward for s in samples]
    none_count = sum(1 for r in rewards if r is None)
    if none_count:
        print(f"[curvytron-ma] WARNING: {none_count}/{len(samples)} samples have None reward for seed {sample.prompt}")
    else:
        print(f"[curvytron-ma] OK: {len(samples)} samples, rewards={rewards[:6]}")

    # Shuffle so GRPO doesn't see player_a samples always before player_b
    random.shuffle(samples)

    return samples
