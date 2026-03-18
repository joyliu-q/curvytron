"""SLIME custom generate function for curvytron self-play.

This is the entry point that SLIME calls via --custom-generate-function-path.
It replaces the default single-turn generation with a full game episode of
two-agent self-play, returning per-turn Samples with episode-level rewards.
"""

import random

from transformers import AutoTokenizer

from slime.utils.types import Sample

from .agent_system import run_selfplay_game

# Cache tokenizer across calls — loading on every sample is brutal at scale
_tokenizer_cache: dict[str, AutoTokenizer] = {}


def _get_tokenizer(hf_checkpoint: str) -> AutoTokenizer:
    if hf_checkpoint not in _tokenizer_cache:
        _tokenizer_cache[hf_checkpoint] = AutoTokenizer.from_pretrained(
            hf_checkpoint, trust_remote_code=True,
        )
    return _tokenizer_cache[hf_checkpoint]


async def generate_curvytron_selfplay(
    args, sample: Sample, sampling_params, evaluation=False,
) -> list[Sample]:
    """Custom generate function for SLIME GRPO.

    Called once per prompt (game seed). Plays a full self-play episode and
    returns all per-turn Samples for both players, each with rewards assigned.
    """
    tokenizer = _get_tokenizer(args.hf_checkpoint)
    max_context_length = (
        args.rollout_max_context_len if not evaluation else args.eval_max_context_len
    )

    args.sampling_params = sampling_params
    args.rollout_max_context_len = max_context_length
    args.tokenizer = tokenizer

    # Run the full self-play game
    samples = await run_selfplay_game(args, sample)

    # Shuffle so GRPO doesn't see player_a samples always before player_b
    random.shuffle(samples)

    return samples
