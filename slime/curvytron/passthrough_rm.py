"""Passthrough reward model for SLIME.

Used with --custom-rm-path when the custom generate function already
computes rewards (e.g., multi-agent self-play). Returns the existing
reward on each sample without recomputing.
"""

from slime.utils.types import Sample


async def passthrough_rm(args, samples, **kwargs):
    """No-op RM — rewards were already set by the custom generate function.

    Called by both batched_async_rm (list[Sample]) and async_rm (single Sample).
    """
    if isinstance(samples, list):
        return [s.reward if s.reward is not None else 0.0 for s in samples]
    return samples.reward if samples.reward is not None else 0.0
