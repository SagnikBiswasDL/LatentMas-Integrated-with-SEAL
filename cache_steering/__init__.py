"""Cache steering for LatentMAS.

Implements the one-shot key-value cache intervention from
"KV Cache Steering for Controlling Frozen LLMs" (Belitsky et al., 2025,
arXiv:2507.08799), adapted to steer the shared latent working-memory KV cache
that LatentMAS passes between agents before the Judger decodes.
"""

from .steering import CacheSteering

__all__ = ["CacheSteering"]
