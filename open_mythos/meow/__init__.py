"""
Meow protocol — discrete inter-agent communication layer for OpenMythos.

Upstream project: https://github.com/wanikua/meow (MIT license)
Vendored components: `VectorQuantizer` (see codebook.py for vendoring rationale).

Public API:
    VectorQuantizer         -- vendored VQ-VAE primitive
    MeowBus                 -- per-position discrete broadcast bus
    BusGate                 -- per-agent learnable gate over received messages
    MeowTrace               -- structured recording of every broadcast in a forward pass
    MeowAuditor             -- codebook → token gloss for post-hoc interpretability
    snapshot_codebook_stats -- one-shot diagnostic dict for logging
"""

from open_mythos.meow.audit import (
    MeowAuditor,
    MeowTrace,
    snapshot_codebook_stats,
)
from open_mythos.meow.bus import BusGate, MeowBus
from open_mythos.meow.codebook import VectorQuantizer

__all__ = [
    "VectorQuantizer",
    "MeowBus",
    "BusGate",
    "MeowTrace",
    "MeowAuditor",
    "snapshot_codebook_stats",
]
