"""
Top-level multi-agent OpenMythos model.

`MultiAgentMythos` is the user-facing entry point for the native-multi-agent
architecture. It wraps Prelude → MultiAgentRecurrentBlock → Coda exactly like
the single-agent `OpenMythos`, but the recurrent block runs N agents in
parallel with a Meow discrete broadcast bus between them.

The class inherits from `OpenMythos` and swaps only the recurrent block and
the forward signature (to optionally return the per-pass info dict with the
VQ loss). Everything else — Prelude, Coda, RoPE, MLA caching, head weight
tying with the embedding — is unchanged. `main.py` is not modified.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from open_mythos.agents import MultiAgentConfig, MultiAgentRecurrentBlock
from open_mythos.main import OpenMythos
from open_mythos.meow import MeowTrace, VectorQuantizer


class MultiAgentMythos(OpenMythos):
    """
    Multi-agent Recurrent-Depth Transformer language model.

    Structure:

        Input tokens
             ↓
        [Prelude]                      — from OpenMythos, unchanged
             ↓
        [MultiAgentRecurrentBlock]     — N agents, shared body, Meow bus
             ↑________ × T ________↓
             ↓
        [Coda]                         — from OpenMythos, unchanged
             ↓
        Output logits

    The Coda sees the mean of the N agents' ACT-weighted hidden states, so
    downstream code (heads, loss functions, tokenizers) needs no changes
    relative to a single-agent model.

    Additional forward return value:
        When `return_info=True`, `forward` returns `(logits, info)`. The
        `info` dict contains `vq_loss` (scalar tensor to be scaled by
        `cfg.vq_loss_weight` and added to the task loss) and `n_steps`
        (integer loop steps executed before ACT early exit). Trainer code
        must pass `return_info=True`.
    """

    def __init__(self, cfg: MultiAgentConfig):
        """
        Args:
            cfg -- MultiAgentConfig; must be a `MultiAgentConfig` instance
                   (not a bare `MythosConfig`) because `n_agents` and the
                   `meow_*` fields are required to build the recurrent block.
        """
        # Build the single-agent skeleton first. This creates and initializes
        # every module (embedding, prelude, single-agent recurrent, coda, head),
        # sets up the RoPE buffers, and ties the head weight to the embedding.
        super().__init__(cfg)

        # Swap the single-agent recurrent for the multi-agent one. Everything
        # else the parent constructed is still valid: Prelude, Coda, embedding,
        # head (with weight tying), RoPE caches, causal-mask builder.
        self.recurrent = MultiAgentRecurrentBlock(cfg)

        # Re-initialize parameters inside the new module with the project's
        # N(0, 0.02) convention, then restore the VQ codebook's specific
        # uniform(-1/K, 1/K) init. Touching only the replaced subtree keeps
        # the rest of the model (including the embed/head tying) intact.
        for m in self.recurrent.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.recurrent.modules():
            if isinstance(m, VectorQuantizer):
                K = m.num_embeddings
                m.embedding.weight.data.uniform_(-1 / K, 1 / K)
                # EMA buffers reset so the codebook starts from the fresh init
                # and is not contaminated by N(0, 0.02) re-initialization.
                m.ema_cluster_size.zero_()
                m.ema_w.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        trace: Optional[MeowTrace] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Forward pass through Prelude → MultiAgentRecurrentBlock → Coda.

        Args:
            input_ids   -- token indices of shape (B, T)
            n_loops     -- recurrent loop depth; defaults to `cfg.max_loop_iters`.
                           Increase at inference for deeper reasoning via depth
                           extrapolation.
            kv_cache    -- dict mutated in place for autoregressive decode.
                           Reuse across decode steps. Per-agent keys are used,
                           so every N × T entries are populated over the loop.
            trace       -- optional MeowTrace to record every broadcast for
                           observability. Pass None in hot-path inference.
            return_info -- if True, also return the per-pass info dict containing
                           `vq_loss` (scalar tensor) and `n_steps`. Training
                           code needs this; pure inference does not.

        Returns:
            logits -- (B, T, vocab_size) tensor
            info   -- (only if `return_info=True`) dict with vq_loss and n_steps
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[:T]
        mask = self._causal_mask(T, device) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x
        x, info = self.recurrent(
            x, e, freqs_cis, mask, n_loops, kv_cache, trace=trace
        )

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        logits = self.head(self.norm(x))

        if return_info:
            return logits, info
        return logits

    @torch.no_grad()
    def trace_forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
    ) -> Tuple[torch.Tensor, MeowTrace]:
        """
        Convenience wrapper: run one forward pass and return a populated MeowTrace.

        Args:
            input_ids -- (B, T) prompt tokens
            n_loops   -- recurrent loop depth, or None to use config default

        Returns:
            logits -- (B, T, vocab_size)
            trace  -- MeowTrace with one entry per (loop_step, agent)

        Use this at debugging / inference time to inspect what each agent
        "said" across the loop. Not intended for the training hot path — it
        retains every broadcast tensor in memory.
        """
        trace = MeowTrace()
        logits, _info = self.forward(
            input_ids, n_loops=n_loops, trace=trace, return_info=True
        )
        return logits, trace
