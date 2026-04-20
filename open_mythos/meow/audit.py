"""
Observability tooling for the Meow discrete broadcast bus.

Every inter-agent message is a tuple of codebook indices — a fully inspectable
discrete trace of what each agent "said" to the others at every recurrent step.
This module turns those raw indices into human-readable reports, which is the
part of the architecture that decisively beats Claude-Code-harness-style
orchestration: in harness, you only see the final tool call or text output, so
debugging a multi-agent plan means reconstructing intent from externally visible
side effects. Here, the protocol itself is the log.

Two complementary views are supported:

- Trace view (`MeowTrace.to_rows`): for one forward pass, produce a table
  (agent, loop_step, token_pos, symbol_idx_0, ..., symbol_idx_{K-1}) so you
  can see every message each agent emitted for every token.
- Vocabulary view (`MeowAuditor.nearest_tokens`): probe the codebook against
  the model's token embeddings to find the natural-language tokens that sit
  closest to each codebook entry. This is a *lens*, not a decoder — the
  codebook lives in its own learned space, but nearest-neighbor tokens give
  a rough gloss for what a symbol "means" at a given training checkpoint.

Neither view does disk I/O. Callers are expected to hand the resulting rows /
lists to their own logger, TensorBoard, or W&B run.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MeowTrace:
    """
    Recording of every bus message emitted during one forward pass.

    Fields are nested lists indexed by [loop_step][agent]. Each list entry is
    a LongTensor of shape (B, T, msg_len) with the codebook indices chosen
    at that step by that agent. `perplexities` and `usage_rates` give
    per-step scalars for plotting during training runs.

    All tensors are kept on the device they arrived on; the caller is
    responsible for `.cpu()` before serialization if needed.
    """

    indices: List[List[Optional[torch.Tensor]]] = field(default_factory=list)
    perplexities: List[List[float]] = field(default_factory=list)
    usage_rates: List[List[float]] = field(default_factory=list)

    def record(
        self,
        loop_step: int,
        agent_idx: int,
        indices: torch.Tensor,
        perplexity: torch.Tensor,
        usage_rate: torch.Tensor,
    ) -> None:
        """
        Append one agent's message at one loop step.

        Args:
            loop_step  -- current recurrent iteration t
            agent_idx  -- 0-based agent index i
            indices    -- LongTensor (B, T, msg_len) of codebook picks
            perplexity -- VQ perplexity scalar from this step
            usage_rate -- fraction of codebook hit in this step
        """
        while len(self.indices) <= loop_step:
            self.indices.append([])
            self.perplexities.append([])
            self.usage_rates.append([])
        while len(self.indices[loop_step]) <= agent_idx:
            self.indices[loop_step].append(None)
            self.perplexities[loop_step].append(0.0)
            self.usage_rates[loop_step].append(0.0)

        self.indices[loop_step][agent_idx] = indices.detach()
        self.perplexities[loop_step][agent_idx] = float(perplexity.detach())
        self.usage_rates[loop_step][agent_idx] = float(usage_rate.detach())

    def to_rows(self) -> List[Tuple[int, int, int, int, List[int]]]:
        """
        Flatten the trace into one row per (loop, agent, batch_pos, token_pos).

        Returns:
            List of tuples (loop_step, agent, b, t, [sym_0, ..., sym_{K-1}])

        Used by loggers to push to W&B tables or pandas frames without
        forcing a pandas dependency into this module.
        """
        rows: List[Tuple[int, int, int, int, List[int]]] = []
        for loop_step, per_agent in enumerate(self.indices):
            for agent_idx, tensor in enumerate(per_agent):
                if tensor is None:
                    continue
                B, T, _K = tensor.shape
                tensor_cpu = tensor.to("cpu").tolist()
                for b in range(B):
                    for t in range(T):
                        syms = tensor_cpu[b][t]
                        rows.append((loop_step, agent_idx, b, t, syms))
        return rows

    def summary(self) -> str:
        """
        One-line text summary for stdout-style logging during training.

        Returns:
            Formatted string with averaged perplexity and usage rate
            across all recorded (loop, agent) pairs. Useful for catching
            codebook collapse (perplexity → 1) or under-use (rate → 0).
        """
        flat_p = [p for step in self.perplexities for p in step]
        flat_u = [u for step in self.usage_rates for u in step]
        if not flat_p:
            return "MeowTrace(empty)"
        avg_p = sum(flat_p) / len(flat_p)
        avg_u = sum(flat_u) / len(flat_u)
        n_loops = len(self.indices)
        n_agents = len(self.indices[0]) if self.indices else 0
        return (
            f"MeowTrace(loops={n_loops}, agents={n_agents}, "
            f"avg_perplexity={avg_p:.2f}, avg_usage_rate={avg_u:.3f})"
        )


class MeowAuditor:
    """
    Probe a trained MeowBus codebook against a token embedding table.

    This is a post-hoc interpretability lens. A codebook entry does not have
    an a priori "meaning"; it has whatever meaning emerges from training.
    For each codebook symbol we find the token-embedding it most closely
    matches in cosine distance. The resulting (symbol, token) pairs give a
    human-readable label for the emergent vocabulary at a given checkpoint.

    Labels are *not stable across checkpoints*. Always re-run this probe
    after the model has been updated.
    """

    def __init__(
        self,
        codebook_weight: torch.Tensor,
        token_embed_weight: torch.Tensor,
        codebook_dim: int,
    ):
        """
        Args:
            codebook_weight    -- bus.quantizer.embedding.weight, shape (K, cdim)
            token_embed_weight -- model.embed.weight, shape (vocab_size, dim)
            codebook_dim       -- cdim of the codebook

        The two tables live in different spaces (cdim vs dim). Callers should
        supply a projection via `set_projection` (typically derived from the
        bus decoder weights) before calling `nearest_tokens` / `gloss`.
        """
        self.codebook = codebook_weight.detach()
        self.token_embed = token_embed_weight.detach()
        self.codebook_dim = codebook_dim
        self.projection: Optional[torch.Tensor] = None  # (cdim, dim) if set

    def set_projection(self, projection: torch.Tensor) -> None:
        """
        Supply a (cdim → dim) projection to align the codebook with tokens.

        In practice the decoder's first K*cdim → dim weight block does this
        job. Callers can pass the decoder's weight slice here, or train a
        probe offline.

        Args:
            projection -- shape (cdim, dim) tensor
        """
        expected = (self.codebook_dim, self.token_embed.shape[1])
        assert projection.shape == expected, (
            f"Expected {expected}, got {tuple(projection.shape)}"
        )
        self.projection = projection.detach()

    @torch.no_grad()
    def nearest_tokens(self, k: int = 3) -> List[List[int]]:
        """
        For each codebook symbol, find its k nearest token-vocabulary indices
        by cosine similarity (after optional projection into token space).

        Args:
            k -- how many nearest token ids to return per symbol

        Returns:
            List of length num_embeddings; each element is a list of length k
            containing vocabulary ids sorted by decreasing similarity.
        """
        codebook_in_token_space = (
            self.codebook @ self.projection
            if self.projection is not None
            else self.codebook
        )
        if codebook_in_token_space.shape[1] != self.token_embed.shape[1]:
            raise ValueError(
                "Codebook dim does not match token embed dim; supply a projection "
                "via `set_projection` that maps codebook_dim -> model hidden dim."
            )

        cb_norm = codebook_in_token_space / (
            codebook_in_token_space.norm(dim=1, keepdim=True) + 1e-8
        )
        tok_norm = self.token_embed / (
            self.token_embed.norm(dim=1, keepdim=True) + 1e-8
        )
        sims = cb_norm @ tok_norm.T  # (num_embeddings, vocab_size)

        _, topk = sims.topk(k, dim=1)
        return topk.tolist()

    @torch.no_grad()
    def gloss(self, tokenizer, k: int = 3) -> List[str]:
        """
        Produce a human-readable label for each codebook symbol.

        Args:
            tokenizer -- any object with a `decode(ids: List[int]) -> str` method
            k         -- number of nearest tokens to include per gloss

        Returns:
            List of length num_embeddings with strings like
            "[sym 42] → 'the', 'a', 'an'"
        """
        nearest = self.nearest_tokens(k)
        out: List[str] = []
        for sym_idx, ids in enumerate(nearest):
            decoded = [tokenizer.decode([tid]) for tid in ids]
            pretty = ", ".join(repr(t) for t in decoded)
            out.append(f"[sym {sym_idx}] → {pretty}")
        return out


def snapshot_codebook_stats(bus: nn.Module) -> dict:
    """
    One-shot diagnostic of the bus codebook's current health.

    Args:
        bus -- a MeowBus module (duck-typed on `.quantizer.ema_cluster_size`,
               `.quantizer.embedding`, and `.quantizer.dead_threshold`)

    Returns:
        Dict with live counts, norm summary, and a "dead code" fraction.
        Intended for periodic logging rather than per-step use.
    """
    ema = bus.quantizer.ema_cluster_size.detach().float()
    weights = bus.quantizer.embedding.weight.detach()
    norms = weights.norm(dim=1)

    return {
        "codebook_size": int(weights.shape[0]),
        "live_codes": int((ema > bus.quantizer.dead_threshold).sum()),
        "dead_codes": int((ema <= bus.quantizer.dead_threshold).sum()),
        "dead_fraction": float((ema <= bus.quantizer.dead_threshold).float().mean()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }
