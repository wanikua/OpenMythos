"""
Multi-agent recurrent block for OpenMythos.

This module extends the single-agent Recurrent-Depth Transformer into a
*native* multi-agent architecture — "native" in the strictest sense: the
agents run inside one forward pass of one set of weights, communicating via
a discrete protocol on every recurrent loop iteration. There is no external
orchestrator, no separate model per agent, and no natural-language exchange
between steps. The "multi-agent" property is a property of the forward pass,
not of the runtime above it.

Why this beats a Claude-Code-style harness
------------------------------------------
A harness stitches together N model calls plus text-level tool use. Every
inter-agent hop pays a decoder cost, a tokenizer round-trip, and a context
concatenation. Latency compounds linearly with hops; the "shared state" is
whatever each call can cram back into a prompt; and the only artifact you
can inspect is the natural-language surface the agents produce.

Here, agents are N personas grafted onto a shared recurrent body:
    - One TransformerBlock, one set of MoE experts, one MLA attention.
    - Per-agent LoRA delta indexed by (loop_step, agent_id) into a single
      depth-wise-LoRA table (Bae et al., 2024), so specialization is cheap.
    - Per-agent scalar identity embedding added once to the initial hidden
      state; thereafter, agent-specific behavior is driven entirely by the
      LoRA deltas and the discrete bus traffic.
    - Meow discrete codebook as the inter-agent wire protocol — each agent
      broadcasts K codebook symbols per token position per loop step, and
      receives the mean of peers' broadcasts on the next update.
    - Per-agent ACT halting: every agent decides independently when each
      token position has converged.

The update rule per agent `i` at step `t` is:

    m_{j,t}          = Bus.decode(VQ(Bus.encode(h_{j,t})))         ∀ j
    r_{i,t}          = mean_{j ≠ i} m_{j,t}                         (bounded)
    ĝ_{i,t}          = BusGate_i · r_{i,t}                           (gate init = 0)
    h_loop_{i,t}     = loop_index_embedding(h_{i,t}, t)
    combined_{i,t}   = RMSNorm(h_loop_{i,t} + e)
    trans_{i,t}      = TransformerBlock(combined_{i,t}) + LoRA(trans_{i,t}, t·N+i)
    h_{i,t+1}        = A · h_{i,t} + B · e + trans_{i,t} + ĝ_{i,t}

Stability
---------
`A` is the diagonal LTI matrix produced by `LTIInjection.get_A`; by
construction ρ(A) < 1 regardless of training dynamics. The new term ĝ_{i,t}
adds a *bounded external drive* to the recurrence, because:
    - The codebook decoder is a fixed linear map at any training step.
    - Codebook entries have bounded norm max_k ‖codebook[k]‖.
    - Mean-aggregation does not grow with N.
    - BusGate is a learned scalar (∈ ℝ, bounded for any finite training step).
Therefore the closed-loop system remains bounded and the ρ(A) < 1 guarantee
is preserved without additional constraints on BusGate. Sum-aggregation
would scale with N and break this — `MeowBus.aggregate` uses mean, not sum.

Output aggregation
------------------
The final `h_out` is the mean over agents. Same boundedness argument. The
Coda transformer blocks then operate on the averaged representation as if
it came from a single agent. This matches the "one brain, many personas"
interpretation: agents cooperate inside the body, but the body emits one
answer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from open_mythos.main import (
    ACTHalting,
    LoRAAdapter,
    LTIInjection,
    MythosConfig,
    RMSNorm,
    TransformerBlock,
    loop_index_embedding,
)
from open_mythos.meow import BusGate, MeowBus, MeowTrace


@dataclass
class MultiAgentConfig(MythosConfig):
    """
    Configuration for a multi-agent OpenMythos model.

    Inherits every field from `MythosConfig` (including MLA params, MoE
    params, loop budget, RoPE base, dropout, etc.) and adds fields
    governing the agent count and the Meow bus.

    Multi-agent fields:
        n_agents              -- number of concurrent reasoning agents sharing
                                 the recurrent body. For N = 1 the class
                                 reduces to the single-agent architecture
                                 with an extra identity-zero perturbation.
        meow_codebook_size    -- number of discrete symbols K in the codebook.
                                 Larger K = richer vocabulary, slower VQ.
                                 512 is a reasonable default per the Meow paper.
        meow_codebook_dim     -- per-symbol embedding dimension cdim inside
                                 the codebook. Kept small (e.g., 128) to force
                                 a tight information bottleneck per symbol.
        meow_msg_len          -- number of codebook picks per token position
                                 per agent per loop step. Effective per-token
                                 message payload is msg_len · log2(codebook_size)
                                 bits. 4 is a conservative default.
        meow_commitment_cost  -- VQ commitment loss weight (β). 0.25 is standard.
        meow_ema_decay        -- EMA decay for codebook updates. 0.99 is standard.
        vq_loss_weight        -- scalar applied to the summed VQ loss before
                                 adding to the task loss in the trainer.
    """

    n_agents: int = 4
    meow_codebook_size: int = 512
    meow_codebook_dim: int = 128
    meow_msg_len: int = 4
    meow_commitment_cost: float = 0.25
    meow_ema_decay: float = 0.99
    vq_loss_weight: float = 0.1


class MultiAgentRecurrentBlock(nn.Module):
    """
    Recurrent block running N agents in parallel inside one forward pass.

    All agents share one TransformerBlock, one LTI update, one ACT halter,
    and one RMSNorm. Specialization comes from:
        - `agent_embed`: scalar identity vector added to each agent's
          initial hidden state (only at the start of the loop).
        - `lora`: one LoRAAdapter with `max_loops × n_agents` scale vectors,
          indexed by `t · N + i`. Adds (loop_step, agent) — specific deltas
          to the transformer output.
        - Per-agent halting state (not a parameter; a runtime tensor).

    Agents communicate via `bus` (a shared MeowBus) and `bus_gate` (a per-
    agent learnable scalar gate, initialized to zero so that at checkpoint-0
    multi-agent is identical to N independent single-agent runs).
    """

    def __init__(self, cfg: MultiAgentConfig):
        """
        Args:
            cfg -- MultiAgentConfig with multi-agent + Meow fields populated.
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        # LoRA scale table is (max_loops × n_agents)-wide so each (t, i) pair
        # has a distinct per-loop per-agent scale vector. One shared low-rank
        # down/up projection; only the scale vector differs per (t, i).
        self.lora = LoRAAdapter(
            cfg.dim, cfg.lora_rank, cfg.max_loop_iters * cfg.n_agents
        )
        self.norm = RMSNorm(cfg.dim)
        self.agent_embed = nn.Embedding(cfg.n_agents, cfg.dim)
        # Initialize agent embedding small so agent identities start as
        # perturbations on the prelude output, not dominant signals. The
        # LoRA deltas will grow them during training as needed.
        nn.init.normal_(self.agent_embed.weight, std=0.02)

        self.bus = MeowBus(
            dim=cfg.dim,
            codebook_size=cfg.meow_codebook_size,
            codebook_dim=cfg.meow_codebook_dim,
            msg_len=cfg.meow_msg_len,
            commitment_cost=cfg.meow_commitment_cost,
            ema_decay=cfg.meow_ema_decay,
        )
        self.bus_gate = BusGate(cfg.n_agents)

        self.loop_dim = cfg.dim // 8

    def _init_agent_states(self, h: torch.Tensor) -> List[torch.Tensor]:
        """
        Create N per-agent initial hidden states from the shared Prelude output.

        Args:
            h -- Prelude output tensor of shape (B, T, D)

        Returns:
            List of length `n_agents`; each entry is a (B, T, D) tensor
            equal to h plus that agent's identity embedding.
        """
        N = self.cfg.n_agents
        agent_ids = torch.arange(N, device=h.device)
        agent_id_emb = self.agent_embed(agent_ids)  # (N, D)
        # Each agent starts from `h` with its own small identity perturbation;
        # the Bus gate starting at 0 means agents do not yet influence one
        # another, so divergence early in training is driven entirely by
        # agent_id_emb + per-(t, i) LoRA.
        return [h + agent_id_emb[i].unsqueeze(0).unsqueeze(0) for i in range(N)]

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        trace: Optional[MeowTrace] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run the multi-agent recurrent loop and return the averaged output.

        Args:
            h         -- Prelude output, shape (B, T, D). Same for all agents.
            e         -- Injection target (frozen encoded input), shape (B, T, D).
                         Shared across agents intentionally; identity separation
                         is via agent_embed + LoRA, not via the injection target.
            freqs_cis -- Precomputed RoPE frequencies.
            mask      -- Additive causal mask, or None for single-token decode.
            n_loops   -- Loop depth; defaults to `cfg.max_loop_iters`. Can be
                         raised at inference for depth-extrapolation reasoning.
            kv_cache  -- Dict mutated in place for autoregressive decode.
                         Keys used: `f"multiagent_{i}_loop_{t}"` per (agent, loop).
            trace     -- Optional MeowTrace; if supplied, every broadcast is
                         recorded for observability. Pass None to skip.

        Returns:
            h_out -- Aggregated post-recurrent hidden state, shape (B, T, D).
                     Mean across all agents of their ACT-weighted sum.
            info  -- Dict with:
                        "vq_loss"   : scalar tensor, sum of all VQ losses
                                      emitted during this forward pass. The
                                      trainer should multiply by
                                      `cfg.vq_loss_weight` and add to task loss.
                        "n_steps"   : number of loop steps executed before
                                      early exit (useful for compute logging).
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape
        N = self.cfg.n_agents

        h_list = self._init_agent_states(h)

        halted = [
            torch.zeros(B, T, device=h.device, dtype=torch.bool) for _ in range(N)
        ]
        cum_p = [torch.zeros(B, T, device=h.device) for _ in range(N)]
        h_out = [torch.zeros_like(h) for _ in range(N)]

        vq_losses: List[torch.Tensor] = []
        steps_executed = 0

        for t in range(n_loops):
            steps_executed = t + 1

            # ----- Broadcast phase: every agent emits a message for peers -----
            messages: List[torch.Tensor] = []
            for i in range(N):
                msg, indices, vq_info = self.bus(h_list[i])
                messages.append(msg)
                vq_losses.append(vq_info["loss"])
                if trace is not None:
                    trace.record(
                        loop_step=t,
                        agent_idx=i,
                        indices=indices,
                        perplexity=vq_info["perplexity"],
                        usage_rate=vq_info["usage_rate"],
                    )

            # ----- Update phase: each agent consumes aggregated peer messages -----
            for i in range(N):
                h_i = h_list[i]

                received = MeowBus.aggregate(messages, i)
                gated_recv = self.bus_gate(received, i)

                h_loop = loop_index_embedding(h_i, t, self.loop_dim)
                combined = self.norm(h_loop + e)
                cache_key = f"multiagent_{i}_loop_{t}"
                trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
                lora_idx = t * N + i
                trans_out = trans_out + self.lora(trans_out, lora_idx) + gated_recv

                h_i_new = self.injection(h_i, e, trans_out)

                # ACT halting — independent per agent
                p = self.act(h_i_new)
                still_running = ~halted[i]
                remainder = (1.0 - cum_p[i]).clamp(min=0)
                weight = torch.where(
                    cum_p[i] + p >= self.cfg.act_threshold,
                    remainder,
                    p,
                )
                h_out[i] = h_out[i] + weight.unsqueeze(-1) * h_i_new
                cum_p[i] = cum_p[i] + p * still_running.float()
                halted[i] = halted[i] | (cum_p[i] >= self.cfg.act_threshold)

                h_list[i] = h_i_new

            # Early exit only when every agent's every position has converged.
            if all(h_agent.all() for h_agent in halted):
                break

        # Aggregate output: mean across agents. Preserves boundedness (see module
        # docstring) and keeps the Coda's interface identical to single-agent.
        h_final = torch.stack(h_out, dim=0).mean(dim=0)

        vq_loss = (
            torch.stack(vq_losses).sum()
            if vq_losses
            else torch.zeros((), device=h.device)
        )
        info: Dict[str, torch.Tensor] = {
            "vq_loss": vq_loss,
            "n_steps": torch.tensor(steps_executed, device=h.device),
        }
        return h_final, info

    @torch.no_grad()
    def spectral_radius(self) -> torch.Tensor:
        """
        Report the current spectral radius ρ(A) of the LTI injection.

        Because A is diagonal and produced by `exp(-exp(...))`, every entry is
        in (0, 1) and the spectral radius equals the largest diagonal entry.
        This is the stability invariant; train-time logging of this value is
        the single cheapest way to catch a regression on the guarantee.

        Returns:
            Scalar tensor in (0, 1).
        """
        A = self.injection.get_A()
        return A.max()
