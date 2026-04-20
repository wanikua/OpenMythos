"""
MeowBus — discrete broadcast bus for multi-agent recurrent streams.

At each recurrent loop step, every agent emits K discrete codebook symbols per
token position (encode → VQ → decode). Peers then aggregate received messages
and inject them into their next hidden-state update. The update rule becomes:

    h_i_{t+1} = A · h_i_t + B · e_i + Transformer(h_i_t, e_i) + G_i · aggregate(m_{≠i,t})

where:
    A           diagonal, ρ(A) < 1 (guaranteed by LTIInjection in main.py)
    e_i         frozen Prelude encoding for agent i's token stream
    m_j,t       agent j's broadcast message at step t (output of this bus)
    G_i         per-agent learnable gate; initialized to 0 (warmup)

Stability preservation (LTI with bounded external drive):
    Decoder weights are bounded at any training step, and VQ output is bounded
    by max_k ‖codebook[k]‖. Hence ‖m_j,t‖ ≤ M_t (a finite constant depending
    only on current Bus parameters, not on any h_i). Mean aggregation keeps
    ‖aggregate(m_{≠i,t})‖ ≤ M_t (triangle inequality + averaging). Because
    ρ(A) < 1 and the extra term is bounded and independent of h_i, the
    recurrence stays bounded. Sum aggregation would break this — do not use.

Gradient stability:
    VectorQuantizer uses a straight-through estimator (quantized = inputs +
    (quantized - inputs).detach()), so the encoder receives identity gradients
    from downstream. The codebook itself is updated via EMA, not SGD, keeping
    the discrete vocabulary stable across scales. Combined with the existing
    grad-clip (max_norm=1.0) in the training script, no extra regularization
    is needed on the Bus path.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .codebook import VectorQuantizer


class MeowBus(nn.Module):
    """
    Per-position discrete broadcast bus shared by all agents.

    One bus instance is shared across every agent in the multi-agent block.
    This is intentional: the codebook is the *shared language*, analogous to
    a natural-language vocabulary all humans reuse. Per-agent encoders would
    fragment the protocol and defeat emergent-communication training.

    For a hidden state h of shape (B, T, D), the bus produces:
        messages  -- (B, T, D), continuous tensor reconstructed from quantized
                     codebook entries (used for injection into peer states)
        indices   -- (B, T, K), long tensor of selected codebook symbols
                     per token position (used for audit / logging only)
        vq_info   -- dict with the VQ commitment/codebook/total losses,
                     codebook perplexity, and usage_rate

    The VQ total loss is *not* automatically added to any graph here. The
    consumer (multi-agent block or training loop) is responsible for summing
    `vq_info["loss"]` across timesteps and agents, scaling it, and adding
    it to the task loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int = 512,
        codebook_dim: int = 128,
        msg_len: int = 4,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        """
        Args:
            dim             -- model hidden dimension D; encoder input & decoder output
            codebook_size   -- number of discrete symbols K in the codebook
            codebook_dim    -- dimension cdim of each codebook entry (keep small,
                               e.g. 128, to force a tight bottleneck)
            msg_len         -- number of codebook picks per token position (K_msg);
                               effective message is msg_len * log2(codebook_size) bits
                               per token position per agent per loop step
            commitment_cost -- VQ commitment loss weight (β in VQ-VAE paper)
            ema_decay       -- EMA decay for codebook updates; 0.99 is standard
        """
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.msg_len = msg_len

        # Encoder projects each (B, T, D) token into msg_len * codebook_dim features,
        # reshaped into msg_len chunks each quantized independently. This gives a
        # fixed-length, variable-semantic message per position.
        self.encoder = nn.Linear(dim, msg_len * codebook_dim)

        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
        )

        # Decoder projects msg_len * codebook_dim back to model hidden dim. Shared
        # across agents so that every agent decodes the same symbol to the same
        # vector — prerequisite for a common language to emerge.
        self.decoder = nn.Linear(msg_len * codebook_dim, dim)

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Encode an agent's hidden state into a discrete broadcast message
        and decode it back into the model's hidden space.

        Args:
            h -- agent hidden state, shape (B, T, D)

        Returns:
            message -- reconstructed message tensor, shape (B, T, D). This is
                       the tensor peers will aggregate and inject into their
                       own updates.
            indices -- codebook symbol indices chosen for each (b, t, k),
                       shape (B, T, msg_len); dtype long. For audit/logging.
            vq_info -- dict from VectorQuantizer.forward with the keys:
                       loss, commitment_loss, codebook_loss, perplexity,
                       usage_rate, encoding_indices (raw flat indices).
        """
        B, T, D = h.shape
        K = self.msg_len
        C = self.codebook_dim

        encoded = self.encoder(h)  # (B, T, K * C)
        flat = encoded.view(B * T * K, C)
        quantized_flat, indices_flat, vq_info = self.quantizer(flat)

        quantized = quantized_flat.view(B, T, K * C)
        message = self.decoder(quantized)  # (B, T, D)

        indices = indices_flat.view(B, T, K)
        return message, indices, vq_info

    @staticmethod
    def aggregate(messages: List[torch.Tensor], agent_idx: int) -> torch.Tensor:
        """
        Compute the received message for one agent as the mean of peer messages.

        Mean aggregation (not sum) is mandatory for stability: the bound on
        ‖aggregate(·)‖ matches the bound on any individual message, keeping
        the perturbation to the LTI recurrence bounded independent of N.

        Args:
            messages  -- list of per-agent messages, each shape (B, T, D).
                         Length N = total number of agents.
            agent_idx -- index of the receiving agent; this agent's own
                         message is excluded from the aggregation.

        Returns:
            received -- aggregated peer message for the receiver, shape (B, T, D).
                        Zeros when N == 1 (no peers to receive from).
        """
        if len(messages) <= 1:
            return torch.zeros_like(messages[0])

        peer_messages = [m for i, m in enumerate(messages) if i != agent_idx]
        # torch.stack + mean is cheaper than a python reduce and keeps autograd clean.
        return torch.stack(peer_messages, dim=0).mean(dim=0)

    @torch.no_grad()
    def codebook_norm_bound(self) -> torch.Tensor:
        """
        Return the current max codebook-entry L2 norm.

        Useful for runtime stability monitoring: a sudden jump in this bound
        during training correlates with the bus drive term gaining disproportionate
        influence over the LTI recurrence. Watch it alongside ρ(A) in logs.
        """
        return self.quantizer.embedding.weight.norm(dim=1).max()


class BusGate(nn.Module):
    """
    Per-agent learnable gate over the aggregated bus message.

    Initialized to zero so that at init a multi-agent model behaves exactly
    like N independent single-agent models. Training opens the gate only when
    inter-agent communication measurably helps the task loss. This warmup
    removes the usual "early training instability" phase of multi-agent
    systems where random noise from peers drowns out useful signal.

    The gate is *per-agent*, not per-channel, so the receiving dynamics for
    all D channels scale together. A per-channel gate would interact with
    the diagonal LTI update A in ways that could break the spectral bound;
    a scalar gate is just a bounded multiplier and preserves it.
    """

    def __init__(self, n_agents: int):
        """
        Args:
            n_agents -- number of agents in the multi-agent block
        """
        super().__init__()
        self.n_agents = n_agents
        self.gate = nn.Parameter(torch.zeros(n_agents))

    def forward(self, message: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """
        Scale an aggregated peer message by agent `agent_idx`'s learned gate.

        Args:
            message   -- aggregated message of shape (B, T, D)
            agent_idx -- receiving agent's index

        Returns:
            Gated message of shape (B, T, D)
        """
        return self.gate[agent_idx] * message

    def set_gate(self, value: float) -> None:
        """
        Force all gates to a fixed value (e.g., during warmup curricula or ablations).

        Not called during normal training. Use for:
        - evaluation of "no-communication" ablation (set to 0)
        - "full-communication" ablation (set to 1)
        - unit tests
        """
        with torch.no_grad():
            self.gate.fill_(value)
