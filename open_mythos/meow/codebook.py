"""
Vendored VectorQuantizer from the Meow Protocol.

Source: https://github.com/wanikua/meow (meow/codebook.py)
License: MIT, Copyright (c) 2026 Meow Protocol Contributors

Rationale for vendoring (not pip dependency):
- This is the single primitive OpenMythos needs; the rest of `meow` (encoders,
  save/load, dataloaders, CLI) is unused in the recurrent-depth setting.
- Vendoring keeps OpenMythos installable without pulling Meow's inference stack.
- The upstream file may receive training-time tweaks (e.g., EMA variants) that
  are orthogonal to how we use it here; a frozen copy isolates that drift.

Local modifications:
- None. The class is imported verbatim. If upstream changes merit adoption,
  refresh this file and re-run tests/test_multi_agent.py::test_vq_stability.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.

    Maps continuous embeddings to discrete codebook symbols via nearest-neighbor
    lookup, with EMA codebook updates and dead-code replacement. The forward
    pass uses a straight-through estimator so that gradients flow back to the
    encoder unchanged while the quantized tensor is the one used downstream.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 768,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        dead_threshold: int = 2,
    ):
        """
        Args:
            num_embeddings  -- number of discrete symbols in the codebook
            embedding_dim   -- dimension of each codebook entry
            commitment_cost -- weight on the commitment loss term (β in VQ-VAE paper)
            decay           -- EMA decay for codebook update; 0.99 is standard
            epsilon         -- numerical-stability constant for Laplace smoothing
            dead_threshold  -- entries with cumulative EMA usage below this
                               are resampled from the current batch to recover
                               collapsed codewords
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize input embeddings to their nearest codebook entries.

        Args:
            inputs -- continuous embeddings of shape (N, embedding_dim)

        Returns:
            quantized -- same shape as inputs; each row replaced by its nearest
                         codebook entry, with a straight-through gradient path
            indices   -- chosen codebook indices of shape (N,)
            info      -- dict with commitment_loss, codebook_loss, total loss,
                         perplexity (diversity of usage in this batch), and
                         usage_rate (fraction of codebook entries hit)
        """
        distances = (
            torch.sum(inputs**2, dim=1, keepdim=True)
            - 2 * torch.matmul(inputs, self.embedding.weight.T)
            + torch.sum(self.embedding.weight**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        quantized = torch.matmul(encodings, self.embedding.weight)

        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        if self.training:
            batch_cluster_counts = encodings.sum(dim=0)
            self.ema_cluster_size = (
                self.ema_cluster_size * self.decay
                + (1 - self.decay) * batch_cluster_counts
            )
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * torch.matmul(
                encodings.T, inputs
            )

            n = self.ema_cluster_size.sum()
            normalized_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            self.embedding.weight.data = self.ema_w / normalized_cluster_size.unsqueeze(1)

            dead_mask = self.ema_cluster_size < self.dead_threshold
            n_dead = dead_mask.sum().item()
            if n_dead > 0 and inputs.shape[0] > 0:
                replace_idx = torch.randint(
                    0, inputs.shape[0], (n_dead,), device=inputs.device
                )
                noise = torch.randn_like(inputs[replace_idx]) * 0.01
                self.embedding.weight.data[dead_mask] = (
                    inputs[replace_idx].detach() + noise
                )
                self.ema_cluster_size[dead_mask] = 1.0
                self.ema_w[dead_mask] = self.embedding.weight.data[dead_mask]

        # Straight-through estimator: identity forward value, identity gradient to inputs
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage_rate = (encodings.sum(dim=0) > 0).float().mean()

        info = {
            "loss": loss,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "usage_rate": usage_rate,
            "encoding_indices": encoding_indices,
        }

        return quantized, encoding_indices, info
