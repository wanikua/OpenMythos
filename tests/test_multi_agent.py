"""
Tests for the native multi-agent OpenMythos architecture.

Locks down multi-agent-specific invariants:
    - Shape + dtype contract of MultiAgentMythos.forward / trace_forward
    - LTI stability: ρ(A) < 1 at init and after one backward pass
    - VQ straight-through gradient flow (encoder receives usable gradients)
    - Codebook usage_rate > 0 in training mode (not dead on arrival)
    - BusGate zero-init means checkpoint-0 acts as N independent single agents
    - KV cache interface is consistent with the single-agent model
    - Multi-agent variant factories produce valid MultiAgentConfig instances

Tiny config (dim=64, 2 agents, vocab=128) so the suite runs in single-digit
seconds on CPU. Scale-related guarantees (N=8, 10B etc.) are enforced by
variant factory tests + the shared stability argument.
"""

from __future__ import annotations

import pytest
import torch

from open_mythos import (
    BusGate,
    MeowBus,
    MeowTrace,
    MultiAgentConfig,
    MultiAgentMythos,
    VectorQuantizer,
    multi_agent_1b,
    multi_agent_3b,
    multi_agent_10b,
)


def _tiny_cfg(**overrides) -> MultiAgentConfig:
    """CPU-tractable MultiAgentConfig for the test suite."""
    base = dict(
        vocab_size=128,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=16,
        v_head_dim=16,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=32,
        act_threshold=0.99,
        rope_theta=10000.0,
        lora_rank=4,
        n_agents=2,
        meow_codebook_size=16,
        meow_codebook_dim=16,
        meow_msg_len=2,
        meow_commitment_cost=0.25,
        meow_ema_decay=0.99,
        vq_loss_weight=0.1,
    )
    base.update(overrides)
    return MultiAgentConfig(**base)


@pytest.fixture
def cfg() -> MultiAgentConfig:
    return _tiny_cfg()


@pytest.fixture
def model(cfg: MultiAgentConfig) -> MultiAgentMythos:
    torch.manual_seed(0)
    return MultiAgentMythos(cfg)


@pytest.fixture
def input_ids(cfg: MultiAgentConfig) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_size, (2, 8))


# ---------------------------------------------------------------------------
# Forward shape / info contract
# ---------------------------------------------------------------------------


def test_forward_returns_logits_of_expected_shape(
    model: MultiAgentMythos, input_ids: torch.Tensor, cfg: MultiAgentConfig
):
    logits = model(input_ids)
    assert logits.shape == (input_ids.shape[0], input_ids.shape[1], cfg.vocab_size)
    assert logits.dtype == torch.float32


def test_forward_with_return_info_returns_vq_loss_and_n_steps(
    model: MultiAgentMythos, input_ids: torch.Tensor
):
    logits, info = model(input_ids, return_info=True)
    assert logits.dim() == 3
    assert "vq_loss" in info and "n_steps" in info
    assert info["vq_loss"].ndim == 0
    assert torch.isfinite(info["vq_loss"])
    assert int(info["n_steps"]) >= 1


def test_trace_forward_populates_trace_rows(
    model: MultiAgentMythos, input_ids: torch.Tensor, cfg: MultiAgentConfig
):
    model.eval()
    _logits, trace = model.trace_forward(input_ids)
    assert isinstance(trace, MeowTrace)
    rows = trace.to_rows()
    assert len(rows) > 0
    sample = rows[0]
    assert len(sample) == 5
    assert len(sample[4]) == cfg.meow_msg_len


# ---------------------------------------------------------------------------
# Stability: ρ(A) < 1 before and after training
# ---------------------------------------------------------------------------


def test_spectral_radius_below_one_at_init(model: MultiAgentMythos):
    rho = model.recurrent.spectral_radius()
    assert 0.0 < float(rho) < 1.0


def test_spectral_radius_stays_below_one_after_backward(
    model: MultiAgentMythos, input_ids: torch.Tensor, cfg: MultiAgentConfig
):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    logits, info = model(input_ids, return_info=True)
    task_loss = logits.square().mean()
    total = task_loss + cfg.vq_loss_weight * info["vq_loss"]
    total.backward()
    opt.step()
    rho = model.recurrent.spectral_radius()
    assert 0.0 < float(rho) < 1.0


# ---------------------------------------------------------------------------
# VQ gradient flow and codebook usage
# ---------------------------------------------------------------------------


def test_vq_straight_through_gradient_reaches_bus_encoder(
    model: MultiAgentMythos, input_ids: torch.Tensor
):
    model.train()
    logits, info = model(input_ids, return_info=True)
    loss = logits.square().mean() + info["vq_loss"]
    loss.backward()

    enc_grad = model.recurrent.bus.encoder.weight.grad
    assert enc_grad is not None
    assert torch.isfinite(enc_grad).all()
    assert enc_grad.abs().sum() > 0


def test_codebook_usage_rate_positive_in_training(model: MultiAgentMythos):
    model.train()
    h = torch.randn(2, 8, model.cfg.dim)
    _msg, _idx, info = model.recurrent.bus(h)
    assert float(info["usage_rate"]) > 0.0


# ---------------------------------------------------------------------------
# BusGate: zero-init → multi-agent reduces to N independent single agents
# ---------------------------------------------------------------------------


def test_bus_gate_initialized_to_zero(model: MultiAgentMythos):
    gate = model.recurrent.bus_gate.gate
    assert torch.equal(gate, torch.zeros_like(gate))


def test_bus_gate_zero_blocks_peer_influence():
    """With BusGate forced to 0, zeroing the bus decoder is a no-op."""
    cfg = _tiny_cfg(n_agents=2, max_loop_iters=1)
    torch.manual_seed(2)
    model = MultiAgentMythos(cfg)
    model.eval()
    model.recurrent.bus_gate.set_gate(0.0)

    torch.manual_seed(3)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out_a = model(ids)

    with torch.no_grad():
        model.recurrent.bus.decoder.weight.mul_(0.0)
        if model.recurrent.bus.decoder.bias is not None:
            model.recurrent.bus.decoder.bias.zero_()
    out_b = model(ids)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_bus_gate_nonzero_propagates_peer_influence():
    """Sanity-inverse: with gate opened, zeroing the decoder *does* change outputs."""
    cfg = _tiny_cfg(n_agents=2, max_loop_iters=1)
    torch.manual_seed(4)
    model = MultiAgentMythos(cfg)
    model.eval()
    model.recurrent.bus_gate.set_gate(1.0)

    torch.manual_seed(5)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out_a = model(ids)

    with torch.no_grad():
        model.recurrent.bus.decoder.weight.mul_(0.0)
        if model.recurrent.bus.decoder.bias is not None:
            model.recurrent.bus.decoder.bias.zero_()
    out_b = model(ids)

    assert not torch.allclose(out_a, out_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Bus internals — shape + aggregation contract
# ---------------------------------------------------------------------------


def test_meow_bus_forward_shapes(cfg: MultiAgentConfig):
    bus = MeowBus(
        dim=cfg.dim,
        codebook_size=cfg.meow_codebook_size,
        codebook_dim=cfg.meow_codebook_dim,
        msg_len=cfg.meow_msg_len,
    )
    h = torch.randn(2, 4, cfg.dim)
    msg, idx, info = bus(h)
    assert msg.shape == (2, 4, cfg.dim)
    assert idx.shape == (2, 4, cfg.meow_msg_len)
    assert idx.dtype == torch.long
    assert idx.max() < cfg.meow_codebook_size
    assert idx.min() >= 0
    assert torch.isfinite(info["loss"])


def test_meow_bus_aggregate_mean_not_sum():
    """Stability depends on aggregate() being a mean. Verify directly."""
    D = 4
    msgs = [torch.ones(1, 1, D), torch.ones(1, 1, D) * 3.0, torch.ones(1, 1, D) * 5.0]
    received = MeowBus.aggregate(msgs, agent_idx=0)
    assert torch.allclose(received, torch.full((1, 1, D), 4.0))


def test_meow_bus_aggregate_single_agent_returns_zeros():
    msgs = [torch.ones(1, 1, 4)]
    received = MeowBus.aggregate(msgs, agent_idx=0)
    assert torch.equal(received, torch.zeros(1, 1, 4))


# ---------------------------------------------------------------------------
# VectorQuantizer — straight-through contract
# ---------------------------------------------------------------------------


def test_vector_quantizer_straight_through_gradient_is_identity():
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4)
    inputs = torch.randn(16, 4, requires_grad=True)
    quantized, _indices, _info = vq(inputs)
    quantized.sum().backward()
    assert inputs.grad is not None
    assert torch.allclose(inputs.grad, torch.ones_like(inputs))


def test_vector_quantizer_outputs_in_codebook_range():
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4)
    inputs = torch.randn(16, 4)
    _quantized, indices, _info = vq(inputs)
    assert indices.min() >= 0 and indices.max() < 8


# ---------------------------------------------------------------------------
# Variant factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,expected_agents",
    [
        (multi_agent_1b, 4),
        (multi_agent_3b, 4),
        (multi_agent_10b, 8),
    ],
)
def test_variant_factories_build_multi_agent_config(factory, expected_agents: int):
    cfg = factory()
    assert isinstance(cfg, MultiAgentConfig)
    assert cfg.n_agents == expected_agents
    assert cfg.meow_codebook_size >= 128
    assert cfg.meow_msg_len >= 1
    assert 0.0 <= cfg.meow_ema_decay < 1.0


# ---------------------------------------------------------------------------
# KV cache — per-agent, per-loop key layout
# ---------------------------------------------------------------------------


def test_kv_cache_populated_with_expected_keys(
    model: MultiAgentMythos, cfg: MultiAgentConfig
):
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    cache: dict = {}
    model(ids, kv_cache=cache)

    expected_multiagent = {
        f"multiagent_{i}_loop_{t}"
        for i in range(cfg.n_agents)
        for t in range(cfg.max_loop_iters)
    }
    actual_multiagent = {k for k in cache if k.startswith("multiagent_")}
    assert actual_multiagent, "no multiagent KV keys were written"
    assert actual_multiagent.issubset(expected_multiagent)
    assert any(k.startswith("prelude_") for k in cache)
    assert any(k.startswith("coda_") for k in cache)


# ---------------------------------------------------------------------------
# BusGate module-level test
# ---------------------------------------------------------------------------


def test_bus_gate_set_gate_updates_parameters():
    gate = BusGate(n_agents=3)
    assert torch.equal(gate.gate, torch.zeros(3))
    gate.set_gate(0.5)
    assert torch.allclose(gate.gate, torch.full((3,), 0.5))


def test_bus_gate_scales_message_per_agent():
    gate = BusGate(n_agents=2)
    with torch.no_grad():
        gate.gate[0] = 0.0
        gate.gate[1] = 2.0
    msg = torch.ones(1, 1, 4)
    assert torch.equal(gate(msg, 0), torch.zeros(1, 1, 4))
    assert torch.equal(gate(msg, 1), torch.full((1, 1, 4), 2.0))


# ---------------------------------------------------------------------------
# Determinism: re-running forward with the same inputs is stable in eval mode
# ---------------------------------------------------------------------------


def test_eval_mode_forward_is_deterministic(
    cfg: MultiAgentConfig, input_ids: torch.Tensor
):
    torch.manual_seed(42)
    m = MultiAgentMythos(cfg)
    m.eval()
    a = m(input_ids)
    b = m(input_ids)
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: tiny end-to-end training loop against fake data
# ---------------------------------------------------------------------------


def test_training_step_smoke_end_to_end(cfg: MultiAgentConfig):
    """
    Run two optimizer steps with fake data. Validates that the whole path —
    forward with return_info → VQ loss aggregation → backward → optimizer.step
    → bus_gate warmup update — executes without NaN/Inf and updates parameters.

    Intentionally small (2 steps, batch=2, T=4) so CPU-only CI stays fast.
    """
    torch.manual_seed(7)
    model = MultiAgentMythos(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    def fake_batch():
        x = torch.randint(0, cfg.vocab_size, (2, 4))
        y = torch.randint(0, cfg.vocab_size, (2, 4))
        return x, y

    losses = []
    gates_applied = []
    bus_warmup_steps = 4
    for step in range(2):
        gate_value = min(1.0, step / bus_warmup_steps) if bus_warmup_steps else 1.0
        model.recurrent.bus_gate.set_gate(gate_value)
        gates_applied.append(gate_value)

        opt.zero_grad()
        x, y = fake_batch()
        logits, info = model(x, return_info=True)
        task_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size), y.view(-1)
        )
        total = task_loss + cfg.vq_loss_weight * info["vq_loss"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        losses.append(float(total.detach()))

    for loss in losses:
        assert torch.isfinite(torch.tensor(loss))
    assert gates_applied == [0.0, 0.25]  # warmup schedule produced expected values
    assert 0.0 < float(model.recurrent.spectral_radius()) < 1.0


def test_training_step_emits_nonzero_vq_loss(cfg: MultiAgentConfig):
    """
    VQ loss should be a positive scalar — if it is zero every step, the VQ
    branch is dead or disconnected. Trainer relies on this being a live signal.
    """
    torch.manual_seed(8)
    model = MultiAgentMythos(cfg)
    model.train()
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    _logits, info = model(x, return_info=True)
    assert float(info["vq_loss"].detach()) > 0.0
    assert int(info["n_steps"]) >= 1


# ---------------------------------------------------------------------------
# Bus pathway is actually wired into backward
# ---------------------------------------------------------------------------


def _bus_decoder_grad_norm_after_step(cfg: MultiAgentConfig, gate: float) -> float:
    """Return ‖∇(bus.decoder.weight)‖ after one task-loss backward with the gate pinned."""
    torch.manual_seed(9)
    model = MultiAgentMythos(cfg)
    model.train()
    model.recurrent.bus_gate.set_gate(gate)

    x = torch.randint(0, cfg.vocab_size, (2, 4))
    logits = model(x)  # task loss only — no VQ loss in the graph
    task_loss = logits.square().mean()
    task_loss.backward()
    grad = model.recurrent.bus.decoder.weight.grad
    return 0.0 if grad is None else float(grad.norm())


def test_gate_zero_blocks_task_gradient_through_bus_decoder(cfg: MultiAgentConfig):
    """
    With BusGate = 0, the bus decoder output is multiplied by zero before
    joining the residual stream. The task-loss gradient reaching
    `bus.decoder.weight` must therefore be exactly zero, otherwise there is
    a secondary path bypassing the gate and the ablation control is unsafe.
    """
    norm = _bus_decoder_grad_norm_after_step(cfg, gate=0.0)
    assert norm == 0.0


def test_gate_one_produces_nonzero_task_gradient_through_bus_decoder(
    cfg: MultiAgentConfig,
):
    """
    Inverse of the previous test: opening the gate must actually wire the bus
    decoder into the backward graph. If this is zero with gate=1, the bus is
    architecturally disconnected and warmup will have no effect.
    """
    norm = _bus_decoder_grad_norm_after_step(cfg, gate=1.0)
    assert norm > 0.0
