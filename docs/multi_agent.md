# Native Multi-Agent OpenMythos

**Module:** `open_mythos.multi_agent_model`, `open_mythos.agents`, `open_mythos.meow`
**Top class:** `MultiAgentMythos(OpenMythos)`

> **Status: architecture + unit tests only.** No pretrained checkpoint
> exists. The stability invariant `ρ(A) < 1` is mathematically guaranteed
> by construction and covered by tests; every other claim about
> "emergent shared protocol," "better reasoning," or "beating a harness"
> is an **untested hypothesis** until pretraining runs and a head-to-head
> evaluation happen. This doc describes what the code *does*, not what
> the model is known to *achieve*.

---

## TL;DR

N agents run inside a single forward pass of a single set of weights.
They share the recurrent body, specialize via per-agent LoRA deltas, and
exchange **discrete codebook messages** every recurrent loop step via a
vendored VQ-VAE codebook (the "Meow protocol" — `wanikua/meow`, MIT).
The whole system is one `nn.Module`, one optimizer, one backward pass.
No external orchestrator, no decoded text between agents, no per-agent
KV cache over the network.

"Native" here means the multi-agent property is a property of the
forward pass itself, not of a harness that sits on top of one or more
models. Whether this translates into measurable downstream quality is
an open question — see the caveat above.

---

## Architecture

```
Input token IDs  (B, T)
        ↓
   [Embedding]
        ↓
   [Prelude]            ── shared across agents
        ↓  h ──────┬────────── e (frozen injection target)
        │         │
        ├── agent 0 state ───┐
        ├── agent 1 state ───┤
        ├── agent 2 state ───┼──── for t in range(T_loop):
        ├── agent 3 state ───┤         (1) every agent encodes h_i_t
        │                    │             through the shared Meow bus
        │                    │             → discrete symbols → decoded
        │                    │             message m_i_t
        │                    │
        │                    │         (2) every agent receives
        │                    │             r_i_t = mean_{j≠i} m_j_t
        │                    │             (mean = bounded, stable)
        │                    │
        │                    │         (3) every agent updates
        │                    │             h_i_{t+1} = A h_i_t + B e
        │                    │             + Transformer(h_i_t)
        │                    │             + LoRA_{t·N+i}(·)
        │                    │             + BusGate_i · r_i_t
        │                    │
        │                    │         (4) per-agent ACT halting
        │                    │
        └── mean over agents ─── h_final
                ↓
           [Coda]
                ↓
           [RMSNorm → LM head]
                ↓
          Output logits  (B, T, vocab_size)
```

Key components:

| Component | Shape / scale | Purpose |
|---|---|---|
| `MultiAgentRecurrentBlock.block` | one `TransformerBlock` | Shared body every agent uses |
| `agent_embed` | `(n_agents, dim)`, init N(0, 0.02) | Per-agent identity, added once to the initial state |
| `lora` | `LoRAAdapter(dim, rank, max_loops × n_agents)` | Per-(loop, agent) specialization deltas |
| `bus` | `MeowBus(dim, K, cdim, msg_len)` | Shared codebook + encoder/decoder; VQ-VAE EMA |
| `bus_gate` | `BusGate(n_agents)`, init 0 | Per-agent scalar that controls peer influence |
| `act` | `ACTHalting(dim)` | Per-agent, per-position halting signal |
| `injection` | `LTIInjection(dim)` | Diagonal A with ρ(A) < 1 (shared across agents) |

---

## Stability Guarantee

The core invariant of the single-agent recurrent block — `ρ(A) < 1` — is
**preserved exactly** in the multi-agent extension. No extra regularizer,
no projection, no gradient penalty needed.

### Update rule

For agent `i`, at recurrent step `t`:

```
m_{j,t}       = Bus.decode( VQ( Bus.encode( h_{j,t} ) ) )           ∀ j
r_{i,t}       = mean_{j ≠ i} m_{j,t}
ĝ_{i,t}       = BusGate_i · r_{i,t}
combined      = RMSNorm( loop_index_embedding(h_{i,t}, t) + e )
trans_{i,t}   = TransformerBlock(combined) + LoRA(trans_{i,t}, t·N+i)
h_{i,t+1}     = A · h_{i,t} + B · e + trans_{i,t} + ĝ_{i,t}
```

### Boundedness argument

The ACT-weighted integral of `h_{i,t}` stays bounded provided the drive
term `ĝ_{i,t}` is bounded independent of h-state. Check each factor:

1. **Codebook bound.** `‖codebook[k]‖ ≤ C_cb` for every entry at every step.
   The EMA update in `VectorQuantizer` contracts each entry toward the
   per-symbol mean of input samples; on bounded input it stays bounded.
2. **Decoder bound.** `Bus.decoder` is a single `Linear` layer; at any
   training step `‖decoder.weight‖ ≤ C_dec`.
3. **Message bound.** A message is `decoder(quantized)` where `quantized`
   stacks `msg_len` codebook entries, so `‖m_j,t‖ ≤ C_msg`, a constant
   of the training step — **not a function of any `h_i,t`**.
4. **Aggregation preserves the bound.** `mean_{j≠i} m_{j,t}` has norm
   `≤ C_msg` by Jensen. **Sum aggregation would multiply by N and break
   this — `MeowBus.aggregate` uses mean, not sum.**
5. **Gate bound.** `BusGate_i ∈ ℝ` is a learned scalar; at any finite
   training step it is finite. So `‖ĝ_{i,t}‖ ≤ |BusGate_i| · C_msg`.

Because `A` is diagonal with every entry in `(0, 1)` and the drive
`B·e + trans + ĝ` is bounded independent of `h_{i,t}`, the recurrence
is a contraction in `h`. The ACT-weighted output `h_out_i` is a convex
combination of bounded `h_{i,t}`, so it is bounded. The cross-agent mean
at the end is again bounded by Jensen.

The `spectral_radius()` method on `MultiAgentRecurrentBlock` is the
cheapest runtime check of this guarantee; `training/multi_agent_pretrain.py`
logs it every `diag_every` steps.

### Why `BusGate_i` is initialized to zero

Before any training, the drive term is identically zero and the
multi-agent model is bit-exact equivalent to N independent single-agent
runs on the same input. Training opens the gate monotonically
(see `get_bus_gate_value` in the trainer), so peer influence only appears
once task learning has settled. This removes the usual "early
multi-agent instability" phase entirely. The guarantee above holds at
every value of BusGate, so the ramp is a pure training curriculum —
not a stability constraint.

---

## Architectural differences from a harness-style setup

**No benchmark comparison has been run.** This section catalogs
architectural differences, not measured outcomes. Whether any of these
differences matter in practice depends on trained behavior that doesn't
yet exist.

A harness orchestrates multiple independent LLM calls over text: each
agent runs a decoder, tokenizer, and prompt pack; each hop pays a full
inference cost; shared state is whatever fits in a prompt; the only
artifact you can inspect is the final tool call or text reply.

| Axis | Harness | Native (OpenMythos multi-agent) |
|---|---|---|
| Inter-agent latency | One full decode per hop | One linear layer + argmin per hop |
| Message bandwidth | Tokens of natural language | `msg_len · log2(K)` bits per token-position per loop step |
| Shared state | Prompt window per agent | Shared hidden state every step via bus |
| Training signal | Only on the final LM head output | Task loss + VQ loss jointly optimize bus usage |
| Auditability | Final text / tool calls | Every message is a discrete index — full trace via `MeowTrace` / `MeowAuditor.gloss` |
| Compute model | N × model inference per turn | 1 × model forward, N agents inside it |
| Memory model | N × KV cache over network | N × KV cache colocated with weights (FSDP-shardable) |
| Determinism | Depends on each agent's sampler | Deterministic in eval mode (locked down by `test_eval_mode_forward_is_deterministic`) |

Harnesses are not wrong — they are the common answer when you need tool
use and adaptation over long horizons, and they let you compose models
you don't control. The **hypothesis** of the native approach is that
when the task is "N cooperating reasoners producing one answer," moving
coordination inside the forward pass reduces latency and produces a
trace you can actually inspect. Validating that hypothesis requires a
trained model and a paired harness baseline — neither exists yet.

---

## Interpretability

The single biggest payoff of the Meow protocol is that every message
is a **discrete, finite-vocabulary index** — not a continuous vector
you have to probe. This means:

- `MeowTrace.to_rows()` returns a table
  `(loop_step, agent, b, t, [sym_0, ..., sym_{msg_len-1}])` for one
  forward pass. Every inter-agent word ever exchanged is visible.
- `MeowAuditor.gloss(tokenizer)` maps codebook symbols to their nearest
  natural-language tokens, producing human-readable labels per symbol.
  Labels are checkpoint-local (the codebook is learned), but within one
  checkpoint they are stable.
- `snapshot_codebook_stats(bus)` returns live/dead code counts and
  norm bounds, for tracking codebook collapse during training.

---

## Usage

### Build a model

```python
from open_mythos import MultiAgentMythos, multi_agent_3b

cfg = multi_agent_3b()          # 4 agents, codebook_size=512, msg_len=4
model = MultiAgentMythos(cfg)
```

### Forward for inference

```python
logits = model(input_ids)       # (B, T, vocab_size)
```

### Forward for training (VQ loss required)

```python
logits, info = model(input_ids, return_info=True)
task_loss = nn.functional.cross_entropy(logits.view(-1, V), targets.view(-1))
loss = task_loss + cfg.vq_loss_weight * info["vq_loss"]
loss.backward()
```

### Inspect inter-agent traffic

```python
model.eval()
logits, trace = model.trace_forward(input_ids)
print(trace.summary())           # avg perplexity + usage rate
for row in trace.to_rows()[:10]: # first 10 agent utterances
    print(row)                   # (loop_step, agent, b, t, [sym...])
```

### Probe the emergent vocabulary

```python
from open_mythos import MeowAuditor

auditor = MeowAuditor(
    codebook_weight=model.recurrent.bus.quantizer.embedding.weight,
    token_embed_weight=model.embed.weight,
    codebook_dim=cfg.meow_codebook_dim,
)
# Use the decoder's weight as a codebook→hidden projection.
W = model.recurrent.bus.decoder.weight
projection = W.view(cfg.dim, cfg.meow_msg_len, cfg.meow_codebook_dim).mean(dim=1).T
auditor.set_projection(projection)

for line in auditor.gloss(tokenizer, k=3):
    print(line)                  # e.g. "[sym 42] → 'because', 'so', 'therefore'"
```

### Ablate peer communication

```python
# Pin the bus gate to 0 → agents behave as N independent single agents.
model.recurrent.bus_gate.set_gate(0.0)
out_no_comms = model(input_ids)

# Pin the bus gate to 1 → full peer integration.
model.recurrent.bus_gate.set_gate(1.0)
out_full_comms = model(input_ids)
```

---

## Variants

Defined in `open_mythos.variants`. Parameter counts below are
`sum(p.numel() for p in model.parameters())` at the declared `dim` /
`n_experts` / `max_loop_iters` — no checkpoint has been trained.

| Factory | `n_agents` | Codebook | Notes |
|---|---|---|---|
| `multi_agent_1b()` | 4 | K=512, cdim=128, msg_len=4 | Smallest variant — usable for protocol smoke tests |
| `multi_agent_3b()` | 4 | K=512, cdim=128, msg_len=4 | Matches single-agent `mythos_3b()` scale |
| `multi_agent_10b()` | 8 | K=1024, cdim=128, msg_len=4 | Not validated that 8 agents help vs 4 |

Parameter overhead of multi-agent over single-agent (at the same `dim`,
`max_loop_iters`, `n_experts`) is under 2% — the bus adds
`2 · dim · msg_len · cdim + K · cdim` weights, and per-agent LoRA adds
`dim · rank + rank · dim + max_loops · n_agents · rank`.

---

## Training

See `training/multi_agent_pretrain.py`.

The trainer differs from the single-agent trainer in exactly three places:

1. Uses `MultiAgentMythos` + `return_info=True`, adds
   `cfg.vq_loss_weight * info["vq_loss"]` to the cross-entropy task loss.
2. Linear BusGate warmup over `bus_warmup_steps` (default 5000). Step 0
   = bit-exact N independent single agents; step `bus_warmup_steps` =
   full peer integration.
3. Every `diag_every` steps, logs `ρ(A)` and codebook health (live / dead
   codes, max norm). Watch for dead-code fraction creeping up or max-norm
   spiking — both are early signals of a regression on the stability
   argument above.

The FSDP wrap policy is extended with `MultiAgentRecurrentBlock` so each
rank shards the multi-agent body exactly like the single-agent
`RecurrentBlock`.

**This script is not auto-launched.** Run manually on authorized
hardware; see the module docstring for invocations.

---

## Tests

`tests/test_multi_agent.py` locks the multi-agent-specific invariants:

- Forward shape / info contract
- `ρ(A) < 1` at init and after one backward pass
- VQ straight-through gradient flow reaches the bus encoder
- Codebook `usage_rate > 0` in training mode
- `BusGate=0` behaves as N independent single agents
- `BusGate≠0` actually propagates peer influence
- Per-agent, per-loop KV cache keys populated
- Variant factories return valid `MultiAgentConfig`
- Eval-mode determinism

Run:

```bash
pytest tests/test_multi_agent.py -q
```

---

## File map

| Path | Role |
|---|---|
| `open_mythos/meow/codebook.py` | Vendored `VectorQuantizer` (MIT, `wanikua/meow`) |
| `open_mythos/meow/bus.py` | `MeowBus` + `BusGate` |
| `open_mythos/meow/audit.py` | `MeowTrace`, `MeowAuditor`, `snapshot_codebook_stats` |
| `open_mythos/agents.py` | `MultiAgentConfig`, `MultiAgentRecurrentBlock` |
| `open_mythos/multi_agent_model.py` | `MultiAgentMythos` top-level class |
| `open_mythos/variants.py` | `multi_agent_1b` / `3b` / `10b` factories |
| `training/multi_agent_pretrain.py` | FSDP pretraining scaffold (manual launch) |
| `tests/test_multi_agent.py` | Invariant tests |
| `docs/multi_agent.md` | This document |
| `docs/multi_agent_decisions.md` | Architectural decision record — **read before changing load-bearing choices** |

Crucially, `open_mythos/main.py` is **unchanged**. The multi-agent
architecture is purely additive: pick `MultiAgentMythos` instead of
`OpenMythos` at construction, and the rest of the stack (tokenizer, LM
head tying, RoPE, MLA caching) is identical.
