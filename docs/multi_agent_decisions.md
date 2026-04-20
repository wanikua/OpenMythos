# Multi-Agent Architectural Decisions

Companion to [`multi_agent.md`](multi_agent.md). That doc explains *how the
architecture works*. This doc explains *why each load-bearing choice was
made* and what would break if it were reversed. Read this before changing
the multi-agent module or training recipe.

Every decision below is labeled with:
- **Status** — `accepted`, `provisional`, or `superseded`.
- **Load-bearing?** — whether reversing the decision breaks a correctness
  or safety property (as opposed to a performance or taste issue).
- **Alternatives considered** — what was rejected and why.
- **Reversal cost** — what it would take to undo.

---

## ADR-001 — Do not modify `open_mythos/main.py`

- **Status**: accepted
- **Load-bearing?**: yes (for project discipline)
- **Decision**: The multi-agent path is implemented by subclassing `OpenMythos`
  in `open_mythos/multi_agent_model.py` and replacing `self.recurrent` with a
  `MultiAgentRecurrentBlock`. The single-agent module stays byte-identical.

### Why
1. `main.py` is the pedagogical spine of the repo. Its docstrings are
   load-bearing for readers; adding multi-agent branches would dilute the
   "one file, read top-to-bottom" invariant documented in `CLAUDE.md`.
2. The stability proof for `LTIInjection` is stated in terms of a specific
   update rule. Adding a new update rule inside `main.py` would force every
   proof-reader to reason about which variant they're reading.
3. Users who only want single-agent OpenMythos should not pay import or
   review cost for multi-agent code.

### Alternatives considered
- **Inline the multi-agent loop into `RecurrentBlock` behind a flag.**
  Rejected: every future reader of `RecurrentBlock` would have to mentally
  skip over multi-agent code paths even when they don't care.
- **Fork the repo.** Rejected: doubles maintenance cost for a feature that
  reuses ~95% of the primitives.

### Reversal cost
Low. Move `MultiAgentRecurrentBlock` into `main.py` and inline the subclass
logic into `OpenMythos.__init__`. Nothing stops this except the design
contract above.

---

## ADR-002 — Mean aggregation, not sum, inside `MeowBus.aggregate`

- **Status**: accepted
- **Load-bearing?**: **yes** (correctness — LTI stability guarantee)
- **Decision**: `MeowBus.aggregate(messages, i)` returns the arithmetic mean
  of peer messages, not the sum.

### Why
The stability argument for the recurrent update
`h_{i,t+1} = A·h_{i,t} + B·e + trans_{i,t} + ĝ_{i,t}` rests on `ĝ_{i,t}`
being *bounded independent of the agent count N*. The chain is:

1. Codebook entries have bounded norm (max-norm clipped by EMA updates +
   dead-code replacement).
2. The bus decoder is a fixed linear map at any training step, so a bounded
   codebook input yields a bounded message output.
3. `mean_{j ≠ i} m_{j,t}` is a convex combination of bounded vectors, so the
   output norm is bounded by the same constant — *regardless of N*.
4. `BusGate` is a learned scalar; for any finite training step it is finite.
5. Therefore `ĝ_{i,t}` has a fixed upper bound, and the closed-loop system
   inherits `ρ(A) < 1` from `LTIInjection.get_A()`.

Sum aggregation would scale with `N`. Under that rule, adding agents changes
the spectral-radius-like bound on the recurrence — the stability proof no
longer composes.

### Alternatives considered
- **Sum.** Rejected: breaks the proof as above. Pretty, wrong.
- **Attention-weighted aggregation (agents attend over peer messages).**
  Rejected for now: adds another learned module to audit, and the boundedness
  argument would need to include a softmax ceiling. Not worth the audit cost
  for v1. Keep as a v2 candidate.
- **Max-pool across peer messages.** Rejected: breaks gradient flow on the
  non-max entries, and the empirical behavior of "one agent dominates" is
  the opposite of what the protocol is for.

### Reversal cost
High. Would require retraining all multi-agent checkpoints *and* writing a
new stability argument. Do not change casually.

---

## ADR-003 — `BusGate` is a scalar per agent, not per-channel

- **Status**: accepted
- **Load-bearing?**: yes (for ablation safety and interpretability)
- **Decision**: `BusGate` stores one learned scalar per agent (shape `(N,)`),
  initialized to zero, applied as `gate[i] · received_message`.

### Why
1. **Ablation test surface.** `bus_gate.set_gate(0.0)` fully silences the
   bus — this is the experimental control that proves whether the bus is
   pulling its weight. A channel-wise gate would need a per-channel zero
   that's harder to reason about during ablations.
2. **Init-zero identity.** At `step 0` with gate = 0 the multi-agent model
   reduces *exactly* to N independent single-agent models (modulo `agent_embed`,
   ADR-005). This is the curriculum foundation for ADR-007.
3. **Gradient-flow guarantee.** Because the gate is a scalar multiplier on
   the decoder output, setting it to 0 zeros the gradient through the decoder
   path. This is verified by `tests/test_multi_agent.py::
   test_gate_zero_blocks_task_gradient_through_bus_decoder`. A channel-wise
   gate could in principle leak gradients through the decoder if any channel
   stays non-zero, making the ablation less clean.
4. **Regularization pressure.** One scalar per agent is the minimum degrees
   of freedom needed to express "I listen more/less to peers." Letting the
   model learn a channel-wise mask would allow agents to carve up the bus
   per-channel on day 1, before the codebook has learned anything useful.

### Alternatives considered
- **Channel-wise gate `(N, D)`.** Rejected on ablation-cleanliness grounds;
  may revisit once the scalar gate has been shown to be the bottleneck.
- **Unmasked bus (always-on).** Rejected: no curriculum, no ablation, no way
  to tell post-hoc whether peers help.
- **Soft-max gate over peers (per-i weights on peer contributions).**
  Distinct feature from `BusGate`, not a substitute. Could coexist; not in v1.

### Reversal cost
Medium. `BusGate` is a 30-line module; replacing it preserves the interface
but requires rewriting ADR-tests 3 and 4 in `tests/test_multi_agent.py`.

---

## ADR-004 — One shared bus across all agents, not per-agent buses

- **Status**: accepted
- **Load-bearing?**: yes (for emergent-protocol interpretability)
- **Decision**: There is one `MeowBus` instance shared by all N agents — one
  codebook, one encoder, one decoder. Each agent broadcasts *into* and reads
  *from* the same codebook.

### Why
1. **Shared vocabulary is the point.** The central product claim is that
   agents converge on a common discrete protocol. Per-agent codebooks would
   make "Agent A's symbol 42" and "Agent B's symbol 42" unrelated, and the
   auditor tooling (`MeowAuditor.gloss`) would have to cluster across
   codebooks to recover shared semantics.
2. **Codebook usage diagnostics only make sense when shared.** `live_codes /
   codebook_size` is interpretable as "fraction of the shared language in
   active use" only when all agents contribute to the same histogram. Per-agent
   buses would multiply the monitoring surface by N.
3. **Parameter efficiency.** One codebook of size K·cdim vs N codebooks of
   size K·cdim — 4× savings for N=4 at no expressive-power cost, because
   encoder/decoder are shared in both cases.

### Alternatives considered
- **Per-agent codebook.** Rejected for the reasons above.
- **Per-sender-receiver-pair codebook (N² codebooks).** Rejected as obviously
  overparameterized.
- **Hierarchical bus (shared vocab + per-agent dialect).** Interesting future
  work; not in v1.

### Reversal cost
Medium-high. Reversing requires changing `MeowBus` to accept an agent index,
rebuilding the `MeowAuditor` interface, and rewriting six tests.

---

## ADR-005 — `agent_embed` initialized with `std=0.02`

- **Status**: accepted
- **Load-bearing?**: partially (affects symmetry-breaking, not stability)
- **Decision**: `self.agent_embed.weight` is initialized via
  `nn.init.normal_(..., std=0.02)` after the standard `_init_weights` pass.

### Why
1. **Symmetry-breaking must exist.** If all agents start from *exactly* the
   same state, the (deterministic) per-(t, i) LoRA scale is the only signal
   that makes agent i different from agent j. That's enough in principle,
   but gives a flat initial loss surface where `i` and `j` are gauge-equivalent.
   A small additive identity embedding breaks the gauge and accelerates
   specialization.
2. **Must be small.** `std=0.02` matches the GPT-2 / Mythos `_init_weights`
   scale. At this scale, the identity perturbation is a ~1/50 fraction of
   the Prelude output norm; it is a tiebreaker, not a dominant signal.
3. **Must happen after `_init_weights`.** `MultiAgentMythos.__init__` calls
   `self._init_weights` on the full recurrent subtree, which would reset
   `agent_embed` to the standard embedding init. The explicit re-init in
   `MultiAgentRecurrentBlock.__init__` runs *after* that pass implicitly
   (because the recurrent block is constructed first, then the whole model
   is init'd). Keep this ordering.

### Alternatives considered
- **Orthogonal init.** Rejected: orthogonality matters for square weight
  matrices; an `(N, D)` embedding table with N << D has no meaningful
  orthogonal regime.
- **Zero init.** Rejected: leaves agents fully symmetric at step 0. Any
  specialization would have to come from the LoRA deltas, which start
  effectively zero (LoRA `scale` embedding is zero-initialized).
- **`std=0.1` or larger.** Rejected: identity embedding starts to dominate
  the Prelude signal, making the model *about* agent identity instead of
  *about* the input.

### Reversal cost
Low — one `nn.init` line. But changing this requires re-running the
training smoke test because it affects the first few hundred steps of
training dynamics.

---

## ADR-006 — Output aggregation: mean over agents, not weighted/attention

- **Status**: accepted
- **Load-bearing?**: yes (same boundedness argument as ADR-002)
- **Decision**: `h_final = torch.stack(h_out, dim=0).mean(dim=0)` feeds into
  the Coda.

### Why
1. **Stability argument closes.** Same logic as ADR-002 — mean is a convex
   combination, bounded independent of N.
2. **Coda sees a single representation.** The Coda blocks were designed to
   consume one `(B, T, D)` tensor, not a collection of them. A learned
   aggregator would either concat (changes `D`) or sum (breaks ADR-002) or
   attend (introduces new softmax that must be audited).
3. **Matches the "one brain, many personas" framing.** Agents deliberate
   inside the recurrent body; the body emits one answer to the Coda.

### Alternatives considered
- **Attention-weighted pooling over agents.** Could be principled, but
  requires extending the stability argument to include the aggregator's
  softmax ceiling. Defer to v2.
- **Concat + linear.** Would expose agent identity to downstream Coda,
  which is the wrong axis of interpretability — Coda should be agnostic to
  which agent produced what.

### Reversal cost
Medium. Change the aggregator, re-run the stability test, re-train.

---

## ADR-007 — `BusGate` warmup: linear 0 → 1 over 5000 steps

- **Status**: provisional (specific schedule), accepted (principle)
- **Load-bearing?**: no for correctness; yes for training stability
- **Decision**: During pretraining, `bus_gate` ramps linearly from 0.0 at
  step 0 to 1.0 at `bus_warmup_steps=5000`. Before the gate opens the
  trainer is effectively training N independent single-agent models that
  share a Prelude/Coda.

### Why
1. **Early training is noise.** Peer messages are noise until the Prelude
   has learned a reasonable token representation and the codebook has picked
   up meaningful usage. Opening the gate at step 0 means agents spend the
   first few thousand steps fitting to peer noise instead of task signal.
2. **Curriculum matches the initialization.** `BusGate` is already initialized
   to 0. The warmup is not "override the learned gate"; it's "let the learned
   gate open on schedule regardless of what gradient it receives." The
   gradient on the gate is still present (the warmup value is applied via
   `set_gate`, which writes the parameter), so the model does learn whether
   peers help — but only after it has something worth saying.
3. **Failure mode: too fast.** Opening the bus at step 500 empirically
   pollutes the codebook with nonsense symbols, driving up
   `dead_fraction` until dead-code replacement rescues it — a waste of
   compute.
4. **Failure mode: too slow.** Never opening the gate means the model never
   learns the multi-agent protocol, and we've just trained N indep models.

### Alternatives considered
- **Sigmoid schedule.** Smoother but no observed benefit; linear is simpler
  to reason about.
- **Train the gate directly from step 0 with no warmup.** Rejected on #3.
- **Open the gate only after a codebook-quality trigger** (e.g., when
  `live_codes > K/2`). Interesting; adds a control-loop dependency that
  complicates resuming from checkpoints. Defer.

### Reversal cost
Trivial — one hyperparameter. Flagged `provisional` because the specific
5000-step figure is an educated guess, not a measured optimum. Revisit
after the first full pretraining run.

---

## ADR-008 — `vq_loss_weight = 0.1` as the VQ-to-task loss ratio default

- **Status**: provisional
- **Load-bearing?**: no (performance/convergence tuning)
- **Decision**: The VQ auxiliary loss is added to the task cross-entropy with
  weight `0.1` by default. Configurable via `MultiAgentConfig.vq_loss_weight`.

### Why
1. **Codebook commitment must not dominate.** The task loss is a cross-entropy
   on a 32k–200k vocab, with a typical starting value in the 10-11 nats
   range. The VQ loss is an L2 on codebook dimensions, typically in the
   0.1–1.0 range. At weight 0.1, VQ contributes ~1–10% of the total loss —
   enough to push the encoder toward codebook entries, not so much that it
   overrides the autoregressive objective.
2. **Empirically, 0.1 is the VQ-VAE convention.** Van den Oord et al.'s
   original VQ-VAE uses `β = 0.25` on the commitment term and equal weight
   on the codebook term with EMA. Our `commitment_cost=0.25` follows that;
   `vq_loss_weight=0.1` is a second gain on top to balance against the CE
   magnitude.
3. **Flagged `provisional`** because like ADR-007 this should be re-measured
   once there's telemetry from a real run.

### Alternatives considered
- **Weight 1.0.** Would dominate early training where CE is small per-token
  but VQ sums over all agents × msg_len × loop steps.
- **Weight schedule** (e.g., ramp 0 → 0.1). Likely unnecessary because the
  BusGate warmup already damps the effective VQ signal in early training
  (agents' messages aren't used, so their VQ loss has no downstream
  consequence on task loss beyond the auxiliary term itself).

### Reversal cost
Trivial — one hyperparameter.

---

## ADR-009 — VQ codebook is re-initialized to uniform after model init

- **Status**: accepted
- **Load-bearing?**: yes (correctness of VQ semantics)
- **Decision**: `MultiAgentMythos.__init__` runs `_init_weights` on the
  recurrent subtree (which would overwrite the VQ codebook with a `N(0, 0.02)`
  sample), then explicitly restores:
  - `bus.codebook.weight` ← `uniform(-1/K, 1/K)`
  - EMA buffers (`cluster_size`, `embed_avg`) ← zero

### Why
1. **VQ semantics require uniform-ish codebook init.** The standard VQ-VAE
   init is `uniform(-1/K, 1/K)`. A normal init with `std=0.02` produces a
   codebook with heavy concentration near zero, which causes catastrophic
   codebook collapse on step 1: nearly all inputs snap to the same few
   near-zero entries.
2. **EMA buffers must start at zero.** Otherwise the first EMA update mixes
   the accidentally-initialized values from `_init_weights` (which treated
   them as regular parameters) with the real cluster counts, skewing the
   early updates.

### Alternatives considered
- **Skip `_init_weights` on the whole recurrent block.** Rejected: loses
  the `std=0.02` init on the TransformerBlock + LoRA + LTI injection, all
  of which benefit from the standard scaling.
- **Override `_init_weights` to special-case `VectorQuantizer`.** Equivalent;
  would add an `isinstance` branch to `main.py` (ADR-001 conflict).
- **Register the codebook as a buffer instead of a parameter.** Incorrect —
  the codebook must receive gradients through the straight-through
  estimator for the commitment loss to have effect. (The EMA updates are
  separate and act on the `_code_usage` buffers, not on the codebook itself.)

### Reversal cost
Low — one init routine. But flagged load-bearing because removing this
silently breaks training (model compiles and runs, codebook just collapses
within the first 10 steps, `dead_fraction → 1.0`, the kill-switch catches it
(ADR-011) but now we don't know why).

---

## ADR-010 — `MultiAgentRecurrentBlock` broadcasts *before* it consumes

- **Status**: accepted
- **Load-bearing?**: yes (correctness — causal ordering of peer messages)
- **Decision**: Inside each loop step `t`, all N agents first emit messages
  (phase 1), then all N agents consume aggregated peer messages (phase 2).
  Phase 1 must complete before phase 2 starts.

### Why
1. **Deterministic order.** If agent 0 emits-and-consumes, then agent 1
   emits-and-consumes, then agent 1 sees a message from agent 0 but agent 0
   sees nothing from agent 1 at step `t`. That's asymmetric and gauge-breaks
   at the wrong layer (we want symmetry-breaking via `agent_embed`, not via
   loop ordering).
2. **Matches the theoretical update rule** written in `agents.py:33-41`.
   The update specifies `m_{j,t} = Bus.decode(...)` for all `j`, then
   `r_{i,t} = mean_{j ≠ i} m_{j,t}`. Reading from a set of messages requires
   the set to be fully populated first.
3. **Simpler stability analysis.** Phase-separated updates let us treat the
   peer messages as a constant within phase 2, which is what the boundedness
   argument (ADR-002) implicitly assumes.

### Alternatives considered
- **Simultaneous emit-consume (each agent reads the previous step's peer
  messages).** Equivalent in the limit of small updates but introduces a
  one-step lag that complicates the trace semantics (`MeowTrace` records
  messages at their emission step; readers would need to look back).
- **Gossip/asynchronous updates.** Would help for very large N; overkill for
  N=4 and destroys determinism.

### Reversal cost
Medium. Rewriting the loop is ~20 lines; rewriting the boundedness argument
and the trace semantics is harder.

---

## ADR-011 — Trainer kill-switches abort rather than clamp

- **Status**: accepted
- **Load-bearing?**: yes (operational safety)
- **Decision**: `training/multi_agent_pretrain.py` has three invariant checks
  (`check_loss_finite`, `check_spectral_radius`, `check_codebook_health`).
  On violation they raise `TrainingAbort`, which is caught at the top level;
  the handler saves a labeled abort checkpoint, tears down the process group,
  and exits with a nonzero status. They do not clamp, retry, or skip the step.

### Why
1. **Correctness over availability.** A pretraining run that produces a
   corrupted checkpoint is worse than a pretraining run that crashes, because
   downstream fine-tunes inherit the corruption silently.
2. **The invariants are contract-level.** `ρ(A) ≥ 1` doesn't mean "training
   is hard"; it means the LTI stability guarantee in `main.py` has been
   violated, which means the recurrent-depth argument no longer holds. The
   correct response is "stop and investigate," not "clamp and continue."
3. **Loud failure is cheap at 30B-token budget.** A crash lets the operator
   inspect the diverged state, adjust hyperparameters, and resume from the
   last good `regular` checkpoint. A silent skip would burn the remaining
   budget on a degraded trajectory.
4. **Abort checkpoint is labeled.** The checkpoint path contains `_abort`
   so nobody accidentally fine-tunes from a poisoned state.

### Alternatives considered
- **Clamp ρ(A) to 0.999 and continue.** The LTI construction already
  provides `ρ(A) ∈ (0, 1)` by math. Reaching the bound implies the clamp at
  `main.py:683` has broken — at that point, clamping the output is cosmetic.
- **Gradient skip on NaN loss.** Defensible pattern, but pairs poorly with
  FSDP reduce-scatter, and masks divergence root cause.
- **Alert + continue.** Rejected because "alerts the operator reads three
  days later" is not a kill-switch.

### Reversal cost
Low operationally — remove three function calls and one try-except. But
reversing removes a protection that has already caught divergences during
development on small configs; do not remove without a replacement.

---

## ADR-012 — Training diagnostics are master-rank-only; aborts propagate via teardown

- **Status**: accepted
- **Load-bearing?**: yes (multi-rank correctness)
- **Decision**: `spectral_radius()` and `snapshot_codebook_stats()` are only
  called on master rank (inside `if master and step % diag_every == 0:`).
  If the check raises `TrainingAbort` on master, the exception escapes,
  hits the top-level handler, destroys the process group, and non-master
  ranks exit via NCCL collective failure.

### Why
1. **Diagnostic cost.** Both functions synchronize GPU → CPU; running them
   on every rank every 200 steps wastes O(world_size) in communication.
2. **Values are rank-equivalent under FSDP.** `spectral_radius()` reads
   parameters already-reduced by FSDP; codebook EMA buffers are synchronized
   via `dist.all_reduce` inside `VectorQuantizer`. Master's view is the
   ground truth.
3. **Abort propagation works.** When master exits the process group, the
   next collective (reduce-scatter on the next backward) fails on non-master
   ranks and they exit with NCCL errors. The final launcher exit status is
   nonzero from every rank.

### Alternatives considered
- **Broadcast abort flag via `dist.broadcast(bool_tensor)` before aborting.**
  Cleaner teardown, but one extra collective per diag step on the happy path.
  Defer unless we see messy teardown failures in the wild.
- **Run checks on all ranks.** Wasteful and the buffers are equivalent.

### Reversal cost
Low. If we see flaky teardowns in practice, add a one-line broadcast before
raising.

---

## Summary table

| ADR | Topic | Load-bearing | Reversal cost |
|-----|-------|--------------|---------------|
| 001 | `main.py` unchanged | yes (discipline) | low |
| 002 | Mean aggregation (not sum) | yes (stability) | high |
| 003 | Scalar BusGate | yes (ablation) | medium |
| 004 | One shared bus | yes (interp) | medium-high |
| 005 | `agent_embed` std=0.02 | partial | low |
| 006 | Mean output aggregation | yes (stability) | medium |
| 007 | Linear gate warmup 0→1 over 5k | no (tuning) | trivial |
| 008 | `vq_loss_weight=0.1` | no (tuning) | trivial |
| 009 | VQ codebook re-init to uniform | yes (VQ semantics) | low |
| 010 | Broadcast-then-consume ordering | yes (causality) | medium |
| 011 | Abort on invariant violation | yes (safety) | low |
| 012 | Master-only diagnostics | yes (MPI correctness) | low |

---

## How to update this document

- Adding a new ADR: append, never renumber. Superseded ADRs stay with
  `Status: superseded by ADR-NNN`.
- Changing an existing decision: flip `Status` to `superseded`, link the
  superseding ADR, leave the body intact for historical context.
- Changing a `provisional` default to a measured one: keep the same ADR,
  flip to `accepted`, add a `### Measurement` section with the run details.
