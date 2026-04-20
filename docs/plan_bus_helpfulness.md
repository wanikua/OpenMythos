# Bus-Helpfulness Experiment — Plan

> **Status:** proposal. No code written yet. Intended scope: CPU-feasible toy
> experiment, not a publication result. This plan was written in response to
> the honest question: *does the Meow bus actually help, or is it architectural
> theater?*

---

## 1. Research question

Given the current multi-agent architecture in `open_mythos/agents.py`, **does
enabling peer messaging through the Meow bus (BusGate > 0) produce a
measurable reduction in training loss on a task that rewards cross-agent
information sharing**, compared to the exact same architecture with the bus
disabled (BusGate pinned to 0)?

This is the simplest falsifiable version of "is the bus doing anything."

### Why this is the question, not something grander

- We do **not** claim this will show the bus beats a Claude-Code-style
  harness. Toy-scale language experiments cannot support that claim.
- We do **not** claim this generalizes to a pretrained 3B+ checkpoint. It
  tests one mechanism on one synthetic task at one scale.
- We only claim: *at matched architecture, matched data, matched
  optimizer, can we detect the bus contributing non-zero signal?* If the
  answer is **no** even on a task designed to reward sharing, the bus
  deserves scrutiny before a real training run.

---

## 2. Hypothesis (stated so it can fail)

> On a synthetic task where the answer depends on information that is
> partitioned across agents at input time, **Condition B (bus on, linear
> warmup)** will achieve lower final eval loss than **Condition A
> (bus off, gate pinned 0)**, with the gap ≥ 1 bootstrap-95 % CI of
> the A-run variability, at chain length k=8.

If the gap is not present, or is smaller than the seed variance, the
honest conclusion is: *the bus did not help here*. That is still a useful
result.

---

## 3. Task design

### 3.1 Task: partitioned modular-arithmetic chain

- Vocabulary: digits `0..6`, operator tokens `+ - *`, a separator `|`,
  and a special `[ASK]` query token.
- Each training example is a sequence of `k` operations in mod-7
  arithmetic, with operands partitioned across **two streams** separated
  by `|`. The target is the final mod-7 result.

Example (k=4, mod 7):
```
Stream-left   : 3 + 2 * a   |
Stream-right  : a - 4 * 5   | [ASK]
Target        : 6
```
where `a` is a shared variable whose value is determined by the
interaction of the two streams (e.g. `a = left_partial ⊕ right_partial`
via a fixed rule).

The key property: **neither stream alone determines the answer.** A
single agent that sees only one partition cannot solve the task above
chance. A model that can share compressed state between agents can.

### 3.2 How agents get partitioned input

For the multi-agent model (`n_agents=2`), we exploit the fact that
`MultiAgentRecurrentBlock` currently feeds *the same* prelude output `h`
to all agents, with divergence driven by `agent_embed + LoRA`. To create
a real information asymmetry, we will mask stream-right tokens for
agent 0 and stream-left tokens for agent 1 **at the prelude input
stage** via a per-agent attention mask passed through an experimental
hook. (See §11 open decisions — this mask injection is the one piece that
is not already implemented.)

For the single-agent baseline (Condition C), the model sees the full
unmasked sequence.

### 3.3 Task parameters

| Param | Value |
|---|---|
| Modulus | 7 |
| Operator set | `{+, -, *}` |
| Chain length `k` | 4 or 8 |
| Vocabulary size | 16 (0–6, +, −, \*, \|, [ASK], [PAD], [BOS], [EOS], [MASK]) |
| Train examples | 20 000 (generated on the fly) |
| Eval examples | 2 000 (held-out, same generator, different seed) |
| Sequence length | ≤ 64 tokens |

Task is synthetic and deterministic given a seed; no external data
dependency.

---

## 4. Experimental conditions

| ID | Config | Gate schedule | Agents | Notes |
|---|---|---|---|---|
| **A-off** | MultiAgentConfig | `set_gate(0.0)` every step | 2 | Bus computed but ignored |
| **B-on** | MultiAgentConfig | `get_bus_gate_value(step, 500)` (linear 0→1 over 500 steps, then held) | 2 | Bus active after warmup |
| **C-single** | MythosConfig (same dim/heads/loops) | N/A | 1 (OpenMythos) | Architectural baseline; not apples-to-apples in capacity |

- **A vs B** is the primary comparison. Same architecture, same parameters,
  only difference is whether gated peer messages are added.
- **C** exists to detect pathological regressions: if both A and B are
  worse than C, the multi-agent architecture is hurting even when the
  bus is off.

### 4.1 Seeds

3 seeds per condition × 2 chain lengths × 3 conditions = **18 runs**.

### 4.2 Gate-pinning mechanism

The existing `BusGate.set_gate(float)` method at
`open_mythos/meow/bus.py:218` already supports this. For Condition A we
will call `rec_block.bus_gate.set_gate(0.0)` every step, overwriting any
learned update (gradients will still flow to the gate parameter, but the
forward value is clamped before use).

For Condition B we use the existing
`get_bus_gate_value(step, bus_warmup_steps=500)` from
`training/multi_agent_pretrain.py:115` (shorter warmup than the real
training script's 5000 because toy runs are short).

---

## 5. Shared hyperparameters

All three conditions use identical:

| Param | Value | Rationale |
|---|---|---|
| `dim` | 128 | CPU-feasible |
| `n_heads` | 4 | dim / 32 |
| `n_kv_heads` | 2 | GQA 2:1 |
| `attn_type` | `"gqa"` | Simpler than MLA at this scale |
| `prelude_layers` | 1 | Minimum depth |
| `coda_layers` | 1 | Minimum depth |
| `max_loop_iters` | 4 | Enough for 8-step chain with shared loops |
| `act_threshold` | 0.99 | Default |
| `use_moe` | False | Reduce variables |
| `lora_rank` | 8 | Default |
| `meow_codebook_size` | 64 | Small enough for CPU |
| `meow_codebook_dim` | 32 | dim / 4 |
| `meow_msg_len` | 2 | 2 tokens per broadcast |
| `meow_commitment_cost` | 0.25 | VQ-VAE default |
| `meow_ema_decay` | 0.99 | Default |
| `vq_loss_weight` | 0.1 | Matches decision record ADR-008 |
| Optimizer | AdamW | `lr=3e-3`, `betas=(0.9, 0.95)`, `weight_decay=0.01` |
| Batch size | 32 | CPU |
| Steps | 5 000 | ~10–20 min on a laptop CPU |
| Eval cadence | every 250 steps | 20 eval points per run |
| Loss | cross-entropy on final target token only | Standard |

---

## 6. Measurement plan

### 6.1 Logged per eval point (250-step cadence)

One line per eval, JSONL, schema:

```json
{
  "run_id": "str",
  "condition": "A-off|B-on|C-single",
  "k": 4,
  "seed": 0,
  "step": 1500,
  "train_loss_ema": 1.23,
  "eval_loss": 1.40,
  "eval_acc": 0.21,
  "bus_gate_effective": 0.0,
  "mean_cum_p_final_loop": 0.87,
  "steps_executed_mean": 3.2,
  "vq_perplexity": 12.4,
  "vq_usage_rate": 0.35,
  "rho_A_max": 0.98
}
```

### 6.2 Logged once per run

- Final eval loss (average of last 3 eval points to reduce noise).
- Total wall-clock training time.
- Parameter count (sanity check that A/B have matched capacity).
- Model config as JSON (for reproducibility).

### 6.3 Aggregation

- For each (condition, k), compute mean ± bootstrap 95 % CI (1 000
  resamples) over 3 seeds on final-3-eval eval-loss.
- Plot: eval loss vs step, one line per (condition, k), shaded
  seed-range.

---

## 7. Acceptance criteria

### 7.1 Primary (the actual question)

At **k=8** (the harder chain):
- Δ = mean(A-off) − mean(B-on) on final eval loss.
- If Δ > 0 **and** the bootstrap-95 % CI of Δ excludes 0 → **bus helps**.
- If Δ ≤ 0 or CI includes 0 → **bus did not help on this task**.

### 7.2 Sanity checks (must pass regardless of Δ)

1. All 18 runs finish without hitting training kill-switches (no NaN, no
   ρ(A) ≥ 1, no codebook collapse).
2. Parameter counts for A-off and B-on runs are identical (same code
   path, only gate schedule differs).
3. Condition C final loss ≤ Condition A final loss at k=4 (if single-
   agent beats the bus-off two-agent, we have a capacity artifact to
   investigate before the A/B comparison means anything).

### 7.3 Interesting-but-not-required observations

- VQ perplexity trajectory (is the codebook actually being used in
  Condition B? If perplexity stays near 1, the bus carries no
  information).
- Usage rate (did the dead-code replacement trigger? How often?).
- Does Condition B need the bus (i.e. does learned gate value settle
  well above 0), or does it drop back to ≈0?

---

## 8. Deliverables

Layout under `experiments/bus_helpfulness/`:

```
experiments/bus_helpfulness/
├── task.py                # generator for partitioned modular-arithmetic chains
├── run.py                 # single-run driver (one condition, one seed, one k)
├── bench.py               # sweep driver: enumerates 18 runs, dispatches sequentially
├── plot.py                # reads results.jsonl, produces figures/*.png
├── results.jsonl          # appended-to by run.py
├── figures/
│   ├── eval_loss_k4.png
│   ├── eval_loss_k8.png
│   └── bus_gate_trajectory.png
└── REPORT.md              # honest write-up: Δ, CI, discussion, limits
```

Implementation constraints:
- `run.py` must use the existing `MultiAgentConfig` and
  `MultiAgentOpenMythos` — no copies or variants of the model code.
- The per-agent input-masking hook is the only new piece of model-
  adjacent code. It must not live in `open_mythos/main.py`; if it can't
  be implemented by a wrapper around `MultiAgentRecurrentBlock`, the
  experiment design needs to change (see §9).
- `bench.py` must be resumable (skip runs whose `run_id` is already
  present in `results.jsonl`).
- `REPORT.md` must include the phrase "no Δ detected" if no Δ is
  detected. No softening.

---

## 9. What this experiment does NOT prove

Explicit non-claims to save later conversations:

1. It does not show the architecture scales. Toy-scale results
   frequently invert at real scale.
2. It does not show the bus is better than any harness-based
   alternative. No harness is tested.
3. It does not show MoE / MLA / FSDP / Coda interact correctly — those
   are disabled or trivial here.
4. It does not show bus messages are semantically meaningful — only
   that they correlate with lower loss.
5. A negative result does not falsify the broader multi-agent design;
   it falsifies only: *on this task, at this scale, with these
   hyperparameters, this bus implementation did not produce a detectable
   gain*.

---

## 10. Failure modes & mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| Bus gate collapses to 0 in Condition B | B-on indistinguishable from A-off | Verify by logging effective gate; may indicate warmup too slow or task too easy |
| Codebook dies (perplexity → 1) | VQ loss stable but eval_loss flat | Use dead-code replacement already in `codebook.py`; log usage_rate |
| Single-agent C beats both multi-agent conditions at k=4 | Capacity artifact from LoRA / agent_embed overhead | Flag in REPORT; compare param counts; do not claim bus is helpful unless A/B comparison holds independently |
| Per-agent input masking leaks info via shared prelude | A-off solves task anyway | Verify by checking A-off's k=8 loss; if it's near B-on's, the masking is ineffective and the experiment is invalid |
| Training diverges (kill-switch fires) | SystemExit(1), `_abort_*.pt` written | Lower LR; reduce warmup; if persistent, bisect which change caused it |
| Runtime > 30 min per run on CPU | Wall-clock too slow | Reduce `steps` to 2500, or `dim` to 96; document in REPORT |

---

## 11. Timeline (rough)

| Phase | Effort |
|---|---|
| 1. Task generator (`task.py`) + unit tests | 45 min |
| 2. Per-agent input-masking hook + test | 60 min |
| 3. Single-run driver (`run.py`) | 30 min |
| 4. Sweep driver (`bench.py`) | 20 min |
| 5. Execute 18 runs sequentially on CPU | ~3 h wall-clock (unattended) |
| 6. Plot + REPORT.md | 45 min |
| **Total active work** | **~4 h** |
| **Total wall-clock including unattended sweep** | **~7 h** |

---

## 12. Open decisions (need user sign-off before coding)

1. **Per-agent input masking.** The experiment hinges on partitioning
   the prelude input asymmetrically to the two agents. Three options:
   - (a) Add an optional `per_agent_input_mask` kwarg to
     `MultiAgentRecurrentBlock.forward`. Small API change.
   - (b) Wrap the model in an experiment-local subclass that overrides
     `_init_agent_states`. No upstream change.
   - (c) Instead of masking inside the model, build two partitioned
     sequences and pass them through a two-stream prelude. Bigger
     refactor; closer to the real multi-agent use case.
   - Default recommendation: **(b)** — keeps `open_mythos/` untouched,
     follows CLAUDE.md rule that `main.py` is frozen.

2. **Task hardness.** Is `k=8` with mod-7 solvable by this tiny model
   at all? If preliminary Condition C runs plateau at chance, we should
   drop to `k ∈ {3, 6}` or increase `dim` to 192. Propose: run 1 short
   pilot of Condition C at k=8 with 1000 steps before committing to the
   full sweep.

3. **Should Condition C be dropped?** It is not strictly needed for the
   A-vs-B comparison, and its mismatched architecture makes it
   distracting. Keeping it costs ~33 % of wall-clock. Propose: keep
   it, because a regression here is the cheapest signal that something
   is wrong with the multi-agent code path generally.

4. **Reporting bar.** Do we require publication-quality plots, or is a
   single matplotlib figure per chain length acceptable? Propose:
   matplotlib defaults, one figure per k, no polish.

5. **Where to link `REPORT.md`.** Default: link from
   `docs/multi_agent.md` under a new "Empirical evidence" section, with
   the result stated honestly regardless of sign.

---

## 13. Out of scope for this plan

- Cloud / GPU runs (explicitly a CPU plan).
- Any comparison to external baselines (harness, ReAct, tree-of-thoughts,
  etc.). Out of scope until A-vs-B is resolved.
- Anything that modifies `open_mythos/main.py`.
- A second experiment for `n_agents > 2`. If B-on wins at `n=2`, a
  follow-up plan for `n ∈ {2, 4, 8}` is justified; writing it now is
  speculation.

---

*Linked from:* `docs/multi_agent.md`
*Written:* 2026-04-20
*Author:* Claude (in-session plan, pending human review)
