# CLAUDE.md

Project guidance for Claude Code working in this repo. Read this before making changes. For the theoretical background on the architecture and published marketing copy, see `README.md` — this file is the working notes.

---

## What this is

**OpenMythos** — a PyTorch, from-scratch, theoretical reconstruction of the hypothesized "Claude Mythos" architecture: a **Recurrent-Depth Transformer (RDT)** with three functional stages and swappable attention/FFN modules. Published on PyPI as `open-mythos` (v0.3.0, MIT, author Kye Gomez). Research/experimental code — not production inference infrastructure.

> Disclaimer from README: independent community reconstruction, NOT affiliated with Anthropic.

---

## Architecture in one screen

```
input_ids (B, T)
     │
     ▼
Embedding (weight-tied to LM head)
     │
     ▼
Prelude           = prelude_layers × TransformerBlock(use_moe=False)   — dense FFN
     │
     ▼           e = encoded input (frozen, re-injected every loop)
RecurrentBlock    = ONE TransformerBlock(use_moe=True) looped T times
     │               per-step update: h_{t+1} = A·h_t + B·e + Transformer(h_t, e) + LoRA(t)
     │               ACT halting weights each h_t; loop exits when all positions halted
     ▼
Coda              = coda_layers × TransformerBlock(use_moe=False)
     │
     ▼
RMSNorm → Linear head (tied weights) → logits (B, T, vocab_size)
```

Implemented entirely in **`open_mythos/main.py`** (1015 lines, one file). Key classes and their file positions:

| Class / fn                | File                        | Purpose |
|---------------------------|-----------------------------|---------|
| `MythosConfig`            | `open_mythos/main.py:9`     | Single dataclass carrying ALL hyperparameters |
| `RMSNorm`                 | `open_mythos/main.py:82`    | RMS-only normalization (no bias) used throughout |
| `precompute_rope_freqs`   | `open_mythos/main.py:117`   | Complex-valued RoPE phasor table |
| `apply_rope`              | `open_mythos/main.py:140`   | Applied to Q and to K **before** caching (cached K is already rotated) |
| `GQAttention`             | `open_mythos/main.py:165`   | Grouped Query Attention |
| `MLAttention`             | `open_mythos/main.py:251`   | DeepSeek-V2 Multi-Latent Attention; caches compressed `c_kv` + `k_rope` only |
| `Expert`                  | `open_mythos/main.py:393`   | SwiGLU FFN unit (also used dense in Prelude/Coda) |
| `MoEFFN`                  | `open_mythos/main.py:423`   | DeepSeekMoE: routed top-K + always-on shared + load-balancing `router_bias` buffer |
| `loop_index_embedding`    | `open_mythos/main.py:504`   | Sinusoidal injection into first `dim//8` channels so shared loop weights can act differently per iteration |
| `LoRAAdapter`             | `open_mythos/main.py:541`   | Depth-wise LoRA — shared `down`/`B`, per-loop `scale` embedding |
| `TransformerBlock`        | `open_mythos/main.py:585`   | Pre-norm block; `use_moe=True` swaps FFN to `MoEFFN` |
| `LTIInjection`            | `open_mythos/main.py:642`   | **Parcae stability trick.** `A = exp(-exp(log_dt + log_A))` guarantees ρ(A) ∈ (0,1) by construction. Do NOT touch unless you understand why |
| `ACTHalting`              | `open_mythos/main.py:708`   | Per-position halting probability (sigmoid scalar) |
| `RecurrentBlock`          | `open_mythos/main.py:746`   | The loop — orchestrates loop_index_embed → block → LoRA → LTI update → ACT accumulation |
| `OpenMythos`              | `open_mythos/main.py:850`   | Full model; `.forward(ids, n_loops=None)` and `.generate(ids, max_new_tokens, n_loops)` with KV cache |

### Invariants worth knowing
- **`A = model.recurrent.injection.get_A()` must always have `max() < 1`.** This is a correctness check; `example.py` prints it. If you refactor `LTIInjection`, verify this still holds for any learned parameter value (including after clamping). The clamp at `main.py:683` exists specifically to keep the product finite under extreme gradient steps.
- **`e` (Prelude output) is frozen across the recurrent loop** — it's the input injection. Don't accidentally make it a loop-local variable.
- **MLA caches `c_kv` (compressed) + `k_rope`** — NOT full K/V. RoPE is applied to `k_rope` before caching. See `main.py:351-362`.
- **Loop KV cache uses distinct keys per iteration**: `recurrent_loop_0`, `recurrent_loop_1`, ... (see `main.py:818`). Prelude/Coda use `prelude_{i}`, `coda_{i}`.
- **Weight tying**: `self.head.weight = self.embed.weight` at `main.py:907`. Don't reinitialize the head.
- **RoPE buffers**: two are registered — `freqs_cis` (for GQA, full head_dim) and `freqs_cis_mla` (qk_rope_head_dim only). The correct one is picked at runtime based on `cfg.attn_type`.

---

## File layout

```
open_mythos/
├── __init__.py         # public API surface (see below)
├── main.py             # the entire model — everything important is here
├── variants.py         # preconfigured MythosConfigs: mythos_1b through mythos_1t
├── tokenizer.py        # thin HF AutoTokenizer wrapper; default model_id = "openai/gpt-oss-20b"
└── moda.py             # ⚠ alternative Mixture-of-Depths + DeepSeek-MoE implementation;
                        #   NOT imported from __init__.py, standalone experimental module.
                        #   Has its own MoDAConfig. Don't confuse it with main.py.

training/
└── 3b_fine_web_edu.py  # FSDP + AdamW on HuggingFaceFW/fineweb-edu sample-10BT (default)

tests/
└── test_tokenizer.py   # pytest, loads gpt-oss-20b tokenizer from HF (requires network)

test_main.py            # ⚠ AT ROOT (not inside tests/). pytest suite for all model classes.
example.py              # smoke-test demo: build tiny model, forward, generate, print ρ(A)
variants_example.py     # even smaller demo: instantiate mythos_1b and count params

docs/
├── open_mythos.md      # long-form API reference for the OpenMythos class
└── datasets.md         # recommended training corpora + token budgets per variant
```

### `open_mythos/__init__.py` exports
Public API: `MythosConfig`, `OpenMythos`, every named class in `main.py`, the RoPE helpers, `MythosTokenizer`, and the `mythos_*` variant factories. If you add a public class to `main.py`, also add it to `__init__.py`'s imports and `__all__`.

---

## Commands

```bash
# install (poetry-based; also works with pip)
pip install -e .

# run the tiny smoke test end-to-end on CPU
python example.py

# full model-component test suite (CPU-only, tiny configs)
pytest test_main.py -v

# tokenizer tests (requires HF Hub network access to fetch gpt-oss-20b)
pytest tests/test_tokenizer.py -v

# combined
pytest -v

# format + lint (both configured at 88 cols, target py310)
black .
ruff check .

# single-GPU training (3B model, FineWeb-Edu 10BT sample)
python training/3b_fine_web_edu.py

# multi-GPU (auto-detect CUDA device count, FSDP)
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    training/3b_fine_web_edu.py
```

Python ≥ 3.10, < 4.0. Deps pinned in `pyproject.toml`: **torch 2.11.0**, transformers ≥ 4.40, datasets ≥ 2.18. `requirements.txt` has looser constraints (torch ≥ 2.1) — `pyproject.toml` is authoritative for packaging.

---

## Conventions

- **Docstrings are dense and load-bearing.** Every class and most functions in `main.py` have multi-paragraph docstrings explaining the math and design rationale (e.g. `MLAttention` docstring describes caching strategy; `LTIInjection` docstring explains the stability proof). **Preserve this style** when adding code. The rationale is often non-obvious and the codebase is pedagogical.
- **One file, many classes.** `main.py` is intentionally a single ~1000-line file so the architecture can be read top-to-bottom. Don't split it into submodules without a strong reason — it breaks the reading flow.
- **Type hints** on every public signature. `Optional` from `typing`, not `X | None`, to stay consistent with existing code.
- **`attn_type: str` is a config string, not an enum.** Valid values: `"gqa"` | `"mla"`. Branching uses `==` checks. Don't reach for an Enum rewrite unless asked.
- **Section dividers** — `# --- ... ---` comment bars separate logical sections in `main.py`. Keep them when adding new classes.
- **Formatter**: black @ 88 cols, py310 target. Ruff @ 88 cols. Both configured in `pyproject.toml`.
- **Tests live in two places**: `test_main.py` at root (legacy layout), `tests/test_tokenizer.py` in the conventional subdirectory. Don't move `test_main.py` without updating CI or the user's workflow.

---

## Known inconsistencies / gotchas

- **README references `mythos_7b()` (line ~117) which does not exist.** Only `1b / 3b / 10b / 50b / 100b / 500b / 1t` are defined in `variants.py`. Don't cite `mythos_7b` in code; either fix the README or add the variant if the user asks.
- **README's "Training" table lists Muon as the optimizer, but `training/3b_fine_web_edu.py` uses `torch.optim.AdamW`.** The most recent commit (`537b116`) explicitly says "just use adam for now in training maybe add muon later." README is aspirational on this point.
- **`moda.py` is dead-ish code** — a standalone alternative architecture (Mixture-of-Depths Attention + DeepSeek-MoE, ~44KB). It is **not** imported by `__init__.py` and has no tests. When a user says "the model" they mean `main.py`. Do not edit `moda.py` unless the user explicitly references MoDA.
- **`requirements.txt` vs `pyproject.toml` drift.** `requirements.txt` is a minimal dev list; `pyproject.toml` has the real pinned versions. Keep them consistent when changing deps.
- **Tokenizer tests require network** (downloads `openai/gpt-oss-20b` from HF Hub on first run). They will fail offline. The default `vocab_size` in `MythosConfig` (32000) does not match the actual gpt-oss-20b vocab — training script overrides `cfg.vocab_size = vocab_size` before instantiating the model (`training/3b_fine_web_edu.py:154`).
- **MoE routing is not batched across experts** — inner loop is `for eid in range(self.n_experts)` at `main.py:486`, with a per-expert boolean mask. Correct but slow for large `n_experts`. Don't "optimize" it without benchmarking; grouped GEMM is a real refactor.
- **`MoEFFN.router_bias` is a buffer, not a parameter.** The DeepSeek load-balancing strategy updates it out-of-band during training. It is not currently updated anywhere in the training script — that's a known TODO, not a bug to fix unprompted.

---

## When asked to change the architecture

1. Edit `open_mythos/main.py` in place.
2. Extend `test_main.py` with matching tests (tiny configs, CPU-only, `B=2, T=8` pattern).
3. If the change alters a public class or adds one, update `open_mythos/__init__.py` exports.
4. Run `python example.py` as a smoke test — it prints `ρ(A).max()`, parameter count, forward shape, and generate shape. If any of these regress, stop.
5. Run `pytest test_main.py -v` before declaring done.
6. Match the existing docstring density. Explain the *why*, not just the *what*.

## When asked about a variant

`variants.py` is just a set of `MythosConfig` factory functions. Editing one is low-risk; the model code doesn't special-case variants. Parameter counts in the README table are approximate — verify with `sum(p.numel() for p in OpenMythos(cfg).parameters())`.

## When asked to train

Real training requires CUDA. `training/3b_fine_web_edu.py` is the only training entry point. It:
- auto-detects DDP vs single-GPU from `RANK` env var
- wraps FSDP with `ModuleWrapPolicy({TransformerBlock, RecurrentBlock})`
- uses bfloat16 on A100/H100, falls back to float16 via autocast on older GPUs (no GradScaler wired up despite README's claim)
- streams FineWeb-Edu shards per-rank-per-worker; checkpoints every 1000 steps to `./checkpoints/`
- target: 30B tokens, ~2000-step linear warmup → cosine decay, lr 3e-4, wd 0.1

Changes to training should preserve FSDP compatibility (i.e. don't introduce modules that FSDP can't wrap).
