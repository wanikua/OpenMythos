#!/usr/bin/env python3
"""
Multi-agent OpenMythos pretraining on FineWeb-Edu with FSDP + AdamW.

This scaffold is NOT auto-launched. Run manually when cloud compute is
authorized. The script does not call any cloud training API on its own —
it uses Hugging Face `datasets` streaming (`HuggingFaceFW/fineweb-edu`)
for data and launches via `python` or `torchrun` on an already-provisioned
machine.

Single GPU:
    python training/multi_agent_pretrain.py

Multi-GPU:
    torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \\
        training/multi_agent_pretrain.py

Differences from `3b_fine_web_edu.py`:
    1. `MultiAgentMythos` replaces `OpenMythos`. The FSDP auto-wrap policy
       adds `MultiAgentRecurrentBlock` so the multi-agent loop body is
       sharded like the single-agent RecurrentBlock.
    2. The forward pass returns `(logits, info)`; the trainer adds
       `cfg.vq_loss_weight * info["vq_loss"]` to the task loss every step.
    3. A linear BusGate warmup curriculum opens the gate from 0 → 1 over
       `bus_warmup_steps`. At step 0 the model behaves as N independent
       single agents; at the end of warmup peers are fully integrated.
    4. Every `diag_every` steps we log codebook stats (live/dead codes,
       usage rate) and the spectral radius of the LTI injection. Cheapest
       way to catch a regression on the stability guarantee or codebook
       collapse.
"""

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from open_mythos import MultiAgentMythos
from open_mythos.agents import MultiAgentRecurrentBlock
from open_mythos.main import TransformerBlock
from open_mythos.meow import snapshot_codebook_stats
from open_mythos.tokenizer import MythosTokenizer
from open_mythos.variants import multi_agent_3b


# ---------------------------------------------------------------------------
# Dataset (identical to single-agent; multi-agent is a model-only change)
# ---------------------------------------------------------------------------


class FineWebEduDataset(IterableDataset):
    def __init__(self, encoding, seq_len: int, subset: str, rank: int, world_size: int):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        total_shards = self.world_size * num_workers
        shard_index = self.rank * num_workers + worker_id

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_index)

        buf = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup, cosine decay to `min_lr`."""
    if step < warmup:
        return max_lr * step / warmup
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def get_bus_gate_value(step: int, bus_warmup_steps: int) -> float:
    """
    Linear ramp from 0 → 1 over `bus_warmup_steps`.

    Early training is noisy and peer messages would drown out task signal.
    Holding the gate at 0 gives the Prelude + per-agent LoRA a head start on
    useful token representations before peers enter the recurrence.
    """
    if bus_warmup_steps <= 0:
        return 1.0
    return min(1.0, step / bus_warmup_steps)


class TrainingAbort(RuntimeError):
    """
    Raised when a monitored invariant fails during training.

    The training loop catches this, saves a best-effort abort checkpoint,
    tears down the process group, and exits with a nonzero status. Never
    silence this — the whole point is to stop before a corrupted checkpoint
    makes it to disk.
    """


def check_loss_finite(loss: float, step: int) -> None:
    """Abort if loss is NaN/Inf. Non-finite loss at step > 0 is a divergence signal."""
    if not math.isfinite(loss):
        raise TrainingAbort(
            f"step {step}: loss is non-finite ({loss!r}). Possible divergence; "
            "check learning rate, gradient clip, and recent codebook stats."
        )


def check_spectral_radius(rho: float, step: int, bound: float = 0.9999) -> None:
    """
    Abort if ρ(A) ≥ bound. The LTI guarantee of `LTIInjection.get_A()` is
    ρ(A) ∈ (0, 1) by construction; a value at or above 1 means the clamp
    has broken and the recurrence is no longer a contraction.
    """
    if rho >= bound:
        raise TrainingAbort(
            f"step {step}: rho(A)={rho:.6f} ≥ {bound}. LTI stability violated."
        )


def check_codebook_health(
    stats: dict, step: int, max_dead_fraction: float = 0.75
) -> None:
    """
    Warn / abort on codebook collapse.

    `dead_fraction > max_dead_fraction` means the codebook has largely
    collapsed to a handful of active symbols — continuing trains against a
    degraded protocol. Dead-code replacement inside `VectorQuantizer` should
    have recovered this; if it didn't, investigate before spending more
    compute.
    """
    dead = float(stats["dead_fraction"])
    if dead > max_dead_fraction:
        raise TrainingAbort(
            f"step {step}: codebook dead_fraction={dead:.3f} > {max_dead_fraction}. "
            "Codebook collapse; inspect dead-code replacement and VQ commitment cost."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # Distributed init
    # ------------------------------------------------------------------
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank = local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    master = rank == 0

    if master:
        print(
            f"GPUs: {torch.cuda.device_count()}  |  World size: {world_size}  |  Device: {device}"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size

    if master:
        print(f"Tokenizer: gpt-oss-20b  |  Vocab size: {vocab_size:,}")

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    seq_len = 2048
    micro_batch = 4
    target_tokens = 30_000_000_000
    grad_accum = max(1, 256 // (world_size * micro_batch))
    global_batch_tok = world_size * micro_batch * grad_accum * seq_len
    total_steps = target_tokens // global_batch_tok
    warmup_steps = 2000
    bus_warmup_steps = 5000
    lr = 3e-4
    wd = 0.1
    log_every = 10
    diag_every = 200
    ckpt_every = 1000
    ckpt_dir = "checkpoints/multi_agent"
    dataset_subset = "sample-10BT"

    if master:
        print(
            f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum}\n"
            f"global_batch_tokens={global_batch_tok:,} | total_steps={total_steps:,}\n"
            f"bus_warmup_steps={bus_warmup_steps}"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = multi_agent_3b()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    model = MultiAgentMythos(cfg)

    if ddp:
        mp_policy = MixedPrecision(
            param_dtype=amp_dtype,
            reduce_dtype=amp_dtype,
            buffer_dtype=amp_dtype,
        )
        # Wrap the multi-agent recurrent block the same way the single-agent
        # trainer wraps RecurrentBlock — one FSDP unit per recurrent body.
        wrap_policy = ModuleWrapPolicy(
            {TransformerBlock, MultiAgentRecurrentBlock}
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
        )
    else:
        model = model.to(device)

    if ddp:
        amp_ctx = nullcontext()
    elif "cuda" in device:
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = nullcontext()

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}  |  AMP dtype: {amp_dtype}")
        print(
            f"Multi-agent: n_agents={cfg.n_agents} | codebook_size={cfg.meow_codebook_size} "
            f"| msg_len={cfg.meow_msg_len} | vq_loss_weight={cfg.vq_loss_weight}"
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), fused=True
    )

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset, rank, world_size)
    loader = DataLoader(
        dataset, batch_size=micro_batch, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if master:
        os.makedirs(ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()
    step = 0

    # Handle to the multi-agent recurrent block for bus-gate updates and
    # diagnostics. Under FSDP the inner `recurrent` submodule remains
    # directly accessible for this scalar-level bookkeeping.
    inner_model = model.module if hasattr(model, "module") else model
    rec_block = inner_model.recurrent

    def save_checkpoint(tag: str) -> str:
        """
        Save a full-state checkpoint at the current step.

        Used by both the regular `ckpt_every` cadence and the abort path.
        Kept as a closure so the abort handler can reuse the same FSDP
        full-state gather without duplicating the sharding logic.
        """
        path = os.path.join(ckpt_dir, f"step_{step:07d}_{tag}.pt")
        if ddp:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state = model.state_dict()
        else:
            model_state = model.state_dict()
        if master:
            torch.save(
                {
                    "step": step,
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "vocab_size": vocab_size,
                    "bus_gate_value": bus_gate_value,
                    "tag": tag,
                },
                path,
            )
        return path

    bus_gate_value = 0.0

    try:
        while step < total_steps:
            cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
            for g in optimizer.param_groups:
                g["lr"] = cur_lr

            bus_gate_value = get_bus_gate_value(step, bus_warmup_steps)
            rec_block.bus_gate.set_gate(bus_gate_value)

            optimizer.zero_grad()
            loss_accum = 0.0
            vq_loss_accum = 0.0

            for micro_step in range(grad_accum):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                sync = (
                    nullcontext()
                    if (not ddp or micro_step == grad_accum - 1)
                    else model.no_sync()
                )
                with sync, amp_ctx:
                    logits, info = model(x, return_info=True)
                    task_loss = nn.functional.cross_entropy(
                        logits.view(-1, vocab_size), y.view(-1)
                    )
                    vq_loss = info["vq_loss"]
                    loss = task_loss + cfg.vq_loss_weight * vq_loss
                    loss = loss / grad_accum

                loss.backward()
                loss_accum += float(loss.detach())
                vq_loss_accum += float(vq_loss.detach()) / grad_accum

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1

            # Kill-switch #1: divergence guard. A NaN/Inf loss at step > 0
            # means the next optimizer.step would poison the weights.
            check_loss_finite(loss_accum, step)

            if master and step % log_every == 0:
                dt = time.perf_counter() - t0
                tok_per_sec = global_batch_tok * log_every / dt
                tokens_seen = step * global_batch_tok
                print(
                    f"step {step:6d}/{total_steps} | loss {loss_accum:.4f} "
                    f"| vq {vq_loss_accum:.4f} | gate {bus_gate_value:.3f} "
                    f"| lr {cur_lr:.2e} | {tok_per_sec / 1e6:.2f}M tok/s "
                    f"| {tokens_seen / 1e9:.1f}B tokens seen"
                )
                t0 = time.perf_counter()

            if master and step % diag_every == 0:
                with torch.no_grad():
                    rho = float(rec_block.spectral_radius())
                    stats = snapshot_codebook_stats(rec_block.bus)
                print(
                    f"  [diag step {step}] rho(A)={rho:.4f} "
                    f"| live={stats['live_codes']}/{stats['codebook_size']} "
                    f"| dead_frac={stats['dead_fraction']:.3f} "
                    f"| norm_max={stats['norm_max']:.3f}"
                )
                # Kill-switch #2 and #3: stability + codebook guards. Only
                # evaluated on master since that's where the diag tensors
                # were materialized, but TrainingAbort on master will reach
                # other ranks via the process-group teardown in the except
                # block.
                check_spectral_radius(rho, step)
                check_codebook_health(stats, step)

            if master and step % ckpt_every == 0:
                path = save_checkpoint("regular")
                print(f"Checkpoint saved → {path}")

    except TrainingAbort as e:
        # Invariant violation. Save a best-effort abort checkpoint so the
        # operator can inspect the diverged state, then surface a nonzero
        # exit status. Do NOT swallow.
        if master:
            print(f"[ABORT step {step}] {e}")
            try:
                path = save_checkpoint("abort")
                print(f"Abort checkpoint saved → {path}")
            except Exception as save_err:
                print(f"[ABORT] failed to save abort checkpoint: {save_err!r}")
        if ddp:
            dist.destroy_process_group()
        raise SystemExit(1)

    if ddp:
        dist.destroy_process_group()

    if master:
        print("Training complete.")


if __name__ == "__main__":
    main()
