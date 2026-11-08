# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python versions of the adaptive EPLB route and statistics ops (``adaptiveEplbKernels.h``): the
reference of the kernel tests, and the ops of the CPU tests."""

from collections import Counter

import numpy as np
import torch

SPLIT_BLOCK = 1024
BLOCK_MULTIPLIER = 0x9E3779B9
EXPERT_MULTIPLIER = 0x632BE5AB
BLOCKS_PER_RANK = 64
UINT32_MAX = (1 << 32) - 1


def route(ids, counts, starts, slots, cumulative_shares, ep_rank) -> list:
    """The slot of every flat expert id; ``cumulative_shares`` as unsigned 32-bit values."""
    num_experts, num_slots = len(counts), len(slots)
    out = []
    for base in range(0, len(ids), SPLIT_BLOCK):
        block = base // SPLIT_BLOCK
        experts = [
            e if 0 <= e < num_experts else num_experts for e in ids[base : base + SPLIT_BLOCK]
        ]
        spread = [e if e < num_experts and counts[e] > 1 else num_experts for e in experts]
        total = Counter(spread)
        seen = Counter()
        for expert, spread_id in zip(experts, spread):
            out.append(num_slots)
            if expert == num_experts:
                continue
            copies, start = counts[expert], starts[expert]
            if copies <= 0 or start < 0 or start + copies > num_slots:
                continue
            k = 0
            if spread_id < num_experts:
                n = total[spread_id]
                step = UINT32_MAX if n <= 1 else (1 << 32) // n
                point = (
                    expert * EXPERT_MULTIPLIER
                    + (ep_rank * BLOCKS_PER_RANK + block) * BLOCK_MULTIPLIER
                )
                point = (point + seen[spread_id] * step) & UINT32_MAX
                seen[spread_id] += 1
                while k < copies - 1 and point >= cumulative_shares[start + k]:
                    k += 1
            out[-1] = slots[start + k]
    return out


def record(ids, ema, forwards, enabled, decay, first=True) -> None:
    """Add the expert ids to ``ema`` / ``forwards`` in place, rounding to float32 after the
    multiply and after the add."""
    if int(enabled[0]) == 0:
        return
    flat = ids.reshape(-1).to(torch.int64).cpu().numpy()
    valid = flat[(flat >= 0) & (flat < ema.numel())]
    counts = np.bincount(valid, minlength=ema.numel()).astype(np.float32)
    values = ema.cpu().numpy().astype(np.float32)
    if first:
        values = values * np.float32(decay)
        forwards += 1
    ema.copy_(torch.from_numpy(values + counts))


def route_into(
    ids,
    counts,
    starts,
    slots,
    cumulative_shares,
    ep_rank,
    out,
    weights=None,
    weights_out=None,
    ema=None,
    forwards=None,
    enabled=None,
    decay=0.0,
    first=True,
):
    """``torch.ops.trtllm.adaptive_eplb_route_into`` on CPU tensors."""
    tables = [t.tolist() for t in (counts, starts, slots)]
    shares = [value & UINT32_MAX for value in cumulative_shares.tolist()]
    routed = route(ids.reshape(-1).tolist(), *tables, shares, ep_rank)
    flat = out.view(-1)
    flat[: len(routed)] = torch.tensor(routed, dtype=out.dtype)
    if weights is not None:
        top_k = ids.shape[-1]
        flat[len(routed) :] = -1
        weights_out[: ids.shape[0], :top_k] = weights
        weights_out[ids.shape[0] : flat.numel() // top_k, :top_k] = 0.0
    if ema is not None:
        record(ids, ema, forwards, enabled, decay, first)
