# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime of adaptive EPLB: the route tables that map every expert to its slots, and the shares
that split an expert's rows over its copies."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import torch

from .policy import ExpertGeometry

_UINT32_MAX = (1 << 32) - 1


def solve_shares(load_rows: np.ndarray, layout: np.ndarray, geometry: ExpertGeometry) -> np.ndarray:
    """``[num_slots]`` shares of ``layout`` that minimize the busiest GPU's ``load_rows``."""
    load = np.ascontiguousarray(load_rows, dtype=np.float64)
    slots = np.asarray(layout)
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("layout must hold integers")
    slots = torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int64))
    solve = torch.ops.trtllm.adaptive_eplb_solve_shares
    return solve(torch.from_numpy(load), slots, geometry.ep_size).numpy()


class RouteTables(NamedTuple):
    """Route tables of one layout: copies (``counts``) and first table entry (``starts``) of every
    expert, the slot of every entry, and the Q32 cumulative shares ``cumulative_shares`` of replicated experts."""

    counts: list[int]
    starts: list[int]
    slots: list[int]
    cumulative_shares: list[int]

    def packed(self) -> list[int]:
        """``counts | starts | slots | cumulative_shares`` as int32 values."""
        cumulative_shares = [
            value - (1 << 32) if value >> 31 else value for value in self.cumulative_shares
        ]
        return self.counts + self.starts + self.slots + cumulative_shares


def build_route_tables(
    layout: Sequence[int], shares: Sequence[float] | None, num_experts: int, num_slots: int
) -> RouteTables:
    """Route tables of a ``[num_slots]`` layout and the ``shares`` of its slots; without
    ``shares``, every expert's rows split evenly over its copies."""
    layout = [int(e) for e in (layout.tolist() if hasattr(layout, "tolist") else layout)]
    shares = [1.0] * num_slots if shares is None else shares
    shares = [float(s) for s in (shares.tolist() if hasattr(shares, "tolist") else shares)]
    expert_slots: list[list[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(layout):
        if expert >= 0:
            expert_slots[expert].append(slot)
    counts = [len(copies) for copies in expert_slots]
    if 0 in counts:
        raise ValueError(f"expert {counts.index(0)} has no copy")
    starts = list(itertools.accumulate(counts, initial=0))[:-1]
    slots = [slot for copies in expert_slots for slot in copies] + [-1] * layout.count(-1)
    cumulative_shares = [_UINT32_MAX] * num_slots
    for copies, start in zip(expert_slots, starts):
        weights = [max(0.0, shares[slot]) for slot in copies]
        total = sum(weights)
        if total <= 0.0:
            weights, total = [1.0] * len(copies), float(len(copies))
        acc = 0.0
        bound = 0
        for k in range(len(copies) - 1):
            acc += weights[k]
            bound = min(max(int(acc / total * 4294967296.0 + 0.5), bound), _UINT32_MAX)
            cumulative_shares[start + k] = bound
    return RouteTables(counts, starts, slots, cumulative_shares)
