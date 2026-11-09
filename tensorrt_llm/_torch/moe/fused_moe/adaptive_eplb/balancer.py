# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Statistics of the adaptive EPLB load balancer's MoE layers, in contiguous device buffers that the
route and statistics ops update inside CUDA graphs."""

from __future__ import annotations

from typing import NamedTuple

import torch


def statistics_buffers(layers: int, num_experts: int, device: torch.device | str) -> dict:
    """Every layer's statistics in one contiguous ``storage`` of bytes, so that a window opens with
    one zeroing and closes with one copy: ``rows`` ``[L, E]`` float32 (the rows routed to every
    expert, exact below 2^24), then ``forwards`` ``[L, 1]`` int64 from the next 8-byte boundary."""
    rows_bytes = layers * num_experts * 4
    forwards_at = -(-rows_bytes // 8) * 8
    storage = torch.zeros(forwards_at + layers * 8, dtype=torch.uint8, device=device)
    return dict(
        rows=storage[:rows_bytes].view(torch.float32).view(layers, num_experts),
        forwards=storage[forwards_at:].view(torch.int64).view(layers, 1),
        storage=storage,
    )


class ExpertStatistics(NamedTuple):
    """One layer's statistics in the ops' argument order, and the ``[1]`` flag that the ops read at
    every launch, so that a captured CUDA graph records statistics only while the flag is set."""

    rows: torch.Tensor
    forwards: torch.Tensor
    enabled: torch.Tensor

    @classmethod
    def of(cls, buffers: dict, layer: int, enabled: torch.Tensor) -> ExpertStatistics:
        """The statistics of layer ``layer`` in buffers made by :func:`statistics_buffers`."""
        return cls(buffers["rows"][layer], buffers["forwards"][layer], enabled)

    def record(self, token_selected_experts: torch.Tensor, first_call: bool = True) -> None:
        """Add ``[tokens, top_k]`` expert ids to the statistics; only a forward's first call counts
        the forward."""
        torch.ops.trtllm.adaptive_eplb_record(token_selected_experts, *self, first_call)
