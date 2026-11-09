# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MegaMoE CuteDSL backend's route staging."""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


def test_route_staging_hands_out_the_staging_of_a_run_that_stages():
    """``route_staging`` (routes) and ``_staging_of_next_run`` hand out a bucket's staging only for
    a forward that ``run_moe`` would stage."""
    t = torch.zeros(4, 2)
    idx = torch.full((4, 2), -1, dtype=torch.int64)
    bufs = mm._MegaMoeBuffers(
        activation=t,
        activation_sf=t,
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=idx,
    )
    acquired = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        tactic_autotune=False,
        _weights_created=True,
        ep_size=4,
        _genphase_direct_input_buckets={4},
        dtype=torch.bfloat16,
        hidden_size=4,
        _local_staging_cache={},
        _select_launch_max_tokens=lambda hint: 4 if hint <= 4 else 8,
        _acquire_buffers=lambda **kw: acquired.append(kw) or bufs,
    )
    backend._staging_of_next_run = types.MethodType(IMPL._staging_of_next_run, backend)
    staging = IMPL.route_staging
    assert staging(backend, 2, 2) is None  # no bucket hint yet
    backend._active_launch_max_tokens = 3
    assert staging(backend, 2, 2) is None  # the staging is not built
    backend._local_staging_cache[(4, 2, 4, "cpu", torch.bfloat16)] = {}
    got = staging(backend, 2, 2)
    assert got.topk_idx is idx and got.topk_weights is t and got.bucket == 4
    assert staging(backend, 2, 3) is None  # no staging built for that top_k
    assert acquired[-1]["launch_max_T"] == 4 and acquired[-1]["device"] == torch.device("cpu")
    for rows in (4, 5):  # an exact-size direct-input bucket; the next bucket is not built
        assert staging(backend, rows, 2) is None
