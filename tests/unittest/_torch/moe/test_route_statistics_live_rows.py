# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The adaptive EPLB statistics ops count only the ids of the live rows of a live-row word
(R real of P requests in bits 0-15 / 16-31; tensorrt_llm/_torch/moe/live_rows.py)."""

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference
from _torch.moe.test_adaptive_eplb_kernels import TOP_K, _dsv4_case, _stats, _tables, needs_gpu

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.runtime import build_route_tables


@needs_gpu
@pytest.mark.parametrize(
    "real,requests,live",
    # 32 rows: R of 8 requests keep 4 R rows; no request or 7 requests (not dividing 32) keep 0 or 32.
    [(8, 8, 32), (5, 8, 20), (0, 8, 0), (0, 0, 0), (3, 7, 32)],
)
def test_statistics_count_the_live_rows(real, requests, live):
    layout, shares, ids = _dsv4_case(2)
    tables = _tables(*build_route_tables(layout, shares, 384, 512))
    ids = torch.tensor(ids[: 32 * TOP_K], dtype=torch.int32, device="cuda").view(32, TOP_K)
    word = torch.tensor([(requests << 16) | real], dtype=torch.int32, device="cuda")
    routed, recorded, expected = _stats(384), _stats(384), _stats(384)
    out = torch.empty_like(ids)
    torch.ops.trtllm.adaptive_eplb_route_into(ids, *tables, 3, out, None, None, *routed, True, word)
    torch.ops.trtllm.adaptive_eplb_record(ids, *recorded, True, word)
    reference.record(ids[:live], *expected)
    for stats in (routed, recorded):
        assert torch.equal(stats.rows.view(torch.int32), expected.rows.view(torch.int32))
        assert torch.equal(stats.forwards, expected.forwards)
