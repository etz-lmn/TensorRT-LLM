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
"""CPU contract test for how the MegaMoE NVLink routers publish per-expert counts.

Each router CTA writes its per-expert counts to shared memory with ordinary stores
and then copies them to every peer with a bulk async copy, which reads shared
memory through the async proxy. Every writing thread must therefore order its
stores against the async proxy (``fence.proxy.async.shared::cta``) and the CTA must
synchronize before the elected thread issues the copy; without the fence a peer
can receive counts that miss some of the writes.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_NVLINK_DOMAIN = (
    _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain"
)
_PUBLISH = "_broadcast_sizes_to_peers"
_PREAMBLE = [
    "cute.arch.fence_proxy('async.shared', space='cta')",
    "cute.arch.sync_threads()",
]


def _publishes_counts(statement: ast.stmt) -> bool:
    value = getattr(statement, "value", None)
    return (
        isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == _PUBLISH
    )


def _publication_preambles(source: Path) -> list[tuple[int, list[str]]]:
    """Each statement that publishes counts, with the two statements before it."""
    sites = []
    for node in ast.walk(ast.parse(source.read_text())):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if _publishes_counts(statement):
                    preamble = [ast.unparse(s) for s in body[max(0, index - 2) : index]]
                    sites.append((statement.lineno, preamble))
    return sites


@pytest.mark.parametrize(
    "source", ["token_comm.py", "token_comm_dedup.py", "token_comm_deterministic.py"]
)
def test_router_counts_reach_the_async_proxy_before_publication(source: str) -> None:
    sites = _publication_preambles(_NVLINK_DOMAIN / source)

    # The single-CTA router publishes from its data CTA, the multi-CTA one from its helper.
    assert len(sites) == 2
    for lineno, preamble in sites:
        assert preamble == _PREAMBLE, f"{source}:{lineno} publishes counts without the fence"
