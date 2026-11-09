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
"""bench_moe's supplied routing plan answers every routing call the MoE scheduler makes.

Routing control (``routing/native_logits.py``, ``_maybe_install_routing_control_patch``)
replaces the MoE's routing method with the materialised plan. The scheduler calls
``routing_method.apply(router_logits, input_ids)`` and, with auxiliary routing data,
``routing_method.apply_with_aux(router_logits, input_ids, routing_aux)``. A replacement that
took the logits alone raised ``TypeError`` at the first forward. Each such call found in
``moe_scheduler.py`` is made here as written there (its positional and keyword arguments)
against the installed patch, and must return the plan.

The patch functions are compiled from ``native_logits.py`` without importing tensorrt_llm,
so the test runs on CPU with torch alone.
"""

import ast
import contextlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT / "tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py"
NATIVE_LOGITS = Path(__file__).resolve().parent / "routing" / "native_logits.py"
PATCH = ("_align_topk_to_batch", "_make_supplied_topk_apply", "_maybe_install_routing_control_patch")


class _OtherBackend:
    """Stands for TRTLLMGenFusedMoE: a MoE of another backend gets the routing patch alone."""


class _Routing:
    def apply(self, router_logits, input_ids):
        raise AssertionError("the model's routing ran instead of the supplied plan")

    def apply_with_aux(self, router_logits, input_ids, routing_aux):
        raise AssertionError("the model's routing ran instead of the supplied plan")


class _Moe:
    def __init__(self):
        self.routing_method = _Routing()

    def forward_impl(self, *args, **kwargs):
        return None


def _patch_functions():
    tree = ast.parse(NATIVE_LOGITS.read_text())
    keep = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in PATCH]
    assert sorted(node.name for node in keep) == sorted(PATCH)
    namespace = dict(torch=torch, contextlib=contextlib, Dict=Dict, Optional=Optional, Tuple=Tuple,
                     TRTLLMGenFusedMoE=_OtherBackend,
                     make_forward_impl_check=lambda moe, forward_impl: forward_impl)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(NATIVE_LOGITS), "exec"), namespace)
    return namespace


def _routing_calls():
    calls = []
    for node in ast.walk(ast.parse(SCHEDULER.read_text())):
        func = node.func if isinstance(node, ast.Call) else None
        if (isinstance(func, ast.Attribute) and func.attr in ("apply", "apply_with_aux")
                and isinstance(func.value, ast.Attribute) and func.value.attr == "routing_method"):
            calls.append((func.attr, len(node.args), tuple(sorted(k.arg for k in node.keywords))))
    return calls


def test_every_scheduler_routing_call_returns_the_supplied_plan():
    calls = _routing_calls()
    assert {name for name, _, _ in calls} == {"apply", "apply_with_aux"}, calls
    install = _patch_functions()["_maybe_install_routing_control_patch"]
    ids = torch.tensor([[0, 5], [3, 1], [2, 7]])
    scales = torch.full((3, 2), 0.5)
    moe = _Moe()
    with install(moe, ids, scales, active=True):
        for name, positional, keywords in calls:
            args = [torch.zeros(3, 8)] + [torch.zeros(3, dtype=torch.int64)] * (positional - 1)
            got_ids, got_scales = getattr(moe.routing_method, name)(*args, **{k: None for k in keywords})
            assert got_ids.dtype == torch.int32 and torch.equal(got_ids, ids.to(torch.int32)), name
            assert torch.equal(got_scales, scales), name
    # restored: the model's own routing again
    for name in ("apply", "apply_with_aux"):
        assert getattr(moe.routing_method, name).__func__ is getattr(_Routing, name), name
