# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive EPLB statistics ops count only the ids of the live rows of a live-row word
(R real of P requests in bits 0-15 / 16-31; tensorrt_llm/_torch/moe/live_rows.py)."""

import ast
import functools
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference
from _torch.moe.test_adaptive_eplb_kernels import (
    TOP_K,
    _dsv4_case,
    _route,
    _stats,
    _tables,
    needs_gpu,
)

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer, policy
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import build_route_tables


@pytest.mark.cpu_only
def test_the_balancer_hands_both_ops_the_live_row_word(monkeypatch):
    trees = [ast.parse(inspect.getsource(module)) for module in (balancer, policy)]
    calls = [n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.Call)]
    last = {ast.unparse(c.func): ast.unparse(c.args[-1]) for c in calls if c.args}
    assert last["torch.ops.trtllm.adaptive_eplb_record"] == "live_row_word"
    route = next(c for c in calls if ast.unparse(c.func) == "self._route_into")
    assert ast.unparse(route.args[-2]) == "self._live_rows(ids)"
    records = [c for c in calls if ast.unparse(c.func).endswith("statistics.record")]
    assert records and all(
        ast.unparse(c.args[-1]) == "self._live_rows(pending[0])" for c in records
    )
    # The statistics take the word only from a kernel that reads it.
    word = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(balancer, "current_word", lambda device: word)
    layer = object.__new__(balancer.AdaptiveMoeLayerLoadBalancer)
    ids = SimpleNamespace(device=torch.device("cuda", 0), is_cuda=True)
    assert layer._live_rows(ids) is None
    layer.counts_live_rows = True
    assert layer._live_rows(ids) is word


@pytest.mark.cpu_only
def test_megamoe_marks_its_balancer_by_what_its_kernels_read():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import megamoe_kernel_classes

    reads = {
        sm: [getattr(kernel, "reads_live_row_word", False) for kernel in megamoe_kernel_classes(sm)]
        for sm in (100, 103, 107)
    }
    assert reads == {100: [True], 103: [True], 107: [False, False]}
    source = (Path(balancer.__file__).parents[1] / "mega_moe" / "mega_moe_cute_dsl.py").read_text()
    marks = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["self.layer_load_balancer.counts_live_rows"]
    ]
    assert len(marks) == 1
    assert "megamoe_kernel_classes(get_sm_version())" in ast.unparse(marks[0].value)


@needs_gpu
@pytest.mark.parametrize(
    "real,requests,live",
    # 32 rows: R of 8 requests keep 4 R rows; no request or 7 requests (not dividing 32) keep 0 or 32.
    [(8, 8, 32), (5, 8, 20), (0, 8, 0), (0, 0, 0), (3, 7, 32)],
)
def test_the_route_in_the_input_staging_counts_the_live_rows(monkeypatch, real, requests, live):
    """The record op and the route run_moe launches with the MegaMoE input staging
    (``MoERunContext.route``), replayed from a CUDA graph, count the live rows' ids only. That launch
    also stages the slots and gate weights and quantizes the activation as the route and
    the quantization launched apart do."""
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl as Impl
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

    layout, shares, ids = _dsv4_case(2)
    route_tables = build_route_tables(layout, shares, 384, 512)
    word = torch.tensor([(requests << 16) | real], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(balancer, "current_word", lambda device: word)
    config = MoeAdaptiveLoadBalancerConfig(num_slots=512)
    config.setup(ep_rank=3, ep_size=32)
    manager = balancer.AdaptiveMoeLoadBalancer(ep_rank=3, ep_size=32, config=config, mapping=None)
    layer = manager.add_layer(384, TOP_K, 16)
    layer.counts_live_rows = True
    staging = SimpleNamespace(activation=torch.zeros(64, 3584, dtype=torch.uint8, device="cuda"))
    staging.activation_sf = torch.zeros(64, 448, dtype=torch.uint8, device="cuda")
    staging.topk_idx_local = torch.zeros(64, TOP_K, dtype=torch.int64, device="cuda")
    staging.topk_weights = torch.ones(64, TOP_K, device="cuda")
    scale = torch.tensor([99.0], device="cuda")
    moe = SimpleNamespace(fc31_input_scale=scale, _staging_of_next_run=lambda n, k: (staging, 64))
    moe._activation_to_quantize = Impl._activation_to_quantize.__get__(moe)
    layer.prepare_device_state(torch.int64, "cuda", Impl.route_staging.__get__(moe))
    manager.route_tables("cuda")[0].copy_(torch.tensor(route_tables.packed()))
    manager.observe_scheduled_batch(8, True, False, lambda start, stop: (0, 0, 0))
    x = torch.zeros(32, 7168, dtype=torch.bfloat16, device="cuda")
    routes = torch.zeros(32, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.zeros(32, TOP_K, device="cuda")
    route = functools.partial(layer.route, routes, False, token_final_scales=weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.update_local_statistic(routes, True, True)
        got = Impl._stage_bf16_inputs(moe, x, routes, weights, route)
    assert [t.data_ptr() for t in got] == [t.data_ptr() for t in vars(staging).values()][:3]
    x.copy_(torch.randn(32, 7168))
    routes.copy_(torch.tensor(ids[: 32 * TOP_K]).view(32, TOP_K))
    weights.copy_(torch.rand(32, TOP_K))
    graph.replay()
    recorded, expected = _stats(384), _stats(384)
    torch.ops.trtllm.adaptive_eplb_record(routes, *recorded, True, word)
    reference.record(routes[:live], *expected)
    for stats in (layer.statistics, recorded):
        assert torch.equal(stats.rows.view(torch.int32), expected.rows.view(torch.int32))
        assert torch.equal(stats.forwards, expected.forwards)
    slots = torch.full((64, TOP_K), -1, device="cuda", dtype=torch.int64)
    slots[:32] = _route(routes, _tables(*route_tables), 3, torch.int64)
    assert torch.equal(staging.topk_idx_local, slots)
    assert torch.equal(staging.topk_weights[:32], weights) and not staging.topk_weights[32:].any()
    want = [torch.empty_like(t[:32]) for t in (staging.activation, staging.activation_sf)]
    torch.ops.trtllm.fp4_quantize_into(x, scale, *want, False)
    assert torch.equal(staging.activation[:32], want[0])
    assert torch.equal(staging.activation_sf[:32], want[1])
