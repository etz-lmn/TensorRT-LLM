# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 route-only launch on whole router logits against routing.apply, bit for bit.

An adaptive-EPLB layer routes whole logits (the small-batch router GEMM's, 64 tokens per rank
and below) through the route-only launch the partial logits use, as the FP32 contiguous rows
``apply`` reads (``MoEScheduler._route_only``). The experts and weights must be ``apply``'s.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (10, 0)
    or not IS_CUTLASS_DSL_AVAILABLE,
    reason="Requires Blackwell or newer and the CuTe DSL",
)

# DeepSeek-V4: 384 experts, top-6; the hidden size only keys the route-only kernel's build.
NUM_EXPERTS, TOP_K, HIDDEN, SCALING = 384, 6, 7168, 2.5
SMALL_BATCHES = [1, 4, 8, 16, 32, 64]


def _routing(bias):
    return DeepSeekV4MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=SCALING,
        callable_e_score_correction_bias=lambda: bias,
        callable_tid2eid=lambda: None,
        is_hashed=False,
    )


def _inputs(num_tokens, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", generator=generator) * 4
    bias = torch.randn(NUM_EXPERTS, device="cuda", generator=generator) * 0.5
    return logits, bias


def _route_only(logits, bias):
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import dsv4_route

    ids, weights, tables = dsv4_route(logits, bias, TOP_K, SCALING, torch.float32, HIDDEN)
    assert tables is None
    return ids, weights


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and got.shape == want.shape
        torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("num_tokens", SMALL_BATCHES)
@torch.inference_mode()
def test_route_only_whole_logits_is_bitwise_to_apply(num_tokens):
    logits, bias = _inputs(num_tokens)
    _assert_equal(_route_only(logits, bias), _routing(bias).apply(logits))


@pytest.mark.parametrize("num_tokens", SMALL_BATCHES)
@torch.inference_mode()
def test_bf16_logits_converted_first_match_apply(num_tokens):
    logits, bias = _inputs(num_tokens, seed=1)
    logits = logits.bfloat16()
    # apply casts to FP32 itself; the scheduler hands the route-only launch the same FP32 rows.
    got = _route_only(logits.to(torch.float32).contiguous(), bias)
    _assert_equal(got, _routing(bias).apply(logits))


@pytest.mark.parametrize("num_tokens", [8, 64])
@torch.inference_mode()
def test_route_only_whole_logits_ties_nans_and_zero_groups(num_tokens):
    logits, bias = _inputs(num_tokens, seed=2)
    # Ties, NaN and infinite logits, a group of equal biased scores and an all-zero row.
    logits[0] = 1.0
    logits[1, 5] = float("nan")
    logits[2, :7] = float("inf")
    bias[100:110] = 0.25
    logits[3, 100:110] = 3.0
    logits[4] = 0.0
    _assert_equal(_route_only(logits, bias), _routing(bias).apply(logits))


@pytest.mark.parametrize("num_tokens", [4, 64])
@pytest.mark.parametrize("view", ["columns", "transpose"])
@torch.inference_mode()
def test_route_only_whole_logits_from_a_strided_view(num_tokens, view):
    logits, bias = _inputs(num_tokens, seed=3)
    if view == "columns":
        wide = torch.randn(num_tokens, 2 * NUM_EXPERTS, device="cuda")
        wide[:, :NUM_EXPERTS] = logits
        strided = wide[:, :NUM_EXPERTS]
    else:
        strided = logits.t().contiguous().t()
    assert not strided.is_contiguous()
    got = _route_only(strided.to(torch.float32).contiguous(), bias)
    _assert_equal(got, _routing(bias).apply(strided))


@pytest.mark.parametrize("num_tokens", SMALL_BATCHES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_scheduler_routes_whole_logits_as_apply_does(num_tokens, dtype):
    """The scheduler's route-only selection and the MegaMoE backend's route-only wrapper, end to end."""
    from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import (
        TrtllmCutedslMegaMoeNvfp4Impl,
    )
    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import MoEScheduler

    class Probe(MoEScheduler):
        def forward(self, *args, **kwargs):
            raise NotImplementedError

    logits, bias = _inputs(num_tokens, seed=4)
    logits = logits.to(dtype)
    routing = _routing(bias)
    backend_self = SimpleNamespace(routing_method=routing, hidden_size=HIDDEN)
    backend = SimpleNamespace(
        fused_route_quant_enabled=True,
        takes_router_logit_partials=lambda: True,
        route_from_logit_partials=functools.partial(
            TrtllmCutedslMegaMoeNvfp4Impl.route_from_logit_partials, backend_self
        ),
    )
    moe = SimpleNamespace(
        backend=backend,
        apply_router_weight_on_input=False,
        _using_dynamic_load_balancer=lambda: False,
        _load_balancer_uses_replica_plan=lambda: False,
        layer_load_balancer=SimpleNamespace(stages_routes=True),
    )
    routes = Probe(moe)._route_only(logits, None)
    assert routes is not None
    _assert_equal(routes, routing.apply(logits))


@pytest.mark.parametrize("num_tokens", [4, 32, 64])
@torch.inference_mode()
def test_route_only_after_a_router_gemm_replays_in_a_cuda_graph(num_tokens):
    """In one graph, a router GEMM writes the logits and the route-only launch (a programmatic
    dependent when PDL is on) reads them: every replay routes the GEMM's finished logits."""
    _, bias = _inputs(num_tokens, seed=5)
    generator = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", generator=generator)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=generator) / HIDDEN**0.5
    routing = _routing(bias)
    _route_only(x @ weight.t(), bias)  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = x @ weight.t()
        ids, weights = _route_only(logits, bias)
    for replay in range(5):
        x.copy_(torch.randn(x.shape, device="cuda", generator=generator) * (1 + replay))
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal((ids, weights), routing.apply(logits))
