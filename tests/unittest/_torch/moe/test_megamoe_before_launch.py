# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the MegaMoE backends issue ``MoERunContext.before_launch``.

A fused-communication scheduler hands a backend that declares
``takes_before_launch`` the wait for the EPLB statistic allreduce instead of
issuing it before ``run_moe``. The backend must issue it after its own input
preparation, so that preparation overlaps the allreduce, and before every
launch of its fused kernel, bucket priming included, because that kernel must
not run alongside the allreduce. The real ``run_moe`` paths run here on CPU
tensors with the kernels and preparation ops replaced by recorders.
"""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as cute_dsl
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_deepgemm as deepgemm
from tensorrt_llm._torch.utils import ActivationType

DEEPGEMM = deepgemm.DeepgemmCudaW4a8Mxfp4Mxfp8Impl
CUTE_DSL = cute_dsl.TrtllmCutedslMegaMoeNvfp4Impl


def _tuning(module, monkeypatch, tuning: bool) -> None:
    monkeypatch.setattr(
        module, "AutoTuner", SimpleNamespace(get=lambda: SimpleNamespace(is_tuning_mode=tuning))
    )


@pytest.mark.cpu_only
def test_both_megamoe_backends_take_the_wait():
    assert DEEPGEMM.takes_before_launch is True
    assert CUTE_DSL.takes_before_launch is True


def _deepgemm(trace, *, primed: bool):
    num_tokens, hidden = 4, 8
    buf = SimpleNamespace(
        x=torch.zeros(num_tokens, hidden),
        x_sf=torch.zeros(num_tokens, 1),
        topk_idx=torch.zeros(num_tokens, 2, dtype=torch.int64),
        topk_weights=torch.zeros(num_tokens, 2),
        num_max_tokens_per_rank=num_tokens,
    )
    impl = SimpleNamespace(
        dtype=torch.bfloat16,
        _symm_buffer=buf,
        max_num_tokens=num_tokens,
        hidden_size=hidden,
        intermediate_size=16,
        ep_size=2,
        num_slots=8,
        routing_method=SimpleNamespace(experts_per_token=2),
        dg_activation="swiglu",
        act_clamp=None,
        act_alpha=None,
        act_beta=None,
        fast_math=True,
        _kernels_primed=primed,
        _dg=SimpleNamespace(get_block_m_for_mega_moe=lambda *args: 64),
        supports_fused_prepare=lambda: True,
        _launch_kernel=lambda output: trace.append(("launch", output.shape[0])),
    )
    impl._prime_kernels = types.MethodType(DEEPGEMM._prime_kernels, impl)
    return impl


@pytest.mark.cpu_only
@pytest.mark.parametrize("priming", [False, True], ids=["primed", "priming"])
@pytest.mark.parametrize("takes_wait", [True, False], ids=["wait", "no_wait"])
def test_deepgemm_waits_after_megamoe_prepare_and_before_each_launch(
    monkeypatch, priming, takes_wait
):
    trace = []
    _tuning(deepgemm, monkeypatch, True)
    monkeypatch.setattr(deepgemm, "_MEGA_MOE_PRIMED_TILES", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.ops.trtllm, "megamoe_prepare", lambda *args: trace.append("prepare"), raising=False
    )
    impl = _deepgemm(trace, primed=not priming)
    ctx = MoERunContext(
        token_selected_experts=torch.zeros(2, 2, dtype=torch.int32),
        token_final_scales=torch.ones(2, 2),
        x=torch.ones(2, 8, dtype=torch.bfloat16),
        x_sf=None,  # fused prepare: run_moe stages the BF16 input itself
        output_dtype=torch.bfloat16,
        before_launch=(lambda: trace.append("wait")) if takes_wait else None,
    )

    DEEPGEMM.run_moe(impl, ctx)

    wait = ["wait"] if takes_wait else []
    # The ladder primes every tile with masked routes, one launch per tile.
    prime = [*wait, ("launch", 0)] if priming else []
    assert trace == [*prime, "prepare", *wait, ("launch", 2)]


def _cute_dsl_staging(trace, wait):
    max_t, top_k = 4, 2
    bufs = SimpleNamespace(
        topk_idx_local=torch.full((max_t, top_k), -1, dtype=torch.int32),
        topk_weights=torch.zeros(max_t, top_k),
        activation=torch.empty(max_t, 4, dtype=torch.uint8),
        activation_sf=torch.empty(max_t, 1, dtype=torch.uint8),
        combine_output=None,
        shared_workspace=None,
        peer_offsets=(0, 0),
    )

    def launch(**kwargs):
        trace.append(("launch", kwargs["before_launch"] is wait))
        return "out"

    return SimpleNamespace(
        _select_launch_max_tokens=lambda hint: max_t,
        _active_launch_max_tokens=max_t,
        _shared_expert=None,
        ep_size=2,
        ep_rank=0,
        tactic_autotune=False,
        _prime_ladder=lambda **kwargs: trace.append("prime"),
        _acquire_buffers=lambda **kwargs: bufs,
        _prestaged=None,
        _quantized_into=None,
        _handed_out_route_staging=None,
        _routes_staged=lambda *args: False,
        _may_read_inputs_in_place=lambda *args: False,
        _padding_drop_buckets=(),
        _stage_inputs=lambda **kwargs: trace.append("stage"),
        _slot_of_expert=None,
        _topk_idx_dtype=torch.int32,
        _launch_megamoe_kernel=launch,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("priming", [False, True], ids=["primed", "priming"])
def test_cute_dsl_stages_before_the_launch_that_waits(monkeypatch, priming):
    trace = []

    def wait():
        trace.append("wait")

    _tuning(cute_dsl, monkeypatch, priming)
    monkeypatch.setattr(cute_dsl, "_MEGAMOE_PRIME_LADDER", True)

    out = CUTE_DSL._run_moe(
        _cute_dsl_staging(trace, wait),
        x=torch.zeros(2, 4, dtype=torch.uint8),
        x_sf=torch.zeros(2, 1, dtype=torch.uint8),
        topk_idx=torch.zeros(2, 2, dtype=torch.int32),
        topk_weights=torch.ones(2, 2),
        weight_view=None,
        num_tokens=2,
        top_k=2,
        hidden=8,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
        before_launch=wait,
    )

    assert out == "out"
    # The ladder's launches run the fused kernel too, so they wait first; the
    # live launch gets the wait to issue after the staging.
    assert trace == [*(["wait", "prime"] if priming else []), "stage", ("launch", True)]


@pytest.mark.cpu_only
@pytest.mark.parametrize("takes_wait", [True, False], ids=["wait", "no_wait"])
def test_cute_dsl_run_moe_hands_the_wait_to_its_launch(takes_wait):
    seen = []
    wait = (lambda: None) if takes_wait else None
    impl = SimpleNamespace(
        dtype=torch.bfloat16,
        ep_size=1,
        hidden_size=8,
        _weights_created=True,
        quant_method=object(),
        _build_weight_view=lambda: None,
        _run_moe=lambda **kwargs: seen.append(kwargs["before_launch"]) or "out",
    )
    ctx = MoERunContext(
        token_selected_experts=torch.zeros(2, 2, dtype=torch.int32),
        token_final_scales=torch.ones(2, 2),
        x=torch.zeros(2, 4, dtype=torch.uint8),
        x_sf=torch.zeros(2, 1, dtype=torch.uint8),
        before_launch=wait,
    )

    assert CUTE_DSL.run_moe(impl, ctx) == "out"
    assert seen == [wait]


@pytest.mark.cpu_only
def test_cute_dsl_issues_the_wait_right_before_the_kernel(monkeypatch):
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op

    trace = []
    max_t, hidden = 4, 8
    monkeypatch.setattr(
        op, "run_cute_dsl_megamoe_nvfp4_blackwell", lambda **kwargs: trace.append("kernel")
    )
    impl = SimpleNamespace(
        _profiling_scratch_provider_ref=None,
        _profiling_scratch_kwargs=None,
        _use_in_kernel_fc2_reduce=lambda launch_max_t: True,
        _shared_expert=None,
        compute_slot_count=4,
        _replica_plan_launch_args=lambda plan: None,
        _wait_rebalance_routes=lambda plan: trace.append("rebalance"),
        _cute_weight_cache=None,
        _symm_providers={},
        intermediate_size_per_partition=16,
        expand_intermediate_size_per_partition=32,
        apply_topk_in_fc1=False,
        act_clamp=None,
        activation_type=ActivationType.Swiglu,
        act_alpha=None,
        act_beta=None,
        combine_format=None,
        tactic_autotune=False,
    )
    weights = torch.zeros(1, 4, dtype=torch.uint8)
    weight_view = SimpleNamespace(
        fc1_weight=weights,
        fc1_weight_sf=weights,
        fc2_weight=weights,
        fc2_weight_sf=weights,
        fc31_alpha=None,
        fc2_alpha=None,
        fc1_norm_const=None,
    )
    combine_output = torch.ones(max_t, 1, hidden, dtype=torch.bfloat16)

    CUTE_DSL._launch_megamoe_kernel(
        impl,
        activation=torch.zeros(max_t, 4, dtype=torch.uint8),
        activation_sf=torch.zeros(max_t, 1, dtype=torch.uint8),
        topk_idx=torch.zeros(max_t, 2, dtype=torch.int32),
        topk_weights=torch.zeros(max_t, 2),
        weight_view=weight_view,
        combine_output=combine_output,
        shared_workspace=torch.zeros(1, dtype=torch.uint8),
        world_size=2,
        local_rank=0,
        top_k=2,
        hidden=hidden,
        peer_offsets=[0, 0],
        num_tokens=2,
        output_dtype=torch.bfloat16,
        launch_max_T=max_t,
        before_launch=lambda: trace.append(("wait", bool(combine_output[:2].eq(0).all()))),
    )

    # The output clear and the route wait are enqueued before the wait; the
    # kernel right after it.
    assert trace == ["rebalance", ("wait", True), "kernel"]
