# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MegaMoE CuteDSL backend's staging of the next run."""

import types
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


def test_staging_view_identity_rechecks_rebound_storage():
    staging = torch.zeros(8, 4, dtype=torch.uint8)
    view = staging[:2]
    assert mm._is_staging_view(view, staging, 2)
    view.set_(torch.empty_like(view))
    assert not mm._is_staging_view(view, staging, 2)


@pytest.mark.parametrize("route_dtype", [torch.int32, torch.int64])
def test_empty_preparation_preserves_staging_identity_and_launches_peer_work(
    monkeypatch, route_dtype
):
    bucket, hidden, top_k = 8, 64, 2
    bufs = mm._MegaMoeBuffers(
        activation=torch.zeros(bucket, hidden // 2, dtype=torch.uint8),
        activation_sf=torch.zeros(bucket, hidden // 16, dtype=torch.uint8),
        topk_weights=torch.ones(bucket, top_k),
        combine_output=torch.zeros(bucket, hidden),
        shared_workspace=torch.zeros(1),
        peer_offsets=(0, 0),
        topk_idx_local=torch.zeros(bucket, top_k, dtype=route_dtype),
    )
    launches = []
    staged = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        _quantized_into=None,
        _shared_expert=None,
        _slot_of_expert=None,
        _prestaged=None,
        _routes_staged_into=None,
        ep_size=2,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets=set(),
        fc31_input_scale=torch.ones(1),
        _select_launch_max_tokens=lambda hint: bucket,
        _staging_of_next_run=lambda rows, device: bufs,
        _acquire_buffers=lambda **kw: bufs,
        _stage_inputs=lambda **kw: staged.append(kw),
        _launch_megamoe_kernel=lambda **kw: launches.append(kw) or "out",
    )
    backend._activation_to_quantize = types.MethodType(IMPL._activation_to_quantize, backend)
    backend._shared_activation_target = types.MethodType(IMPL._shared_activation_target, backend)
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    monkeypatch.setattr(mm, "_MEGAMOE_PRIME_LADDER", False)
    monkeypatch.setattr(torch.ops.trtllm, "megamoe_stage_inputs", lambda *args: None)
    x = torch.empty(0, hidden, dtype=torch.bfloat16)
    ids = torch.empty(0, top_k, dtype=torch.int64)
    weights = torch.empty(0, top_k)

    backend._routes_staged = types.MethodType(IMPL._routes_staged, backend)

    def prepare():
        return IMPL._stage_bf16_inputs(backend, x, ids, weights, None)

    def run(inputs):
        return IMPL._run_moe(
            backend,
            x=inputs[0],
            x_sf=inputs[1],
            topk_idx=inputs[2],
            topk_weights=weights,
            weight_view=None,
            num_tokens=0,
            top_k=top_k,
            hidden=hidden,
            device=x.device,
            output_dtype=x.dtype,
        )

    prepared = prepare()
    assert all(t.data_ptr() == 0 for t in prepared)
    assert run(prepared) == "out"
    assert len(launches) == 1 and launches[0]["world_size"] == 2
    assert launches[0]["num_tokens"] == 0 and not staged
    for index in range(3):
        unrelated = list(prepare())
        unrelated[index] = torch.empty_like(unrelated[index])
        with pytest.raises((RuntimeError, ValueError), match="quantize_input|staging"):
            run(unrelated)
    wrong_offset = list(prepare())
    wrong_offset[0] = bufs.activation[1:1]
    with pytest.raises(RuntimeError, match="quantize_input"):
        run(wrong_offset)
    assert len(launches) == 1


def test_the_staging_of_the_next_run_is_handed_out_only_for_a_run_that_stages():
    """``_staging_of_next_run`` (where ``quantize_input`` writes) hands out a bucket's staging only
    for a forward that ``run_moe`` would stage."""
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
        _active_launch_output_dtype=torch.bfloat16,
        hidden_size=4,
        routing_method=SimpleNamespace(experts_per_token=2),
        _local_staging_cache={},
        _select_launch_max_tokens=lambda hint: 4 if hint <= 4 else 8,
        _acquire_buffers=lambda **kw: acquired.append(kw) or bufs,
    )
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    cpu = torch.device("cpu")
    staging = types.MethodType(IMPL._staging_of_next_run, backend)
    assert staging(2, cpu) is None  # no bucket hint yet
    backend._active_launch_max_tokens = 3
    assert staging(2, cpu) is None  # the staging is not built
    backend._local_staging_cache[mm._StagingKey(4, 2, 4, "cpu", torch.bfloat16)] = {}
    assert staging(2, cpu) is bufs
    assert acquired[-1]["launch_max_T"] == 4 and acquired[-1]["device"] == cpu
    assert staging(2, torch.device("meta")) is None  # a staging set of another device
    assert staging(4, cpu) is None  # an exact-size direct-input bucket
    assert staging(5, cpu) is None  # more rows than the hinted bucket holds
    backend._active_launch_max_tokens = 5
    assert staging(5, cpu) is None  # bucket 8 is selected but not built


@pytest.mark.parametrize("route_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("static_slots", [False, True])
def test_run_moe_launches_from_the_staging_only_the_rows_quantize_input_put_there(
    route_dtype, static_slots
):
    """``run_moe`` takes the staging's rows as already quantized only when ``quantize_input``
    recorded that it quantized exactly those rows into it; any other pairing raises instead of
    copying the rows again or launching stale ones."""
    act = torch.zeros(8, 4, dtype=torch.uint8)
    act_sf = torch.zeros(8, 1, dtype=torch.uint8)
    t = torch.zeros(8, 2)
    bufs = mm._MegaMoeBuffers(
        activation=act,
        activation_sf=act_sf,
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=torch.full((8, 2), -1, dtype=route_dtype),
    )
    stage_activation = []
    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        _quantized_into=None,
        _shared_expert=None,
        _slot_of_expert=torch.tensor([1, 0], dtype=torch.int32) if static_slots else None,
        _replicated=False,
        _prestaged=None,
        ep_size=1,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets=set(),
        _select_launch_max_tokens=lambda hint: 8,
        _acquire_buffers=lambda **kw: bufs,
        _stage_inputs=lambda **kw: stage_activation.append(kw["stage_activation"]),
        _launch_megamoe_kernel=lambda **kw: "out",
    )
    mapped_dtypes = []

    def slots_of_experts(ids):
        mapped_dtypes.append(ids.dtype)
        return IMPL.slots_of_experts(backend, ids)

    backend.slots_of_experts = slots_of_experts
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    backend._routes_staged = types.MethodType(IMPL._routes_staged, backend)
    routes = torch.zeros(3, 2, dtype=torch.int64)

    def run(x: torch.Tensor, x_sf: torch.Tensor) -> str:
        return IMPL._run_moe(
            backend,
            x=x,
            x_sf=x_sf,
            topk_idx=routes,
            topk_weights=routes.float(),
            weight_view=None,
            num_tokens=x.shape[0],
            top_k=2,
            hidden=8,
            device=torch.device("cpu"),
            output_dtype=torch.bfloat16,
        )

    staged, fresh = (
        (act[:3], act_sf[:3]),
        (torch.zeros(3, 4, dtype=torch.uint8), act_sf[:3].clone()),
    )
    backend._quantized_into = (act.data_ptr(), 3)
    assert run(*staged) == "out" and stage_activation == [False]
    assert run(*fresh) == "out" and stage_activation == [False, True]
    for record, inputs in (
        (
            (act.data_ptr(), 3),
            fresh,
        ),  # quantized into this staging, launched from elsewhere
        ((act.data_ptr(), 2), staged),  # another row count
        (
            None,
            staged,
        ),  # a view of the staging that this run's quantize_input did not write
    ):
        backend._quantized_into = record
        with pytest.raises(RuntimeError, match="quantize_input quantized"):
            run(*inputs)
        assert backend._quantized_into is None  # consumed by the launch it was meant for
    assert stage_activation == [False, True]
    # Routes a load balancer wrote into the staging (route_staging) are not staged again: after
    # quantize_input nothing is staged, otherwise only the activation rows are copied.
    assert all(dtype == torch.int32 for dtype in mapped_dtypes)
    mapped_before_handout = len(mapped_dtypes)
    routes = bufs.topk_idx_local[:3]
    backend._handed_out_route_staging = routes.data_ptr()
    backend._quantized_into = (act.data_ptr(), 3)
    assert run(*staged) == "out"
    backend._handed_out_route_staging = routes.data_ptr()
    assert run(torch.full((3, 4), 7, dtype=torch.uint8), act_sf[:3].clone()) == "out"
    assert (act[:3] == 7).all() and stage_activation == [False, True]

    assert len(mapped_dtypes) == mapped_before_handout  # Handed-out slots are never mapped twice.


@pytest.mark.parametrize("drops_padding_rows", [True, False])
def test_run_moe_stages_with_the_forward_word_only_where_the_kernel_masks_dropped_routes(
    monkeypatch: pytest.MonkeyPatch, drops_padding_rows: bool
):
    """A bucket whose kernel's top-k reduce masks routes with an invalid expert stages the routes
    under the forward's live-row word, so the rows of padding requests are dropped; any other
    bucket stages every row (its reduce would read the dropped rows' stale cells)."""
    t = torch.zeros(8, 2)
    bufs = mm._MegaMoeBuffers(
        activation=torch.zeros(8, 4, dtype=torch.uint8),
        activation_sf=torch.zeros(8, 1, dtype=torch.uint8),
        topk_weights=t,
        combine_output=t,
        shared_workspace=t,
        peer_offsets=(0,),
        topk_idx_local=torch.full((8, 2), -1, dtype=torch.int64),
    )
    staged = []

    def stage_inputs(**kwargs):
        staged.append(kwargs["live_row_word"])
        assert kwargs["topk_idx"].dtype == torch.int32

    backend = SimpleNamespace(
        _active_launch_max_tokens=None,
        _quantized_into=None,
        _shared_expert=None,
        _slot_of_expert=None,
        _prestaged=None,
        ep_size=1,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets={8} if drops_padding_rows else {16},
        _select_launch_max_tokens=lambda hint: 8,
        _acquire_buffers=lambda **kw: bufs,
        _stage_inputs=stage_inputs,
        _launch_megamoe_kernel=lambda **kw: "out",
    )
    backend._may_read_inputs_in_place = types.MethodType(IMPL._may_read_inputs_in_place, backend)
    backend._routes_staged = types.MethodType(IMPL._routes_staged, backend)
    word = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(mm.live_rows, "current_word", lambda device: word)
    routes = torch.zeros(3, 2, dtype=torch.int64)
    IMPL._run_moe(
        backend,
        x=torch.zeros(3, 4, dtype=torch.uint8),
        x_sf=torch.zeros(3, 1, dtype=torch.uint8),
        topk_idx=routes,
        topk_weights=routes.float(),
        weight_view=None,
        num_tokens=3,
        top_k=2,
        hidden=8,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
    )
    assert len(staged) == 1 and staged[0] is (word if drops_padding_rows else None)


@pytest.fixture
def preparation_backend(monkeypatch):
    """The existing CPU staging harness, with only quantization and launch endpoints replaced."""

    class Quantize:
        def __call__(self, x, *args, **kwargs):
            return torch.zeros(len(x), 32, dtype=torch.uint8), torch.zeros(
                len(x), 4, dtype=torch.uint8
            )

        def into(self, x, scale, activation, sf, *args):
            activation.zero_()
            sf.zero_()

    def stage(x, scale, activation, sf, ids, weights, out_ids, out_weights, *args):
        activation.zero_()
        sf.zero_()
        out_ids[: len(ids)].copy_(ids)
        out_weights[: len(ids)].copy_(weights)

    monkeypatch.setattr(mm, "_MEGAMOE_PRIME_LADDER", False)
    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", Quantize())
    monkeypatch.setattr(torch.ops.trtllm, "megamoe_stage_inputs", stage)
    backend = IMPL.__new__(IMPL)
    torch.nn.Module.__init__(backend)
    for name, value in dict(
        _active_launch_max_tokens=8,
        _active_launch_output_dtype=torch.bfloat16,
        _quantized_into=None,
        _prestaged=None,
        _handed_out_route_staging=None,
        _shared_expert=None,
        _slot_of_expert=None,
        _replicated=False,
        _padding_drop_buckets=set(),
        ep_size=2,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _maxt_buckets=(8, 16),
        max_num_tokens=16,
        _topk_idx_dtype=torch.int64,
        _weights_created=True,
        _symm_providers={16: object()},
        _local_staging_cache={},
        _launch_buffers={},
        dtype=torch.bfloat16,
        hidden_size=64,
        routing_method=SimpleNamespace(experts_per_token=2),
        fc31_input_scale=torch.ones(1),
        fc31_act_scale=None,
        quant_method=object(),
        layer_load_balancer=None,
    ).items():
        setattr(backend, name, value)

    def build(**kwargs):
        bucket, dtype = kwargs["launch_max_T"], kwargs["output_dtype"]
        buffers = mm._MegaMoeBuffers(
            activation=torch.zeros(bucket, 32, dtype=torch.uint8),
            activation_sf=torch.zeros(bucket, 4, dtype=torch.uint8),
            topk_idx_local=torch.zeros(bucket, 2, dtype=torch.int64),
            topk_weights=torch.zeros(bucket, 2),
            combine_output=torch.zeros(bucket, 64, dtype=dtype),
            shared_workspace=torch.zeros(1),
            peer_offsets=(0, 0),
        )
        backend._local_staging_cache[
            mm._StagingKey(bucket, 2, 64, str(kwargs["device"]), dtype)
        ] = {}
        return buffers

    backend._build_buffers = build
    backend._build_weight_view = lambda: None
    backend._stage_inputs = lambda **kwargs: None
    backend._launch_megamoe_kernel = lambda **kwargs: kwargs["combine_output"]
    backend._acquire_buffers(
        top_k=2, hidden=64, device=torch.device("cpu"), output_dtype=torch.bfloat16, launch_max_T=8
    )
    return backend


def test_quantization_prepares_the_requested_output_dtype(preparation_backend):
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

    backend = preparation_backend
    x, ids, weights = (
        torch.zeros(2, 64, dtype=torch.bfloat16),
        torch.zeros(2, 2, dtype=torch.int64),
        torch.ones(2, 2),
    )
    activation, sf = backend.quantize_input(x, output_dtype=torch.float16)
    context = MoERunContext(
        x=activation,
        x_sf=sf,
        token_selected_experts=ids,
        token_final_scales=weights,
        output_dtype=torch.float16,
    )
    assert backend.run_moe(context).dtype == torch.float16
    backend.set_adaptive_launch_tokens(8, output_dtype=torch.float16)
    buffers = backend._staging_of_next_run(len(x), x.device)
    activation, sf = backend.quantize_input(x)
    assert activation.data_ptr() == buffers.activation.data_ptr()
    assert sf.data_ptr() == buffers.activation_sf.data_ptr()
    context = replace(context, x=activation, x_sf=sf)
    assert backend.run_moe(context).dtype == torch.float16
    # A legacy hint without a known output dtype must not promise direct staging.
    backend.set_adaptive_launch_tokens(8)
    assert backend._staging_of_next_run(len(x), x.device) is None


@pytest.mark.parametrize("first_rows", [0, 2])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_preparation_consumes_handout_before_an_uncached_bucket(
    preparation_backend, first_rows, ep_size
):
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

    backend = preparation_backend
    backend.ep_size = ep_size
    backend._slot_of_expert = torch.arange(8, dtype=torch.int32) if ep_size > 1 else None
    for bucket, rows in ((8, first_rows), (16, 3)):
        backend.set_adaptive_launch_tokens(bucket)
        # run_moe knows the output dtype before its own BF16 preparation.
        context = MoERunContext(
            x=torch.zeros(rows, 64, dtype=torch.bfloat16),
            x_sf=None,
            token_selected_experts=torch.zeros(rows, 2, dtype=torch.int64),
            token_final_scales=torch.ones(rows, 2),
            output_dtype=torch.bfloat16,
        )
        assert backend.run_moe(context).dtype == torch.bfloat16
    assert backend._handed_out_route_staging is None
