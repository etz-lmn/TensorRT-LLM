# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the MegaMoE backends issue ``MoERunContext.before_launch``.

A fused-communication scheduler hands a backend that declares
``takes_before_launch`` the wait for the EPLB statistic allreduce instead of
issuing it before ``run_moe``. The backend must issue it before every launch of
its fused kernel, bucket priming included, because that kernel must not run
alongside the allreduce. CuteDSL issues it after its staging, which then
overlaps the allreduce; DeepGEMM issues it before ``megamoe_prepare``, which
beside the allreduce slows it. The real ``run_moe`` paths run here on CPU
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
def test_deepgemm_waits_before_megamoe_prepare_and_each_launch(monkeypatch, priming, takes_wait):
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
    prime = [("launch", 0)] if priming else []
    assert trace == [*wait, *prime, "prepare", ("launch", 2)]


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


# ---------------------------------------------------------------------------
# GPU: MegaMoEDeepGemm under online EPLB on four ranks, multi-stream on, so the statistic
# allreduce runs on the load balancer's auxiliary stream. Per rank it records the median forward
# time and how long the launch stream stalls in the statistic-allreduce wait (wherever the build
# issues it: before run_moe, or from the backend's launch path), for an A/B of two builds.
# ---------------------------------------------------------------------------
_WORLD = 4
_ROWS = 64
_FORWARDS = 8

if torch.cuda.is_available():
    import pickle
    import sys

    import cloudpickle

    # The 4-GPU MoE cases' shared pool; test_moe_module queries the device when imported.
    from _torch.moe.test_moe_module import moe_multi_gpu_executor  # noqa: F401
    from mpi4py import MPI

    # The pool's workers get this module's worker by value.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _deepgemm_eplb_worker() -> dict:
    import traceback

    try:
        return _deepgemm_eplb_worker_impl()
    except Exception:
        traceback.print_exc()
        raise


def _deepgemm_eplb_worker_impl() -> dict:
    from _torch.moe.moe_test_utils import MoeBackendType, MoeModelConfig
    from _torch.moe.quantize_utils import get_test_quant_params
    from _torch.moe.test_moe_module import (
        _create_mapping_for_parallel_mode,
        _create_model_config,
        _create_moe_load_balancer,
        _ensure_dist_for_megamoe,
    )

    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.moe.fused_moe import (
        DEFAULT_MOE_ACTIVATION,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    model = MoeModelConfig(8, 2, 512, 512)
    num_slots, dtype = 16, torch.bfloat16
    backend = MoeBackendType.MEGAMOE_DEEPGEMM
    rank = mpi_rank()
    mapping = _create_mapping_for_parallel_mode(_WORLD, "DEP")
    mapping.rank = rank
    torch.cuda.set_device(rank)
    _ensure_dist_for_megamoe(backend.value, rank, _WORLD)
    with torch.device(f"cuda:{rank}"):
        torch.manual_seed(0)
        routing_method = RenormalizeMoeRoutingMethod(top_k=model.top_k)
        x = torch.randn((_ROWS, model.hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((_ROWS, model.num_experts), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.W4A8_MXFP4_MXFP8, x, backend
        )
        quant_kwargs.pop("ref_cls", None)
        quantize_util = quantize_util_cls(
            num_experts=model.num_experts,
            dtype=dtype,
            intermediate_size=model.intermediate_size,
            hidden_size=model.hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            num_local_experts=model.num_experts // mapping.moe_ep_size,
            activation_type=ActivationType.Swiglu,
        )
        model_cfg = _create_model_config(
            num_experts=model.num_experts,
            hidden_size=model.hidden_size,
            intermediate_size=model.intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=backend.value,
            enable_eplb=True,
            num_slots=num_slots,
            layer_updates_per_iter=1,
            max_num_tokens=max(256, _ROWS),
        )
        balancer = _create_moe_load_balancer(model_cfg, True)
        with (
            nccl_window_tensor_scope(x),
            balancer,
            create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=False,
                activation=DEFAULT_MOE_ACTIVATION,
                weight_loading_mode=getattr(
                    quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
                ),
            ) as fused_moe,
        ):
            fused_moe.create_weights()
            weights, _, _ = quantize_util.prepare_weights_from_backend(fused_moe, **quant_kwargs)
            fused_moe.load_weights(
                [{k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in weights.items()}]
            )
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{rank}")
            balancer.register_weight_slots_after_to_cuda()
            balancer.finalize_model()
            balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)

            # The wait, wherever the build issues it, timed on the launch stream.
            waits = []
            wait = fused_moe._load_balancer_wait_for_statistic_allreduce

            def timed_wait():
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                wait()
                end.record()
                waits.append((begin, end))

            fused_moe._load_balancer_wait_for_statistic_allreduce = timed_wait
            counts = [_ROWS] * _WORLD
            times, outputs = [], []
            with torch.inference_mode(), with_multi_stream(True):
                for i in range(_FORWARDS + 1):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with MoeLoadBalancerIterContext(balancer):
                        begin.record()
                        out = fused_moe.forward(x, router_logits, all_rank_num_tokens=counts)
                        end.record()
                    end.synchronize()
                    if i:  # the first forward primes the kernels
                        times.append(begin.elapsed_time(end) * 1e3)
                        outputs.append(out.float())
            torch.cuda.synchronize()
            stalls = [b.elapsed_time(e) * 1e3 for b, e in waits[1:]]
    return dict(
        rank=rank,
        forward_us=sorted(times)[len(times) // 2],
        wait_us=sorted(stalls)[len(stalls) // 2] if stalls else None,
        waits=len(waits),
        finite=all(bool(torch.isfinite(out).all()) for out in outputs),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD,
    reason=f"needs {_WORLD} GPUs",
)
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
def test_deepgemm_online_eplb_layer_time_and_statistic_wait(
    moe_multi_gpu_executor, record_property
):
    """MegaMoEDeepGemm under online EPLB with multi-stream on: every forward finishes (no rank waits
    on another whose kernel runs beside the allreduce) and gives finite rows; the median forward
    time and the launch stream's stall in the statistic-allreduce wait are recorded per rank."""
    from _torch.moe.moe_test_utils import MoeBackendType, MoeModelConfig
    from _torch.moe.test_moe_module import MEGAMOE_IGNORE_COMM_METHOD, should_skip_MegaMoEDeepGemm

    from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    reason = should_skip_MegaMoEDeepGemm(
        "DEP",
        MEGAMOE_IGNORE_COMM_METHOD,
        MoeBackendType.MEGAMOE_DEEPGEMM,
        QuantAlgo.W4A8_MXFP4_MXFP8,
        torch.bfloat16,
        MoeModelConfig(8, 2, 512, 512),
        RenormalizeMoeRoutingMethod,
        False,
    )
    if reason:
        pytest.skip(reason)
    results = list(moe_multi_gpu_executor.map(_deepgemm_eplb_worker_wrapper, range(_WORLD)))
    for result in results:
        record_property(f"rank{result['rank']}_forward_us", round(result["forward_us"], 2))
        if result["wait_us"] is not None:
            record_property(f"rank{result['rank']}_statistic_wait_us", round(result["wait_us"], 2))
        record_property(f"rank{result['rank']}_statistic_waits", result["waits"])
    assert all(result["finite"] for result in results)


def _deepgemm_eplb_worker_wrapper(_index: int) -> dict:
    return _deepgemm_eplb_worker()
