# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract: where a DeepSeek-V4 MoE layer forks its shared expert under the fused-comm scheduler.

A load balancer that stages its routes (adaptive EPLB) gets the shared expert after the staging and
before the dispatch, so the shared expert's first GEMM (every SM) no longer holds the staging launch
back; every other layer keeps the fork right after the router gate. The shared expert waits for the
one event recorded at the fork, and the fork is recorded once.

The real ``compute_routed_output`` and ``_fork_after_routing`` (modeling), ``_run_chunks``,
``_forward_chunk`` and ``forks_after_staging`` (scheduler) and ``run_moe`` (MegaMoE CuteDSL) run on
stand-ins that log each launch.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_TORCH_ROOT = _ROOT / "tensorrt_llm/_torch"
_SCHEDULER = _TORCH_ROOT / "moe/fused_moe/moe_scheduler.py"
_MODELING = _TORCH_ROOT / "models/modeling_deepseekv4.py"
_MEGAMOE = _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"


def _extract(path, name, methods, namespace, bases=()):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
    )
    if isinstance(node, ast.ClassDef) and methods is not None:
        body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in body} == methods
        node = ast.ClassDef(
            name=name,
            bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases],
            keywords=[],
            body=body,
            decorator_list=[],
        )
    elif isinstance(node, ast.FunctionDef):
        node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class _Rows:
    dtype = "int32"
    shape = (4, 6)

    def __init__(self, name):
        self.name = name


class _Scales:
    dtype = "float32"

    @staticmethod
    def is_contiguous():
        return True


def _run(mode, *, backend_runs_after_staging=True):
    """Drive one MoE layer forward; return the launch log."""
    trace = []
    x_q = SimpleNamespace(shape=(4, 16), device="cpu")  # an input the backend can launch
    torch = SimpleNamespace(
        int32="int32",
        float32="float32",
        bfloat16="bf16",
        cuda=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(
                wait_event=lambda event: trace.append("shared waits fork event")
            )
        ),
    )
    namespace = {
        "torch": torch,
        "functools": functools,
        "dataclasses": dataclasses,
        "Optional": None,
        "Callable": None,
        "get_active_route_capture": lambda: None,
        "Fp4QuantizedTensor": type("Fp4QuantizedTensor", (), {}),
        "RouterLogitPartials": type("RouterLogitPartials", (), {}),
        "MoERunContext": lambda **kwargs: SimpleNamespace(**kwargs),
        "ExpertStatistic": SimpleNamespace(
            set_layer=lambda i: None, maybe_add_info=lambda *a: None, get=lambda: None
        ),
        "get_calibrator": lambda: SimpleNamespace(
            maybe_collect_or_replay_slots=lambda n, routes: routes,
            mode=SimpleNamespace(name="NONE"),
        ),
    }
    for fn in ("_record_routes", "_leading_rows", "_routes_recorded", "_num_rows"):
        _extract(_SCHEDULER, fn, None, namespace)
    hooks_cls = _extract(_SCHEDULER, "MoEForwardHooks", None, namespace)
    scheduler_cls = _extract(
        _SCHEDULER,
        "FusedCommMoEScheduler",
        {"_run_chunks", "_forward_chunk", "forks_after_staging"},
        namespace,
    )

    # ----- the MegaMoE backend: the real run_moe over logging stand-ins -----
    backend_cls = _extract(_MEGAMOE, "TrtllmCutedslMegaMoeNvfp4Impl", {"run_moe"}, {"torch": torch})
    backend = backend_cls()
    backend.ep_size, backend.dtype, backend.hidden_size = 1, "bf16", 16
    backend._weights_created, backend.quant_method = True, object()
    backend._build_weight_view = lambda: "weights"
    backend.supports_fused_prepare = lambda: mode == "deferred"
    backend.fused_route_quant_enabled = True
    if backend_runs_after_staging:
        backend.runs_after_staging = True

    def stage_bf16_inputs(x, ids, weights, route):
        # The balancer's route launch, with the input quantization into the bucket staging.
        trace.append(("stage", route(quantize="q").name))
        return x, "sf", _Rows("slots")

    def run_megamoe(**kwargs):
        trace.append(("dispatch+megamoe", kwargs["topk_idx"].name))
        return "routed"

    backend._stage_bf16_inputs, backend._run_moe = stage_bf16_inputs, run_megamoe
    backend.set_adaptive_launch_tokens = lambda tokens: None

    # ----- the layer and its load balancer -----
    adaptive = mode in ("deferred", "eager", "fused")
    layer = (
        SimpleNamespace(stages_routes=True)
        if adaptive
        else SimpleNamespace(stages_routes=False)
        if mode == "dynamic"
        else None
    )

    def balancer_route(routes, use_dp, token_final_scales=None, **quantize):
        trace.append(("stage", "slots") if not quantize else ("route launch", *quantize.values()))
        return _Rows("slots")

    moe = SimpleNamespace(
        apply_router_weight_on_input=False,
        mapping=SimpleNamespace(moe_ep_rank=0),
        layer_idx=7,
        num_experts=16,
        num_slots=24,
        use_dp=True,
        repeat_idx=0,
        repeat_count=1,
        layer_load_balancer=layer,
        backend=backend,
        _maybe_get_perfect_router_expert_ids=lambda routes: routes,
        _load_balancer_uses_replica_plan=lambda: mode == "dynamic",
        _load_balancer_start_wait_gpu_stage=lambda first: None,
        _load_balancer_done_wait_gpu_stage=lambda first: None,
        _load_balancer_update_statistic=lambda *a, **k: None,
        _load_balancer_wait_for_statistic_allreduce=lambda: None,
        _load_balancer_route=balancer_route,
        _load_balancer_count_warmup=lambda *a: None,
        _warmup_placement_route_counters=lambda row_offset: None,
        _load_balancer_start_set_cpu_stage=lambda last: None,
        _load_balancer_done_set_cpu_stage=lambda last: None,
        require_replica_plan=None,
        routing_method=SimpleNamespace(
            apply=lambda logits, ids: (trace.append("route") or _Rows("experts"), _Scales())
        ),
    )
    if mode == "dynamic":
        layer.require_replica_plan = lambda: SimpleNamespace(reserved_sms=0)
        backend.quantize_input = lambda x, **kw: (x_q, "sf")
    if mode in ("off", "eager"):
        backend.quantize_input = lambda x, **kw: (x_q, "sf")
    scheduler = scheduler_cls()
    scheduler.moe = moe

    def try_fused(x, logits, aux, counters, first=True, adaptive=False):
        if mode != "fused":
            return None
        trace.append(("stage", "fused route+map+stage launch"))
        return SimpleNamespace(
            token_selected_experts=_Rows("slots"),
            token_final_scales=_Scales(),
            x=x_q,
            x_sf="sf",
            routes_counted=True,
            routes_mapped=True,
        )

    scheduler._try_fused_route_quant = try_fused
    scheduler._route_only = lambda logits, aux: None

    class Experts:
        def __init__(self):
            self.scheduler = scheduler

        def __call__(self, x, logits, *, moe_hooks=None, **kwargs):
            return scheduler._run_chunks(
                [x],
                [logits],
                [],
                num_chunks=1,
                x_real=x,
                rl_real=logits,
                all_rank_chunk_size_list=[[4], [6]],
                had_meta=True,
                output_dtype="bf16",
                do_finalize=True,
                moe_hooks=moe_hooks,
            )[0]

    # ----- the model layer: the real routed path and fork -----
    model_ns = {
        "torch": torch,
        "FusedCommMoEScheduler": scheduler_cls,
        "MoEForwardHooks": hooks_cls,
    }
    producer = _extract(
        _MODELING, "DeepseekV4MoE", {"compute_routed_output", "_fork_after_routing"}, model_ns
    )()
    producer.use_dp, producer.experts = False, Experts()
    producer.gate = lambda x, partials=False: trace.append("router gate") or SimpleNamespace(
        shape=(4, 24)
    )
    producer._post_routing_event = SimpleNamespace(
        record=lambda: trace.append("fork event recorded")
    )

    def routed(hooks=None):
        x = SimpleNamespace(shape=(4, 16), dtype="bf16", device="cpu")
        return producer.compute_routed_output(x, None, None, [4, 6], True, moe_hooks=hooks)

    def shared():
        trace.append("shared expert")
        return "shared"

    routed_from_entry, shared_after_routing = producer._fork_after_routing(routed, shared)
    # _run_in_parallel's order: the routed path is issued on the main stream, then the shared one on the aux stream.
    assert routed_from_entry() == "routed"
    assert shared_after_routing() == "shared"
    return trace


def _between(trace, first, last):
    return trace.index(first) < trace.index("fork event recorded") < trace.index(last)


@pytest.mark.parametrize("mode", ["deferred", "eager", "fused"])
def test_a_staging_balancer_forks_the_shared_expert_after_staging_before_dispatch(mode):
    trace = _run(mode)
    stage = next(t for t in trace if isinstance(t, tuple) and t[0] == "stage")
    dispatch = next(t for t in trace if isinstance(t, tuple) and t[0] == "dispatch+megamoe")
    assert trace.count("fork event recorded") == 1
    assert _between(trace, stage, dispatch)
    # Nothing is launched on the main stream between the staging and the fork.
    assert trace.index("fork event recorded") == trace.index(stage) + 1
    # The shared expert waits for that one event, then runs; the join after it is the caller's.
    assert trace[-2:] == ["shared waits fork event", "shared expert"]


@pytest.mark.parametrize("mode", ["off", "dynamic"])
def test_other_layers_keep_the_fork_right_after_the_router_gate(mode):
    trace = _run(mode)
    assert trace.count("fork event recorded") == 1
    assert trace.index("fork event recorded") == trace.index("router gate") + 1
    assert trace[-2:] == ["shared waits fork event", "shared expert"]


def test_a_backend_that_does_not_run_after_staging_keeps_the_fork_after_the_gate():
    trace = _run("deferred", backend_runs_after_staging=False)
    assert trace.index("fork event recorded") == trace.index("router gate") + 1
    assert trace.count("fork event recorded") == 1


def test_run_moe_starts_the_forked_work_between_staging_and_the_dispatch_launch():
    """The order inside run_moe, read from its source: staging, then after_staging, then _run_moe."""
    tree = ast.parse(_MEGAMOE.read_text())
    cls = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "TrtllmCutedslMegaMoeNvfp4Impl"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run_moe")
    calls = [
        n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
    ]
    lines = {
        name: min(
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) == name
        )
        for name in ("_stage_bf16_inputs", "after_staging", "_run_moe")
        if name in calls
    }
    assert lines["_stage_bf16_inputs"] < lines["after_staging"] < lines["_run_moe"]
