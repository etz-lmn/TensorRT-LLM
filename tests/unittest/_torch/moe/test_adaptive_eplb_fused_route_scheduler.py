# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts of the adaptive fused route on the scheduler, backend and balancer side (no GPU, methods extracted
from their sources as test_dynamic_eplb_contracts does):

  * the fused-comm forward takes the one-launch adaptive route exactly when no fallback condition holds, and every
    fallback condition (routing aux, router weight on the input, dynamic balancer, replica plan, a balancer that does
    not stage routes, a backend without the adaptive mode or with fusion off, an NVFP4 input, warmup counters, a slot
    recorder, a caller that does not handle ``routes_mapped``) keeps the separate launches;
  * the balancer hands its statistics to the launch once: a stale pending record is flushed, nothing stays pending,
    and the statistics go only when update_local_statistic would keep the routes;
  * the backend stages once: the launch result is the bucket's prestaged inputs, so run_moe does not stage again,
    and without a bucket nothing is launched;
  * in the forward, a mapped result neither counts the statistics again nor builds the deferred route.
"""

import ast
from types import SimpleNamespace

import pytest
from _torch.moe.test_dynamic_eplb_contracts import _TORCH_ROOT, _extract_class

FALLBACKS = [
    "aux", "input_weight", "dynamic", "replica", "not_staged", "no_adaptive_backend", "fusion_off",
    "nvfp4_input", "warmup_counters", "recorded", "caller",
]


def _scheduler(blocked, recorded_calls):
    fp4 = type("Fp4QuantizedTensor", (), {})
    ns = {
        "Fp4QuantizedTensor": fp4,
        "_routes_recorded": lambda: blocked == "recorded",
        "MoEFusedRouteQuant": object,
    }
    cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/moe_scheduler.py",
        "MoEScheduler",
        {"_adaptive_fused_route", "_try_fused_route_quant", "_route_quant_eligible"},
        ns,
    )
    balancer = SimpleNamespace(
        stages_routes=blocked != "not_staged",
        fused_route=lambda first: recorded_calls.append(("fused_route", first)) or ("table", 3, "stats", first),
    )
    backend = SimpleNamespace(
        fused_route_quant_enabled=blocked != "fusion_off",
        try_fused_route_quant=lambda x, logits, route_counters=None, adaptive_route=None: recorded_calls.append(
            ("backend", adaptive_route, route_counters)
        )
        or "fused",
    )
    if blocked != "no_adaptive_backend":
        backend.fuses_adaptive_route = True
    scheduler = cls()
    scheduler.moe = SimpleNamespace(
        backend=backend,
        apply_router_weight_on_input=blocked == "input_weight",
        _using_dynamic_load_balancer=lambda: blocked == "dynamic",
        _load_balancer_uses_replica_plan=lambda: blocked == "replica",
        layer_load_balancer=balancer,
    )
    x = fp4() if blocked == "nvfp4_input" else object()
    return scheduler, x


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("blocked", [None] + FALLBACKS)
def test_the_fused_comm_forward_takes_the_adaptive_route_unless_a_fallback_holds(blocked, first):
    calls = []
    scheduler, x = _scheduler(blocked, calls)
    result = scheduler._try_fused_route_quant(
        x,
        "logits",
        object() if blocked == "aux" else None,
        object() if blocked == "warmup_counters" else None,
        first=first,
        adaptive=blocked != "caller",
    )
    if blocked is None:
        assert result == "fused"
        assert calls == [("fused_route", first), ("backend", ("table", 3, "stats", first), None)]
    else:
        # The adaptive route is not asked for; an adaptive layer (stages_routes) is not eligible for the
        # upstream one-launch path either, so the forward keeps the route-only launch and RouteWriter.
        assert not any(c[0] == "fused_route" for c in calls)
        assert not any(c[0] == "backend" and c[1] is not None for c in calls)
        if blocked != "not_staged":
            assert result is None


class _Statistics(tuple):
    rows = SimpleNamespace(is_cuda=True)

    def record(self, ids, first):
        self.recorded.append((ids, first))


@pytest.mark.parametrize("recording,capturing", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("stale", [False, True])
def test_the_balancer_hands_its_statistics_to_the_launch_once(recording, capturing, stale):
    torch_ns = SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: capturing))
    cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/adaptive_eplb/balancer.py",
        "AdaptiveMoeLayerLoadBalancer",
        {"fused_route"},
        {"torch": torch_ns},
    )
    layer = cls()
    stats = _Statistics(("rows", "forwards", "enabled"))
    stats.recorded = []
    layer.statistics = stats
    layer._pending = ("old-ids", True) if stale else None
    layer._table_row = "table-row"
    layer.manager = SimpleNamespace(recording=recording, ep_rank=5)
    table, ep_rank, statistics, first = layer.fused_route(False)
    assert (table, ep_rank, first) == ("table-row", 5, False)
    assert statistics == (("rows", "forwards", "enabled") if recording or capturing else None)
    assert layer._pending is None, "nothing is left for a later route to count again"
    assert stats.recorded == ([("old-ids", True)] if stale else [])


@pytest.mark.parametrize("bucket", [False, True])
@pytest.mark.parametrize("padding_drop", [False, True])
def test_the_backend_stages_the_bucket_once(bucket, padding_drop):
    launched = []

    def fake_launch(*args, staged=None, mxfp8=None, adaptive=None):
        launched.append((staged, adaptive))
        return "ids", "weights", "x_fp4", "x_sf", None

    ns = {
        "dsv4_route_nvfp4_quant": fake_launch,
        "AdaptiveRoute": lambda *a: a,
        "MoEFusedRouteQuant": lambda *a, **k: (a, k),
        "live_rows": SimpleNamespace(current_word=lambda device: "word"),
        "torch": SimpleNamespace(float32="fp32"),
    }
    cls = _extract_class(
        _TORCH_ROOT / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py",
        "TrtllmCutedslMegaMoeNvfp4Impl",
        {"_fused_adaptive_route_quant"},
        ns,
    )
    backend = cls()
    backend._slot_of_expert = None
    backend._padding_drop_buckets = {256} if padding_drop else set()
    backend._shared_activation_target = lambda n: None
    backend._prestaged = None
    backend.routing_method = SimpleNamespace(e_score_correction_bias="bias", top_k=6, routed_scaling_factor=2.5)
    backend.fc31_input_scale = "scale"
    staging = (SimpleNamespace(shape=(256, 6)), "w", "a", "sf") if bucket else None
    x = SimpleNamespace(shape=(8, 7168), device="cuda")
    result = backend._fused_adaptive_route_quant(x, "logits", staging, ("table", 3, "stats", True))
    if not bucket:
        assert result is None and not launched and backend._prestaged is None
        return
    (staged, adaptive), = launched
    assert staged is staging
    assert adaptive == ("table", 3, "stats", True, "word" if padding_drop else None)
    # run_moe finds exactly these tensors prestaged in this bucket (inputs_staged) and stages nothing again.
    assert backend._prestaged == (staging[0], "ids", "weights", "x_fp4", "x_sf")
    assert result == (("ids", "weights", "x_fp4", "x_sf"), {"routes_mapped": True})


def test_a_mapped_result_is_neither_counted_nor_routed_again_in_the_forward():
    """In FusedCommMoEScheduler's forward, the routes_mapped branch comes first and contains neither the statistics
    update nor the deferred route; the fused-comm call is the one that asks for the adaptive mode."""
    source = (_TORCH_ROOT / "moe/fused_moe/moe_scheduler.py").read_text()
    tree = ast.parse(source)
    branch = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and "routes_mapped" in ast.get_source_segment(source, node.test)
    )
    mapped = "\n".join(ast.get_source_segment(source, stmt) for stmt in branch.body)
    assert "_load_balancer_update_statistic" not in mapped
    assert "deferred_route" not in mapped and "_load_balancer_route" not in mapped
    assert "token_selected_slots = token_selected_experts" in mapped
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_try_fused_route_quant"
    ]
    adaptive = [c for c in calls if any(k.arg == "adaptive" for k in c.keywords)]
    assert len(calls) == 2 and len(adaptive) == 1, "only the fused-comm forward asks for the adaptive mode"
