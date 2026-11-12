# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement line: the GPU time per MoE layer of preparing MegaMoE's inputs, with the
quantization and the route staging as separate kernels and as one input staging launch, for both arms
(no balancing: megamoe_stage_routes; adaptive EPLB: adaptive_eplb_route_into).

Each variant is a CUDA graph of 61 layers, each layer a producer kernel (the gate's ids), the
variant's kernels and a consumer kernel (a read of the staging), so a kernel's cost includes its
launch and its tail. Rounds alternate the variants; a variant's time is the median per layer of
its rounds, a difference the median of the paired per-round differences. The times go into the
JUnit properties; nothing is asserted about them."""

import json
import os
import random
import statistics

os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")  # as the deployment runs; read once per process

import pytest  # noqa: E402
import torch  # noqa: E402
from _torch.moe.test_adaptive_eplb_kernels import _random_layout, _stats, _tables  # noqa: E402

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import build_route_tables  # noqa: E402

LAYERS, HIDDEN, TOP_K, EXPERTS, SLOTS, EP_SIZE = 61, 7168, 6, 384, 512, 32
ROUNDS, REPLAYS = 30, 20
PAIRS = {
    "none: separate - fused": ("none_old", "none_new"),
    "ours: separate - fused": ("ours_old", "ours_new"),
    "ours - none, separate": ("ours_old", "none_old"),
    "ours - none, fused": ("ours_new", "none_new"),
    "none fused - quantize alone": ("none_new", "quant"),
    "ours fused - quantize alone": ("ours_new", "quant"),
    "quantize alone - producer and consumer": ("quant", "base"),
    "ours separate: statistics - none": ("ours_old", "ours_old_nostats"),
    "ours fused: statistics - none": ("ours_new", "ours_new_nostats"),
}


def _case(tokens: int, replicas: bool) -> dict:
    """One layer's buffers at ``tokens`` rows per rank: the staging of its bucket (256 rows or more,
    MegaMoE's smallest bucket), the activation and its NVFP4 staging, route tables, statistics."""
    rng = random.Random(tokens)
    layout = _random_layout(rng, EXPERTS, SLOTS, EP_SIZE)
    if not replicas:  # every expert on its home slot only
        per_rank, local = EXPERTS // EP_SIZE, SLOTS // EP_SIZE
        layout = [
            s % local + s // local * per_rank if s % local < per_rank else -1 for s in range(SLOTS)
        ]
    shares = [rng.random() for _ in range(SLOTS)] if replicas else None
    bucket = max(256, tokens)
    ids = [rng.randrange(EXPERTS) for _ in range(tokens * TOP_K)]
    return dict(
        tables=_tables(*build_route_tables(layout, shares, EXPERTS, SLOTS)),
        src=torch.tensor(ids, dtype=torch.int32, device="cuda").view(tokens, TOP_K),
        ids=torch.empty(tokens, TOP_K, dtype=torch.int32, device="cuda"),
        weights=torch.rand(tokens, TOP_K, device="cuda"),
        x=torch.randn(tokens, HIDDEN, device="cuda").bfloat16(),
        scale=torch.tensor([448 * 6 / 8.0], device="cuda"),
        fp4=torch.empty(tokens, HIDDEN // 2, dtype=torch.uint8, device="cuda"),
        sf=torch.empty(tokens, HIDDEN // 16, dtype=torch.uint8, device="cuda"),
        slots=torch.empty(bucket, TOP_K, dtype=torch.int64, device="cuda"),
        gate=torch.empty(bucket, TOP_K, device="cuda"),
        sink=torch.empty(bucket, TOP_K, dtype=torch.int64, device="cuda"),
        stats=_stats(EXPERTS),
    )


def _variants(c: dict) -> dict:
    ops = torch.ops.trtllm
    staging = (c["ids"], c["weights"], c["slots"], c["gate"])
    activation = (c["x"], c["scale"], c["fp4"], c["sf"])

    def quantize():
        ops.fp4_quantize_into(*activation, False)

    def route(stats, quantized):
        stat_args = (*c["stats"], True) if stats else (None, None, None, True)
        ops.adaptive_eplb_route_into(
            c["ids"],
            *c["tables"],
            3,
            c["slots"],
            c["weights"],
            c["gate"],
            *stat_args,
            None,
            *(activation if quantized else ()),
        )

    return {
        "base": lambda: None,
        "quant": quantize,
        "none_old": lambda: (quantize(), ops.megamoe_stage_routes(*staging)),
        "none_new": lambda: ops.megamoe_stage_inputs(*activation, *staging),
        "ours_old": lambda: (route(True, False), quantize()),
        "ours_new": lambda: route(True, True),
        "ours_old_nostats": lambda: (route(False, False), quantize()),
        "ours_new_nostats": lambda: route(False, True),
    }


def _graph(c: dict, layer) -> torch.cuda.CUDAGraph:
    def forward():
        for _ in range(LAYERS):
            c["ids"].copy_(c["src"])
            layer()
            c["sink"].copy_(c["slots"])

    forward()  # eager once: lazy initialization outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_stage_inputs_launch_time_per_layer(record_property):
    if not hasattr(torch.ops.trtllm, "megamoe_stage_inputs"):
        pytest.skip("the build has no megamoe_stage_inputs")
    results = {}
    for tokens in (32, 64, 256):
        for replicas in (False, True):
            c = _case(tokens, replicas)
            graphs = {name: _graph(c, fn) for name, fn in _variants(c).items()}
            for graph in graphs.values():
                for _ in range(5):
                    graph.replay()
            torch.cuda.synchronize()
            per_layer = {name: [] for name in graphs}
            names = list(graphs)
            for r in range(ROUNDS):
                for name in names[r % len(names) :] + names[: r % len(names)]:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(REPLAYS):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    per_layer[name].append(start.elapsed_time(end) * 1e3 / (REPLAYS * LAYERS))
            key = f"tokens {tokens}, {'replicated' if replicas else 'home'} layout"
            results[key] = {
                "us_per_layer": {n: round(statistics.median(v), 3) for n, v in per_layer.items()},
                "paired_us_per_layer": {
                    label: round(
                        statistics.median(x - y for x, y in zip(per_layer[a], per_layer[b])), 3
                    )
                    for label, (a, b) in PAIRS.items()
                },
            }
            print(key, json.dumps(results[key]))
    record_property("stage_inputs_timing", json.dumps(results))
