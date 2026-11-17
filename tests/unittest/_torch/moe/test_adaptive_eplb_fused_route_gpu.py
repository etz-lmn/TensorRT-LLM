# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU: the adaptive mode of ``dsv4_route_nvfp4_quant`` (one launch) is bit for bit the two launches it replaces on an
adaptive EPLB layer: ``dsv4_route`` (route only) then ``adaptive_eplb_route_into`` with quantize into the bucket.

Every output is compared over all its rows: returned ids and weights, the whole staged slot and weight bucket (pre-filled
with a sentinel), the NVFP4 rows and scale rows of the tokens, the statistics rows (float bits) and forwards. Dimensions:
home tables, replicated tables and mixed (some experts replicated); tokens 1..64 and full buckets, 256 and 1024 buckets;
live-row word none / padding / no real row; statistics off, enabled 0/1, first 0/1; whole logits and K-chunk partials;
then 200 CUDA graph replays of both paths with the route tables rewritten between replays.
"""

import random

import pytest
import torch

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10, reason="needs an SM100+ GPU"
)

EXPERTS, SLOTS, TOP_K, HIDDEN, SCALING = 384, 512, 6, 7168, 2.5
SENTINEL_SLOT, SENTINEL_WEIGHT = 0x5EED, 1234.5


def _table(rng, replicas, device):
    """A route table row: counts | starts | slots | Q32 cumulative shares (int32 bits)."""
    counts = [1] * EXPERTS
    for _ in range(replicas):
        counts[rng.randrange(EXPERTS)] += 1
    slots = list(range(SLOTS))
    rng.shuffle(slots)
    starts, shares, pos = [], [], 0
    for e in range(EXPERTS):
        starts.append(pos)
        shares += sorted(rng.randrange(1 << 32) for _ in range(counts[e] - 1)) + [(1 << 32) - 1]
        pos += counts[e]
    shares += [(1 << 32) - 1] * (SLOTS - pos)
    shares = [s - (1 << 32) if s >= 1 << 31 else s for s in shares]
    return torch.tensor(counts + starts + slots + shares, dtype=torch.int32, device=device)


def _stage(bucket, device):
    return (
        torch.full((bucket, TOP_K), SENTINEL_SLOT, dtype=torch.int64, device=device),
        torch.full((bucket, TOP_K), SENTINEL_WEIGHT, dtype=torch.float32, device=device),
        torch.zeros(bucket, HIDDEN // 2, dtype=torch.uint8, device=device),
        torch.zeros(bucket, HIDDEN // 16, dtype=torch.uint8, device=device),
    )


def _statistics(device, enabled):
    return (
        torch.arange(EXPERTS, dtype=torch.float32, device=device),
        torch.full((1,), 5, dtype=torch.int64, device=device),
        torch.full((1,), int(enabled), dtype=torch.int32, device=device),
    )


def _two_launch(logits, bias, x, scale, table, ep_rank, stage, stats, first, word):
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import dsv4_route

    ids, weights, _ = dsv4_route(logits, bias, TOP_K, SCALING, torch.float32, HIDDEN)
    counts, starts, slots, shares = table.split([EXPERTS, EXPERTS, SLOTS, SLOTS])
    staged_ids, staged_weights, x_fp4, x_sf = stage
    tokens = ids.shape[0]
    rows, forwards, enabled = stats if stats is not None else (None, None, None)
    torch.ops.trtllm.adaptive_eplb_route_into(
        ids, counts, starts, slots, shares, ep_rank, staged_ids.view(-1), weights, staged_weights,
        rows, forwards, enabled, first, word, x, scale, x_fp4[:tokens], x_sf[:tokens],
    )
    return ids, weights


def _one_launch(logits, bias, x, scale, table, ep_rank, stage, stats, first, word):
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import AdaptiveRoute, dsv4_route_nvfp4_quant

    ids, weights, _, _, _ = dsv4_route_nvfp4_quant(
        logits, bias, x, scale, TOP_K, SCALING, torch.float32, staged=stage,
        adaptive=AdaptiveRoute(table, ep_rank, stats, first, word),
    )
    return ids, weights


def _inputs(rng, tokens, device, partials):
    g = torch.Generator(device="cpu").manual_seed(rng.randrange(1 << 30))
    logits = torch.randn((2, tokens, EXPERTS) if partials else (tokens, EXPERTS), generator=g).to(device)
    if partials:
        from tensorrt_llm._torch.custom_ops.router_logit_partials import RouterLogitPartials

        logits = RouterLogitPartials(logits)
    bias = (torch.randn(EXPERTS, generator=g) * 0.1).to(device)
    x = torch.randn(tokens, HIDDEN, generator=g).to(device=device, dtype=torch.bfloat16)
    return logits, bias, x, torch.ones(1, dtype=torch.float32, device=device)


def _word(kind, tokens):
    if kind == "none":
        return None
    requests = max(1, tokens // 4)
    real = 0 if kind == "zero" else max(0, requests - 1)
    return torch.tensor([(requests << 16) | real], dtype=torch.int32, device="cuda")


def _assert_equal(a, b):
    (ids_a, w_a, stage_a, stats_a), (ids_b, w_b, stage_b, stats_b) = a, b
    assert torch.equal(ids_a, ids_b), "returned ids"
    assert torch.equal(w_a.view(torch.int32), w_b.view(torch.int32)), "returned weights"
    tokens = ids_a.shape[0]
    assert torch.equal(stage_a[0], stage_b[0]), "staged slots, whole bucket"
    assert torch.equal(stage_a[1].view(torch.int32), stage_b[1].view(torch.int32)), "staged weights, whole bucket"
    assert torch.equal(stage_a[2][:tokens], stage_b[2][:tokens]), "NVFP4 rows"
    assert torch.equal(stage_a[3][:tokens], stage_b[3][:tokens]), "scale rows"
    if stats_a is not None:
        assert torch.equal(stats_a[0].view(torch.int32), stats_b[0].view(torch.int32)), "statistics rows, bit for bit"
        assert torch.equal(stats_a[1], stats_b[1]), "forwards"


def _run_both(rng, tokens, bucket, replicas, word_kind, stats_mode, first, partials, ep_rank=7):
    device = torch.device("cuda")
    table = _table(rng, replicas, device)
    logits, bias, x, scale = _inputs(rng, tokens, device, partials)
    word = _word(word_kind, tokens)
    out = []
    for launch in (_two_launch, _one_launch):
        stage = _stage(bucket, device)
        stats = None if stats_mode == "off" else _statistics(device, stats_mode == "on")
        ids, weights = launch(logits, bias, x, scale, table, ep_rank, stage, stats, first, word)
        torch.cuda.synchronize()
        out.append((ids, weights, stage, stats))
    _assert_equal(*out)


@needs_gpu
@pytest.mark.parametrize("replicas", [0, 40, 128], ids=["home", "mixed", "replicated"])
@pytest.mark.parametrize("tokens,bucket", [(t, 256) for t in (1, 2, 3, 4, 8, 16, 31, 32, 64, 171, 256)]
                         + [(t, 1024) for t in (64, 342, 512, 1024)])
@pytest.mark.parametrize("word_kind", ["none", "padding", "zero"])
def test_one_launch_is_the_two_launches(tokens, bucket, replicas, word_kind):
    rng = random.Random(tokens * 1009 + replicas * 31 + bucket)
    for stats_mode in ("off", "on", "disabled"):
        for first in (True, False):
            _run_both(rng, tokens, bucket, replicas, word_kind, stats_mode, first, partials=False)
    _run_both(rng, tokens, bucket, replicas, word_kind, "on", True, partials=True)


@needs_gpu
@pytest.mark.parametrize("tokens,bucket", [(8, 256), (64, 256), (256, 256), (1024, 1024)])
def test_two_hundred_graph_replays_follow_the_tables(tokens, bucket):
    """Both paths captured once; before each replay the route table is rewritten in place (home / mixed /
    replicated in turn, as _apply_tables publishes) and the inputs and statistics flag change."""
    device = torch.device("cuda")
    rng = random.Random(tokens + bucket)
    tables = [_table(rng, r, device) for r in (0, 40, 128)]
    table = tables[0].clone()
    logits, bias, x, scale = _inputs(rng, tokens, device, partials=False)
    word = torch.tensor([((tokens // 2 or 1) << 16) | max(0, tokens // 2 - 1)], dtype=torch.int32, device=device)
    paths = []
    for launch in (_two_launch, _one_launch):
        stage, stats = _stage(bucket, device), _statistics(device, True)
        launch(logits, bias, x, scale, table, 3, stage, stats, True, word)  # warm up (compile) outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ids, weights = launch(logits, bias, x, scale, table, 3, stage, stats, True, word)
        paths.append((graph, ids, weights, stage, stats))
    for replay in range(200):
        table.copy_(tables[replay % 3])
        logits.copy_(torch.randn_like(logits))
        x.copy_(torch.randn_like(x))
        for _, _, _, stage, stats in paths:
            stats[2].fill_(replay % 4 != 3)
            stage[0].fill_(SENTINEL_SLOT)
            stage[1].fill_(SENTINEL_WEIGHT)
        for graph, *_ in paths:
            graph.replay()
        torch.cuda.synchronize()
        _assert_equal(*[(ids, weights, stage, stats) for _, ids, weights, stage, stats in paths])
