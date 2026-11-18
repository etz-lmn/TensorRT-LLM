# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``megamoe_stage_routes`` writes a MegaMoE bucket's route staging byte for byte as torch copies
and fills do: the routes widened into rows ``[0, n)``, -1 and 0 in the rows after them. A launch
without rows, such as the priming of the other buckets while tuning, stages -1 and 0 in every row.
With the forward's live-row word, only the word's live rows get their routes, on the CPU and on the
device alike. The staging refuses routes whose row count is not the launch's token count."""

import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe import live_rows
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl

ROWS, TOP_K = 16, 6

_DEVICES = [
    pytest.param("cpu", marks=pytest.mark.cpu_only),
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
    ),
]


def _garbage(shape, dtype, generator, device):
    """A buffer of random bytes, as a bucket holds after earlier launches."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    return raw.view(dtype).view(shape).to(device)


def _cuda_last_error() -> int:
    """cudaPeekAtLastError of the CUDA runtime this process loaded (0: none pending), without clearing it."""
    import ctypes

    path = next(
        line.split()[-1]
        for line in open("/proc/self/maps")
        if "libcudart.so" in line and line.split()[-1].startswith("/")
    )
    return int(ctypes.CDLL(path).cudaPeekAtLastError())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_stage_routes_first_launch_of_the_process_leaves_no_error(record_property):
    """The first route staging of the process, of a launch without rows (an empty grid of routes, a full bucket of
    -1 and 0): no CUDA error is pending before it, none after it, and it stages what torch fills. Records where an
    error first shows (before / after) for a run that fails."""
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    before = _cuda_last_error()
    ids = torch.empty((0, TOP_K), dtype=torch.int32, device="cuda")
    weights = torch.empty((0, TOP_K), device="cuda")
    ids_out = torch.full((ROWS, TOP_K), 7, dtype=torch.int64, device="cuda")
    weights_out = torch.full((ROWS, TOP_K), 7.0, device="cuda")
    error = None
    try:
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised by the assertion below
        error = repr(exc)
    after = _cuda_last_error()
    record_property("first_stage_routes_launch", f"before={before} after={after} error={error}")
    assert before == 0 and error is None and after == 0, (before, error, after)
    assert (ids_out == -1).all() and (weights_out == 0).all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "id_dtype,out_dtype",
    [(torch.int32, torch.int64), (torch.int32, torch.int32), (torch.int64, torch.int64)],
)
@pytest.mark.parametrize("rows", [0, 1, 7, ROWS])
@pytest.mark.parametrize("hidden", [None, 7168, 2080])
def test_stage_routes_matches_torch_copies_and_fills(device, id_dtype, out_dtype, rows, hidden):
    if device == "cpu" and hidden is not None:
        pytest.skip("fused quantization requires CUDA")
    gen = torch.Generator().manual_seed(rows)
    info = torch.iinfo(torch.int32)
    ids = torch.randint(-1, 384, (rows, TOP_K), dtype=id_dtype, generator=gen)
    if rows:
        ids[0, :2] = torch.tensor([info.max, info.min])  # widening keeps the sign
    ids = ids.to(device)
    weights = torch.rand(rows, TOP_K, generator=gen).to(device)
    ids_out = _garbage((ROWS, TOP_K), out_dtype, gen, device)
    weights_out = _garbage((ROWS, TOP_K), torch.float32, gen, device)
    want_ids, want_weights = ids_out.clone(), weights_out.clone()
    want_ids[:rows].copy_(ids)
    want_weights[:rows].copy_(weights)
    want_ids[rows:].fill_(-1)
    want_weights[rows:].zero_()

    if hidden is None:
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out)
    else:
        x = (torch.randn(rows * 97, hidden, generator=gen) * 3).to(torch.bfloat16).cuda()
        x[::3, 16:32] = 0
        scale = torch.tensor([448 * 6 / 7.5], device="cuda")
        fp4 = _garbage((x.shape[0], hidden // 2), torch.uint8, gen, device)
        sf = _garbage((x.shape[0], hidden // 16), torch.uint8, gen, device)
        want_fp4, want_sf = fp4.clone(), sf.clone()
        torch.ops.trtllm.fp4_quantize.into(x, scale, want_fp4, want_sf, False)
        torch.ops.trtllm.megamoe_stage_inputs(x, scale, fp4, sf, ids, weights, ids_out, weights_out)
        assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)

    assert torch.equal(ids_out.view(torch.uint8), want_ids.view(torch.uint8))
    assert torch.equal(weights_out.view(torch.uint8), want_weights.view(torch.uint8))


@pytest.mark.parametrize("device", _DEVICES)
def test_stage_routes_stages_only_the_live_rows_of_the_word(device):
    """The op computes the host reference's live rows (``live_rows.live_row_count``) from the
    word on the device; the rows after them get -1 and 0 like the bucket's tail."""
    top = live_rows.MAX_REQUESTS
    words = [
        live_rows.NO_ROWS,
        live_rows.ALL_ROWS,
        live_rows.encode(0, 4),
        live_rows.encode(3, 4),  # 4 requests of 3 rows: 9 of 12 rows live
        live_rows.encode(4, 4),
        (4 << live_rows.REQUEST_BITS) | 5,  # more real requests than requests: every row
        live_rows.encode(1, 5),  # 5 requests do not divide 12 rows: every row
        live_rows.encode(1, 32768),  # the sign bit of the word set
        live_rows.encode(top - 1, top),
    ]
    gen = torch.Generator().manual_seed(0)
    for n in (0, 1, 12, ROWS):
        ids = torch.randint(0, 384, (n, TOP_K), dtype=torch.int32, generator=gen).to(device)
        weights = torch.rand(n, TOP_K, generator=gen).to(device)
        for value in words:
            word = torch.tensor([value], dtype=torch.int64).to(torch.int32).to(device)
            ids_out = _garbage((ROWS, TOP_K), torch.int64, gen, device)
            weights_out = _garbage((ROWS, TOP_K), torch.float32, gen, device)
            live = live_rows.live_row_count(value, n)
            want_ids = torch.full((ROWS, TOP_K), -1, dtype=torch.int64, device=device)
            want_weights = torch.zeros(ROWS, TOP_K, device=device)
            want_ids[:live].copy_(ids[:live])
            want_weights[:live].copy_(weights[:live])

            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out, word)

            assert torch.equal(ids_out, want_ids), (n, hex(value))
            assert torch.equal(weights_out.view(torch.uint8), want_weights.view(torch.uint8))


@pytest.mark.cpu_only
def test_stage_routes_refuses_a_word_it_cannot_read():
    ids, weights = torch.zeros(2, TOP_K, dtype=torch.int64), torch.zeros(2, TOP_K)
    ids_out, weights_out = torch.empty(ROWS, TOP_K, dtype=torch.int64), torch.empty(ROWS, TOP_K)
    for word in (torch.zeros(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int32)):
        with pytest.raises(RuntimeError, match="live_row_word"):
            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out, word)


@pytest.mark.cpu_only
def test_stage_routes_refuses_weights_it_would_write_through_a_strided_view():
    ids = torch.zeros(2, TOP_K, dtype=torch.int64)
    weights = torch.zeros(2, TOP_K)
    ids_out = torch.empty(ROWS, TOP_K, dtype=torch.int64)
    for weights_out in (
        torch.empty(ROWS, TOP_K + 2)[:, :TOP_K],  # rows padded past top_k
        torch.empty(1, TOP_K).expand(ROWS, TOP_K),  # every row the same cells
    ):
        with pytest.raises(RuntimeError, match="contiguous"):
            torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_out)


@pytest.mark.cpu_only
def test_staging_refuses_routes_of_another_row_count():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._wait_rebalance_routes = lambda replica_plan=None: None
    bufs = SimpleNamespace(
        topk_idx_local=torch.zeros(8, TOP_K, dtype=torch.int64), topk_weights=torch.zeros(8, TOP_K)
    )
    for routes in (3, 5):
        with pytest.raises(ValueError, match=f"got {routes} route rows for 4 tokens"):
            moe._stage_inputs(
                bufs=bufs,
                x=None,
                x_sf=None,
                topk_idx=torch.zeros(routes, TOP_K, dtype=torch.int64),
                topk_weights=torch.zeros(routes, TOP_K),
                num_tokens=4,
                top_k=TOP_K,
                stage_activation=False,
            )


@pytest.mark.cpu_only
def test_staging_drops_the_rows_the_forward_word_drops():
    """``_stage_inputs`` hands the forward's word to the op: of 4 rows of 2 requests, 1 real, the
    leading 2 keep their routes."""
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._wait_rebalance_routes = lambda replica_plan=None: None
    bufs = SimpleNamespace(
        topk_idx_local=torch.zeros(8, TOP_K, dtype=torch.int64), topk_weights=torch.zeros(8, TOP_K)
    )
    ids = torch.arange(4 * TOP_K, dtype=torch.int32).view(4, TOP_K)
    moe._stage_inputs(
        bufs=bufs,
        x=None,
        x_sf=None,
        topk_idx=ids,
        topk_weights=torch.ones(4, TOP_K),
        num_tokens=4,
        top_k=TOP_K,
        stage_activation=False,
        live_row_word=torch.tensor([live_rows.encode(1, 2)], dtype=torch.int32),
    )
    assert torch.equal(bufs.topk_idx_local[:2], ids[:2].long())
    assert (bufs.topk_idx_local[2:] == -1).all()
    assert bufs.topk_weights[:2].eq(1).all() and not bufs.topk_weights[2:].any()


@pytest.mark.cpu_only
def test_bucket_priming_stages_a_launch_without_rows():
    moe = MegaMoECuteDsl.__new__(MegaMoECuteDsl)
    torch.nn.Module.__init__(moe)
    moe._maxt_buckets = [8, 16]
    moe.ep_size, moe.ep_rank = 4, 0
    moe._wait_rebalance_routes = lambda replica_plan=None: None
    bufs = SimpleNamespace(
        topk_idx_local=torch.full((8, TOP_K), 5, dtype=torch.int64),
        topk_weights=torch.ones(8, TOP_K),
        activation=None,
        activation_sf=None,
        combine_output=None,
        shared_workspace=None,
        peer_offsets=(0,),
    )
    moe._acquire_buffers = lambda **kwargs: bufs
    launches = []
    moe._launch_megamoe_kernel = lambda **kwargs: launches.append(kwargs)

    moe._prime_ladder_buckets(
        live_T=16,
        top_k=TOP_K,
        hidden=16,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
        weight_view=None,
        primed=[],
    )

    assert [(launch["launch_max_T"], launch["num_tokens"]) for launch in launches] == [(8, 0)]
    assert (bufs.topk_idx_local == -1).all() and not bufs.topk_weights.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("rows,bucket", [(0, 16), (5, 16), (16, 16), (37, 64)])
def test_bf16_inputs_stage_in_one_launch_as_in_two(rows, bucket, monkeypatch):
    """run_moe's fused prepare without a load balancer (``_stage_bf16_inputs``) leaves a bucket's
    staging byte for byte as the quantization and the route staging leave it, replayed from a CUDA
    graph over new inputs; ``_run_moe`` then finds the activation and the routes staged."""
    gen = torch.Generator().manual_seed(rows)
    shapes = dict(activation=(3584, torch.uint8), activation_sf=(448, torch.uint8))
    shapes.update(topk_idx_local=(TOP_K, torch.int64), topk_weights=(TOP_K, torch.float32))
    fused = SimpleNamespace(
        **{k: _garbage((bucket, n), t, gen, "cuda") for k, (n, t) in shapes.items()}
    )
    separate = SimpleNamespace(**{k: v.clone() for k, v in vars(fused).items()})
    scale = torch.tensor([448 * 6 / 7.5], device="cuda")
    moe = SimpleNamespace(
        fc31_input_scale=scale,
        _staging_of_next_run=lambda n, device: fused,
        _padding_drop_buckets={bucket},
        _shared_expert=None,
        _slot_of_expert=None,
        _prestaged=None,
    )
    moe._activation_to_quantize = MegaMoECuteDsl._activation_to_quantize.__get__(moe)
    moe._shared_activation_target = MegaMoECuteDsl._shared_activation_target.__get__(moe)
    x = torch.zeros(rows, 7168, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(rows, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.zeros(rows, TOP_K, device="cuda")
    word = torch.tensor([live_rows.encode(0, 1)], dtype=torch.int32, device="cuda")
    monkeypatch.setattr(live_rows, "current_word", lambda device: word)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = MegaMoECuteDsl._stage_bf16_inputs(moe, x, ids, weights, None)
    for view, full in zip(got, (fused.activation, fused.activation_sf, fused.topk_idx_local)):
        assert view.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
        assert view.storage_offset() == full.storage_offset()
        assert view.shape == (rows, *full.shape[1:]) and view.stride() == full.stride()
        assert view.dtype == full.dtype and view.device == full.device
    assert moe._handed_out_route_staging == fused.topk_idx_local.data_ptr()
    for value in (live_rows.NO_ROWS, live_rows.ALL_ROWS):
        word.fill_(value)
        x.copy_(torch.randn(rows, 7168, generator=gen) * 3)
        ids.copy_(torch.randint(-1, 384, (rows, TOP_K), generator=gen))
        weights.copy_(torch.rand(rows, TOP_K, generator=gen))
        graph.replay()
        if rows:
            torch.ops.trtllm.fp4_quantize.into(
                x, scale, separate.activation[:rows], separate.activation_sf[:rows], False
            )
        torch.ops.trtllm.megamoe_stage_routes(
            ids, weights, separate.topk_idx_local, separate.topk_weights[:, :TOP_K], word
        )
        for name, want in vars(separate).items():
            assert torch.equal(getattr(fused, name).view(torch.uint8), want.view(torch.uint8)), name


@pytest.mark.cpu_only
def test_bf16_inputs_launch_apart_where_the_next_run_stages_nothing(monkeypatch):
    """Without a staging (inputs read in place, tactic autotuning, SF rows that need padding) the fused
    prepare launches a load balancer's route alone, then quantize_input; a layer whose balancer
    plans replicas keeps quantize_input, which leaves SMs to the plan's HALO-Q/TMA kernel."""
    calls = []
    moe = SimpleNamespace(
        _staging_of_next_run=lambda rows, top_k: calls.append("staging"),
        quantize_input=lambda x: calls.append("quantize") or ("fp4", "sf"),
    )
    ids = torch.zeros(4, TOP_K, dtype=torch.int32)

    def route(quantize=None):
        calls.append(("route", quantize))
        return "slots"

    for hidden, looked in (
        (7168, ["staging"]),
        (2080, []),
    ):  # 2080 / 16 = 130 SF bytes, padded to 132
        calls.clear()
        got = MegaMoECuteDsl._stage_bf16_inputs(moe, torch.zeros(4, hidden), ids, None, route)
        assert got == ("fp4", "sf", "slots") and calls == looked + [("route", None), "quantize"]
    monkeypatch.setattr(torch.ops.trtllm, "megamoe_stage_inputs", object(), raising=False)
    for balancer, dynamic, fused in (
        (None, False, True),
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ):
        plans = (
            None
            if balancer is None
            else SimpleNamespace(produces_replica_plan=balancer, is_dynamic_routing=lambda: dynamic)
        )
        assert (
            MegaMoECuteDsl.supports_fused_prepare(SimpleNamespace(layer_load_balancer=plans))
            is fused
        )


# ---------------------------------------------------------------------------------------------------------------------
# The adaptive route of the one-round dispatch router (adaptive_eplb/router_route.py): the router's own device code
# (_OneRoundRouter._adaptive_counted_rows / _adaptive_pick), launched one warp per token row as the router launches it,
# against the module's plain-integer definition, bit for bit; and the kernels a build without the route compiles.
# ---------------------------------------------------------------------------------------------------------------------

_requires_cute_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs a Blackwell GPU",
)
_AR_WORLD, _AR_SLOTS, _AR_EXPERTS, _AR_TOP_K = 32, 512, 384, 6


def _router_route_module():
    import importlib.util
    import pathlib

    import tensorrt_llm

    path = (
        pathlib.Path(tensorrt_llm.__file__).parent
        / "_torch/moe/fused_moe/adaptive_eplb/router_route.py"
    )
    spec = importlib.util.spec_from_file_location("router_route", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _route_table(rng, zero_one):
    """A packed route table row of 384 experts in 512 slots of 32 GPUs: every expert at home, the 128
    spare slots holding replicas, a share vector with some 0/1 copies (policy.build_route_tables)."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        ExpertGeometry,
        build_route_tables,
    )

    geometry = ExpertGeometry(_AR_EXPERTS, _AR_SLOTS, _AR_WORLD)
    layout = geometry.home_layout().tolist()
    free = [s for s, e in enumerate(layout) if e < 0]
    rng.shuffle(free)
    hot = rng.sample(range(_AR_EXPERTS), 40)
    for slot in free[:96]:
        layout[slot] = rng.choice(hot)
    shares = [rng.random() for _ in range(_AR_SLOTS)]
    by_expert = {}
    for slot, e in enumerate(layout):
        if e >= 0:
            by_expert.setdefault(e, []).append(slot)
    for slots in by_expert.values():
        if len(slots) > 1 and rng.random() < zero_one:
            keep = rng.randrange(len(slots))
            for k, slot in enumerate(slots):
                shares[slot] = 1.0 if k == keep else 0.0
    return build_route_tables(layout, shares, _AR_EXPERTS, _AR_SLOTS).packed()


def _compile_router_probe(rows):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cutlass_dsl import Int32, Int64

    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.api import ImplDesc, ProblemDesc
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain.one_round_router import (
        _OneRoundRouter,
    )

    router = _OneRoundRouter(
        ProblemDesc(
            {
                "world_size": _AR_WORLD,
                "expert_count": _AR_SLOTS,
                "topk": _AR_TOP_K,
                "max_tokens_per_rank": rows,
                "hidden_size": 7168,
                "quant_kind": "nvfp4",
                "apply_topk_at_fc1": True,
            }
        ),
        ImplDesc(
            {
                "token_padding_block": 64,
                "one_round_cta_count": 512 // _AR_WORLD,
                "adaptive_route_experts": _AR_EXPERTS,
            }
        ),
    )
    warps = 4

    class Probe:
        @cute.kernel
        def kernel(
            self,
            ids: cute.Tensor,
            out: cute.Tensor,
            tables: Int64,
            stat_rows: Int64,
            forwards: Int64,
            enabled: Int64,
            word: Int64,
            num_tokens: Int32,
            first: Int32,
            ep_rank: Int32,
        ):
            thread_idx, _, _ = cute.arch.thread_idx()
            cta_idx, _, _ = cute.arch.block_idx()
            warp = thread_idx // Int32(32)
            lane = thread_idx % Int32(32)
            adaptive = (tables, stat_rows, forwards, enabled, word, num_tokens, first)
            counted = router._adaptive_counted_rows(adaptive, cta_idx, thread_idx)
            token = cta_idx * Int32(warps) + warp
            if token < Int32(rows):
                if lane < Int32(_AR_TOP_K):
                    raw = ids[token, lane]
                    slot = Int32(-1)
                    if cutlass.Uint64(raw) < cutlass.Uint64(_AR_EXPERTS):
                        slot = router._adaptive_pick(adaptive, counted, Int32(raw), token, ep_rank)
                    out[token, lane] = slot

        @cute.jit
        def __call__(
            self,
            ids: cute.Tensor,
            out: cute.Tensor,
            tables: Int64,
            stat_rows: Int64,
            forwards: Int64,
            enabled: Int64,
            word: Int64,
            num_tokens: Int32,
            first: Int32,
            ep_rank: Int32,
            stream,
        ):
            self.kernel(
                ids, out, tables, stat_rows, forwards, enabled, word, num_tokens, first, ep_rank
            ).launch(
                grid=[(rows + warps - 1) // warps, 1, 1], block=[warps * 32, 1, 1], stream=stream
            )

    fake = cute.runtime.make_fake_compact_tensor
    return cute.compile(
        Probe(),
        fake(cutlass.Int64, (rows, _AR_TOP_K), stride_order=(1, 0)),
        fake(cutlass.Int32, (rows, _AR_TOP_K), stride_order=(1, 0)),
        *(Int64(0),) * 5,
        Int32(0),
        Int32(1),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@_requires_cute_gpu
@pytest.mark.parametrize("rows", [8, 32, 64, 256])
def test_router_adaptive_route_is_bitwise_its_definition(rows):
    """Slots and statistics of the router's adaptive route equal router_route.py's, with replicated experts,
    0/1 shares, invalid ids, padding rows past the forward's rows, live-row words and the recording flag."""
    import random

    rr = _router_route_module()
    rng = random.Random(rows)
    compiled = _compile_router_probe(rows)
    device = torch.device("cuda")
    for case in range(12):
        packed = _route_table(rng, zero_one=0.4 if case % 2 else 0.0)
        table = torch.tensor(packed, dtype=torch.int32, device=device)
        num_tokens = rng.randint(1, rows)
        ids = [[rng.randrange(_AR_EXPERTS) for _ in range(_AR_TOP_K)] for _ in range(rows)]
        for t in range(rows):
            if t >= num_tokens:
                ids[t] = [-1] * _AR_TOP_K
            elif rng.random() < 0.1:
                ids[t][rng.randrange(_AR_TOP_K)] = rng.choice([-1, _AR_EXPERTS, _AR_SLOTS + 3])
        requests = rng.choice([0, 1, 2, 4, num_tokens])
        real = rng.randint(0, requests) if requests else 0
        word = None if case % 3 == 0 else (requests << 16) | real
        enabled = case % 4 != 3
        first = case % 5 != 4
        ep_rank = rng.randrange(_AR_WORLD)

        stat_rows = torch.zeros(_AR_EXPERTS, dtype=torch.float32, device=device)
        forwards = torch.zeros(1, dtype=torch.int64, device=device)
        flag = torch.tensor([1 if enabled else 0], dtype=torch.int32, device=device)
        word_t = torch.tensor(
            [0x7FFFFFFF if word is None else word], dtype=torch.int32, device=device
        )
        out = torch.full((rows, _AR_TOP_K), -7, dtype=torch.int32, device=device)
        ids_t = torch.tensor(ids, dtype=torch.int64, device=device)
        compiled(
            ids_t,
            out,
            table.data_ptr(),
            stat_rows.data_ptr(),
            forwards.data_ptr(),
            flag.data_ptr(),
            word_t.data_ptr(),
            num_tokens,
            1 if first else 0,
            ep_rank,
        )
        torch.cuda.synchronize()

        want_slots = rr.slots_of(ids, ep_rank, packed, _AR_EXPERTS, _AR_SLOTS)
        want_rows, want_forwards = [0.0] * _AR_EXPERTS, [0]
        # A word that keeps every row reads as 0x7FFF real of 0x7FFF requests.
        rr.record(
            want_rows,
            want_forwards,
            ids,
            num_tokens,
            0x7FFFFFFF if word is None else word,
            enabled,
            first,
        )
        assert out.cpu().tolist() == want_slots, (rows, case)
        assert stat_rows.cpu().tolist() == want_rows, (rows, case)
        assert forwards.item() == want_forwards[0], (rows, case)


def _kernel_ptx_digests(adaptive_route_experts, out_dir, tokens=32):
    """Build and compile the DeepSeek-V4 EP32 MegaMoE kernel of a 32-token bucket in a fresh interpreter with its
    PTX and cubins kept: {file: sha256 of the PTX without source locations, of the cubin}, and the resource usage."""
    import json
    import subprocess
    import sys

    script = f"""
import hashlib, json, os, pathlib, re, subprocess, torch
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
kwargs = dict(world_size=32, local_rank=0, num_topk=6, num_experts_per_rank=16, hidden_size=7168,
              intermediate_size_per_partition=3072, expand_intermediate_size_per_partition=6144,
              max_tokens_per_rank={tokens}, output_dtype=torch.bfloat16, apply_topk_in_fc1=True, swiglu_alpha=None,
              swiglu_beta=None, gate_up_clamp=None, situ_beta=None, situ_linear_beta=None, in_kernel_fc2_reduce=False,
              combine_format="16e2m1xbf16", tactic_autotune=False, helper_expert_count=0)
if {adaptive_route_experts}:
    kwargs["adaptive_route_experts"] = {adaptive_route_experts}
# The kernel builder reads the device's limits through the driver: it needs this process's CUDA context.
torch.cuda.set_device(0)
torch.zeros(1, device="cuda")
out = pathlib.Path({str(out_dir)!r})
out.mkdir(parents=True, exist_ok=True)
import cutlass.cute as cute
_compile = cute.compile
class _KeepCode:
    # cute.compile with the DSL's keep-PTX / keep-cubin dump of every kernel it builds into `out`.
    keep = " --keep-ptx --keep-cubin --dump-dir " + str(out)
    def _with(self, fn):
        return lambda *a, **k: fn(*a, **dict(k, options=(k.get("options", "") + self.keep).strip()))
    def __getitem__(self, item):
        return self._with(_compile[item])
    def __call__(self, *a, **k):
        return self._with(_compile)(*a, **k)
cute.compile = _KeepCode()
runner = op._megamoe_get_runner(**kwargs)
tactic = op._default_megamoe_tactic_for_key(runner._tactic_lookup_key(), num_tokens={tokens})
kernel = runner._build_kernel_once(tactic)
kernel.aot_compile()
digests = {{}}
for f in sorted(out.rglob("*")):
    if f.suffix == ".ptx":
        text = "\\n".join(l for l in f.read_text().splitlines() if not re.match(r"\\s*(\\.loc|\\.file|//)", l))
        digests[f.name] = ["ptx", hashlib.sha256(text.encode()).hexdigest()]
    elif f.suffix == ".cubin":
        try:
            dump = ["cuobjdump", "--dump-resource-usage", str(f)]
            usage = subprocess.run(dump, capture_output=True, text=True).stdout
        except OSError as error:
            usage = f"cuobjdump: {{error}}"
        digests[f.name] = ["cubin", hashlib.sha256(f.read_bytes()).hexdigest(), usage]
print("DIGESTS " + json.dumps(digests))
"""
    env = dict(
        os.environ, CUTE_DSL_KEEP_PTX="1", CUTE_DSL_KEEP_CUBIN="1", CUTE_DSL_DUMP_DIR=str(out_dir)
    )
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=1800
    )
    assert done.returncode == 0, done.stderr[-4000:]
    line = next(x for x in done.stdout.splitlines() if x.startswith("DIGESTS "))
    return json.loads(line[len("DIGESTS ") :])


@_requires_cute_gpu
@pytest.mark.parametrize("tokens", [32, 1024])
def test_megamoe_kernels_without_the_adaptive_route_record_their_ptx(
    tmp_path, record_property, tokens
):
    """The no-EPLB EP32 build's kernels of a 32-token bucket (the one-round router) and a 1024-token bucket (the
    metadata-push router), digested (PTX without source locations, cubin) into the JUnit properties of each arm of
    the GPU unit run: base and change must record the same PTX digests. The adaptive build's resource usage is
    recorded beside them where the arm has the route."""
    import json

    digests = _kernel_ptx_digests(0, tmp_path / "base", tokens)
    kept = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert any(kind == "ptx" for kind, *_ in digests.values()), (
        f"the compile kept no PTX: {kept[:20]}"
    )
    record_property(
        "no_eplb_kernel_digests", json.dumps({k: v[:2] for k, v in digests.items()}, sort_keys=True)
    )
    record_property(
        "no_eplb_resource_usage",
        json.dumps({k: v[2] for k, v in digests.items() if v[0] == "cubin"}),
    )
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain import (
        one_round_router,
    )

    if hasattr(one_round_router._OneRoundRouter, "_adaptive_pick"):
        adaptive = _kernel_ptx_digests(_AR_EXPERTS, tmp_path / "adaptive", tokens)
        record_property(
            "adaptive_resource_usage",
            json.dumps({k: v[2] for k, v in adaptive.items() if v[0] == "cubin"}),
        )
