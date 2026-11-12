# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``megamoe_stage_routes`` writes a MegaMoE bucket's route staging byte for byte as torch copies
and fills do: the routes widened into rows ``[0, n)``, -1 and 0 in the rows after them, the
weight columns past ``top_k`` untouched; ``megamoe_stage_inputs`` writes the same bytes and, in the
same launch, the activation's NVFP4 bytes as ``fp4_quantize_into`` does. A launch without rows,
such as the priming of the other buckets while tuning, stages -1 and 0 in every row."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_cute_dsl import MegaMoECuteDsl

ROWS, TOP_K, WEIGHT_COLUMNS = 16, 6, 8


def _garbage(shape, dtype, generator):
    """A buffer of random bytes, as a bucket holds after earlier launches."""
    nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=generator)
    return raw.view(dtype).view(shape).cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    "id_dtype,out_dtype",
    [(torch.int32, torch.int64), (torch.int32, torch.int32), (torch.int64, torch.int64)],
)
@pytest.mark.parametrize("rows", [0, 1, 7, ROWS])
# Hidden sizes of 896 and 260 vectors of eight: two runs of 512 per row, the second short, and one
# run shorter than a block; 97 activation rows per route row stride past the resident blocks.
@pytest.mark.parametrize("hidden", [None, 7168, 2080])
def test_stage_routes_matches_torch_copies_and_fills(id_dtype, out_dtype, rows, hidden):
    gen = torch.Generator().manual_seed(rows)
    info = torch.iinfo(torch.int32)
    ids = torch.randint(-1, 384, (rows, TOP_K), dtype=id_dtype, generator=gen)
    if rows:
        ids[0, :2] = torch.tensor([info.max, info.min])  # widening keeps the sign
    ids, weights = ids.cuda(), torch.rand(rows, TOP_K, generator=gen).cuda()
    ids_out = _garbage((ROWS, TOP_K), out_dtype, gen)
    weights_buf = _garbage((ROWS, WEIGHT_COLUMNS), torch.float32, gen)
    want_ids, want_weights = ids_out.clone(), weights_buf.clone()
    want_ids[:rows].copy_(ids)
    want_weights[:rows, :TOP_K].copy_(weights)
    want_ids[rows:].fill_(-1)
    want_weights[rows:, :TOP_K].zero_()
    if hidden is None:
        torch.ops.trtllm.megamoe_stage_routes(ids, weights, ids_out, weights_buf[:, :TOP_K])
    else:
        x = (torch.randn(rows * 97, hidden, generator=gen) * 3).to(torch.bfloat16).cuda()
        x[::3, 16:32] = 0  # a group of zeros: scale factor 0
        scale = torch.tensor([448 * 6 / 7.5], device="cuda")
        fp4 = _garbage((x.shape[0], hidden // 2), torch.uint8, gen)
        sf = _garbage((x.shape[0], hidden // 16), torch.uint8, gen)
        want_fp4, want_sf = fp4.clone(), sf.clone()
        torch.ops.trtllm.fp4_quantize_into(x, scale, want_fp4, want_sf, False)
        torch.ops.trtllm.megamoe_stage_inputs(
            x, scale, fp4, sf, ids, weights, ids_out, weights_buf[:, :TOP_K]
        )
        assert torch.equal(fp4, want_fp4) and torch.equal(sf, want_sf)

    assert torch.equal(ids_out.view(torch.uint8), want_ids.view(torch.uint8))
    assert torch.equal(weights_buf.view(torch.uint8), want_weights.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("rows,bucket", [(0, 16), (5, 16), (16, 16), (37, 64)])
def test_bf16_inputs_stage_in_one_launch_as_in_two(rows, bucket):
    """run_moe's fused prepare without a load balancer (``_stage_bf16_inputs``) leaves a bucket's
    staging byte for byte as the quantization and the route staging leave it, replayed from a CUDA
    graph over new inputs; ``_run_moe`` then finds the activation and the routes staged."""
    gen = torch.Generator().manual_seed(rows)
    shapes = dict(activation=(3584, torch.uint8), activation_sf=(448, torch.uint8))
    shapes.update(topk_idx_local=(TOP_K, torch.int64), topk_weights=(WEIGHT_COLUMNS, torch.float32))
    fused = SimpleNamespace(**{k: _garbage((bucket, n), t, gen) for k, (n, t) in shapes.items()})
    separate = SimpleNamespace(**{k: v.clone() for k, v in vars(fused).items()})
    scale = torch.tensor([448 * 6 / 7.5], device="cuda")
    moe = SimpleNamespace(fc31_input_scale=scale, _staging_of_next_run=lambda n, k: (fused, bucket))
    moe._activation_to_quantize = MegaMoECuteDsl._activation_to_quantize.__get__(moe)
    x = torch.zeros(rows, 7168, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(rows, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.zeros(rows, TOP_K, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = MegaMoECuteDsl._stage_bf16_inputs(moe, x, ids, weights, None)
    assert [t.data_ptr() for t in got] == [
        fused.activation.data_ptr(),
        fused.activation_sf.data_ptr(),
        fused.topk_idx_local.data_ptr(),
    ]
    assert moe._handed_out_route_staging == fused.topk_idx_local.data_ptr()
    for _ in range(2):
        x.copy_(torch.randn(rows, 7168, generator=gen) * 3)
        ids.copy_(torch.randint(-1, 384, (rows, TOP_K), generator=gen))
        weights.copy_(torch.rand(rows, TOP_K, generator=gen))
        graph.replay()
        if rows:
            torch.ops.trtllm.fp4_quantize_into(
                x, scale, separate.activation[:rows], separate.activation_sf[:rows], False
            )
        torch.ops.trtllm.megamoe_stage_routes(
            ids, weights, separate.topk_idx_local, separate.topk_weights[:, :TOP_K]
        )
        for name, want in vars(separate).items():
            assert torch.equal(getattr(fused, name).view(torch.uint8), want.view(torch.uint8)), name


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
    for balancer, fused in ((None, True), (False, True), (True, False)):
        plans = None if balancer is None else SimpleNamespace(produces_replica_plan=balancer)
        assert (
            MegaMoECuteDsl.supports_fused_prepare(SimpleNamespace(layer_load_balancer=plans))
            is fused
        )


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
