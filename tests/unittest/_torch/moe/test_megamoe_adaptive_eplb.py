# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the MegaMoE CuteDSL backend's methods for adaptive EPLB and its route staging."""

import types
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import TimeModel
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import LayerBlockLayout, SlotPlane
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm
from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


def test_slot_weight_planes_are_the_quant_methods_live_planes():
    bundle = BundleLayout.create(hidden=512, intermediate=256)
    slots = 4
    params = {}
    for plane in bundle.planes:
        shape, stride, dtype_name, _, _ = plane.storage_metadata(slots)
        params[plane.name] = torch.empty_strided(shape, stride, dtype=getattr(torch, dtype_name))
    backend = SimpleNamespace(
        quant_method=SimpleNamespace(live_weight_plane_spec=lambda module: bundle),
        compute_slot_count=slots,
        layer_idx=3,
        get_parameter=params.__getitem__,
    )
    declared = IMPL.slot_weight_planes(backend)
    assert [name for name, _ in declared] == [plane.name for plane in bundle.planes]
    planes = []
    for name, alignment in declared:
        param = params[name]
        shape, stride = tuple(param.shape), tuple(param.stride())
        planes.append(SlotPlane(name, shape, stride, param.dtype, alignment))
    assert [plane.slot_bytes for plane in planes] == [plane.nbytes for plane in bundle.planes]
    layout = LayerBlockLayout(planes, slots)
    assert all(layout.offsets[plane.name] % plane.alignment == 0 for plane in planes)
    first = bundle.planes[0].name
    params[first] = torch.empty(1, dtype=torch.uint8)
    with pytest.raises(ValueError, match=f"plane {first!r}"):
        IMPL.slot_weight_planes(backend)


def test_weights_stop_being_movable_once_the_kernel_holds_views():
    backend = SimpleNamespace(_cute_weight_cache={}, layer_idx=5)
    IMPL.check_weights_movable(backend)
    backend._cute_weight_cache["views"] = object()
    with pytest.raises(RuntimeError, match="MoE layer 5 has launched its kernel"):
        IMPL.check_weights_movable(backend)
    assert IMPL.slot_id_dtype.fget(SimpleNamespace(_topk_idx_dtype=torch.int32)) is torch.int32


def test_decode_time_model_scales_with_the_expert_bytes():
    gb300 = IMPL._DECODE_TIME_MODELS["GB300"]
    us, measured = gb300["per_active_slot_us"], gb300["measured_expert_bytes"]
    model = TimeModel(**IMPL.decode_time_model(IMPL, measured, "NVIDIA GB300"))
    assert (model.per_active_slot_us, model.tile_rows) == (us, 256)
    assert (
        IMPL.decode_time_model(IMPL, 2 * measured, "NVIDIA GB300")["per_active_slot_us"] == 2 * us
    )
    assert IMPL.has_decode_time_model("NVIDIA GB300") and not IMPL.has_decode_time_model("A100")
    with pytest.raises(ValueError, match="A100"):
        IMPL.decode_time_model(IMPL, measured, "NVIDIA A100")


def test_route_staging_hands_out_the_staging_of_a_run_that_stages():
    """``route_staging`` (routes) and ``_staging_of_next_run`` hand out a bucket's staging only for
    a forward that ``run_moe`` would stage."""
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
        hidden_size=4,
        _local_staging_cache={},
        _select_launch_max_tokens=lambda hint: 4 if hint <= 4 else 8,
        _acquire_buffers=lambda **kw: acquired.append(kw) or bufs,
    )
    backend._staging_of_next_run = types.MethodType(IMPL._staging_of_next_run, backend)
    staging = IMPL.route_staging
    assert staging(backend, 2, 2) is None  # no bucket hint yet
    backend._active_launch_max_tokens = 3
    assert staging(backend, 2, 2) is None  # the staging is not built
    backend._local_staging_cache[(4, 2, 4, "cpu", torch.bfloat16)] = {}
    got = staging(backend, 2, 2)
    assert got.topk_idx is idx and got.topk_weights is t and got.bucket == 4
    assert staging(backend, 2, 3) is None  # no staging built for that top_k
    assert acquired[-1]["launch_max_T"] == 4 and acquired[-1]["device"] == torch.device("cpu")
    for rows in (4, 5):  # an exact-size direct-input bucket; the next bucket is not built
        assert staging(backend, rows, 2) is None
