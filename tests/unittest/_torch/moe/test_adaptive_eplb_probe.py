# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stamps are GPU kernels")
@pytest.mark.parametrize("stages", [False, True])
def test_probe_rows_forwards_stages_and_mtp_passes(tmp_path, stages):
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.probe import MoeProbe

    probe = MoeProbe(str(tmp_path), 0, stages=stages)
    probe.stamp_point("kernel_start")  # outside a pass: no stamp
    for _ in range(3):  # three forwards: layers 0 and 1, then the MTP layer 61 passed twice
        for layer in (0, 1, 61, 61):
            probe.stamp(layer, 0, 8 + layer)
            if layer < 61:  # the fused-communication layers stamp inside their pass
                for point in ("routed", "kernel_start", "kernel_end"):
                    probe.stamp_point(point)
            probe.stamp(layer, 1, 8 + layer)
    probe.write(True)
    out = np.load(tmp_path / "rank0.moe-probe.npz")
    assert list(out["forward"][1:4]) == [1, 2, 3] and int(out["forwards"]) == 3 and bool(out["final"])
    assert (out["duration_ns"][1:4][:, [0, 1, 61]] > 0).all() and out["tokens"][2, 61] == 69
    assert ("stage_ns" in out) == stages
    if stages:
        parts = out["stage_ns"][1:4]
        assert (parts[:, [0, 1]] >= 0).all() and (parts[:, [0, 1]].sum(axis=2) > 0).all()
        assert (parts[:, [0, 1]].sum(axis=2) == out["duration_ns"][1:4][:, [0, 1]]).all()
        assert (parts[:, 61] == 0).all()  # a pass without the inner stamps has no parts
