# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stamps are GPU kernels")
@pytest.mark.parametrize("stages", [False, True])
def test_probe_rows_forwards_stages_and_mtp_passes(tmp_path, stages):
    from tensorrt_llm._torch.moe.fused_moe.moe_probe import MoeProbe

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
    assert (
        list(out["forward"][1:4]) == [1, 2, 3] and int(out["forwards"]) == 3 and bool(out["final"])
    )
    assert (out["duration_ns"][1:4][:, [0, 1, 61]] > 0).all() and out["tokens"][2, 61] == 69
    assert ("stage_ns" in out) == stages
    if stages:
        parts = out["stage_ns"][1:4]
        assert (parts[:, [0, 1]] >= 0).all() and (parts[:, [0, 1]].sum(axis=2) > 0).all()
        assert (parts[:, [0, 1]].sum(axis=2) == out["duration_ns"][1:4][:, [0, 1]]).all()
        assert (parts[:, 61] == 0).all()  # a pass without the inner stamps has no parts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stamps are GPU kernels")
def test_probe_records_installs_at_their_forward(tmp_path):
    from tensorrt_llm._torch.moe.fused_moe.moe_probe import MoeProbe

    probe = MoeProbe(str(tmp_path), 0)
    tables = np.zeros((3, 4), dtype=np.int32)
    probe.home = tables.copy()
    for forward in range(4):
        for layer in (0, 1, 2):
            probe.stamp(layer, 0, 8)
            probe.stamp(layer, 1, 8)
        if forward == 1:  # layer 1 takes replicas after forward 1
            tables[1, 2] = 7
            probe.installed(tables, [1])
        if forward == 2:  # layer 1 goes home, layer 2 takes replicas
            tables[1, 2], tables[2, 0] = 0, 5
            probe.installed(tables, [1, 2])
    probe.write(True)
    out = np.load(tmp_path / "rank0.moe-probe.npz")
    assert list(out["install_forward"]) == [2, 3]
    assert out["install_changed"][:, :3].tolist() == [[False, True, False], [False, True, True]]
    assert out["install_replicated"][:, :3].tolist() == [[False, True, False], [False, False, True]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stamps are GPU kernels")
def test_the_writer_thread_leaves_a_graph_capture_on_another_thread_intact(tmp_path, monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe import moe_probe

    monkeypatch.setattr(moe_probe, "POLL_S", 0.001)
    monkeypatch.setattr(moe_probe, "FLUSH_S", 0.0)
    probe = moe_probe.MoeProbe(str(tmp_path), 0)
    probe.stamp(
        0, 0, 8
    )  # a forward, so that the writer copies the ring while the capture below runs
    probe.stamp(0, 1, 8)
    x = torch.zeros(1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(20000):  # long enough on the host for the writer to poll and copy many times
            x.add_(1.0)
    graph.replay()
    torch.cuda.synchronize()
    assert float(x[0]) == 20000.0
