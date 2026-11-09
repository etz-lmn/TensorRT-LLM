# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy times of the adaptive EPLB peer transport over MNNVL memory.

Uses the blocks and peers of the byte check
(``tests/unittest/_torch/moe/multi_gpu/test_adaptive_eplb_transport_peer.py``) and times one expert
at a time, one batch over every layer, and a batch of experts on a side stream while the default
stream runs a GEMM loop, whose slowdown shows whether the copies take SMs; then it runs the check.
The flags and the launch are the check's, with ``tests/unittest`` on the path::

    PYTHONPATH=tests/unittest torchrun --nproc-per-node 4 tests/microbenchmarks/bench_adaptive_eplb_transport.py
"""

import socket
import statistics
import sys
import time

import torch
from _torch.moe.multi_gpu import test_adaptive_eplb_transport_peer as peer

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory


def _parse(argv=None):
    parser = peer.parser(__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=20, help="timed single-expert copies")
    parser.add_argument("--group-repeat", type=int, default=5, help="timed install groups")
    parser.add_argument("--compute-iters", type=int, default=400, help="GEMM loop iterations")
    return peer.parse(parser, argv)


def _gemm_loop(x, w, iterations, stream) -> list:
    events = []
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            torch.matmul(x, w)
            end.record(stream)
            events.append((start, end))
    return events


def _median_ms(events) -> float:
    return statistics.median(start.elapsed_time(end) for start, end in events)


def _bench(args, rank: int, world: int, per_node: int, device: torch.device) -> dict:
    rig = peer.make_rig(args, rank, world, per_node, device)
    transport = rig.transport
    copy_stream = rig.copy_stream
    half = rig.half
    main_stream = torch.cuda.Stream(device)
    # The GEMM-loop batch refills every destination slot four times in one group.
    compute_copies = 4 * args.layers * args.slots
    timed: list[tuple] = []
    result = {
        "rank": rank,
        "host": socket.gethostname(),
        "handle_type": str(MnnvlMemory.get_allocation_prop(device.index).requestedHandleTypes),
        "add_layer_ms": rig.add_layer_ms,
        "peers": {},
    }

    def events() -> tuple:
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for event in pair:
            event.record(copy_stream)  # makes the CUDA event outside any timed issue
        timed.append(pair)
        return pair

    def timed_group(copies) -> int:
        transport.copy(copies, copy_stream, *events())
        return len(timed) - 1

    def readings(ids) -> list:
        copy_stream.synchronize()
        return [timed[group][0].elapsed_time(timed[group][1]) for group in ids]

    x = torch.randn(256, 7168, dtype=torch.bfloat16, device=device)
    w = torch.randn(7168, 8192, dtype=torch.bfloat16, device=device)
    for kind, source in rig.peers.items():
        ids = []
        issue_us = []
        for i in range(-2, args.repeat):
            pair = events()
            begin = time.perf_counter()
            copy = (i % args.layers, half + i % half, source, i % half)
            transport.copy([copy], copy_stream, *pair)
            issue_us.append((time.perf_counter() - begin) * 1e6)
            ids.append(len(timed) - 1)
            copy_stream.synchronize()
        single_ms = readings(ids)[2:]
        layer_copies = [
            (layer, half + layer % half, source, layer % half) for layer in range(args.layers)
        ]
        group_ms = readings([timed_group(layer_copies) for _ in range(args.group_repeat)])
        idle = _gemm_loop(x, w, args.compute_iters, main_stream)
        main_stream.synchronize()
        busy = _gemm_loop(x, w, args.compute_iters, main_stream)
        batch = [
            (i % args.layers, half + i % half, source, i % half) for i in range(compute_copies)
        ]
        (batch_ms,) = readings([timed_group(batch)])
        main_stream.synchronize()
        result["peers"][kind] = {
            "peer": source,
            "expert_ms": statistics.median(single_ms),
            "expert_ms_max": max(single_ms),
            "issue_us": statistics.median(issue_us[2:]),
            "group_ms": statistics.median(group_ms),
            "under_gemm_ms": batch_ms / compute_copies,
            "gemm_slowdown": _median_ms(busy) / _median_ms(idle) - 1.0,
        }
    rows, bad = peer.verify(rig, args, device)
    peer.close_rig(rig)
    result.update(
        verify_rows=rows,
        verify_bad=bad,
        cross_peer=rig.peers.get("cross-node"),
        expert_bytes=rig.layout.expert_bytes,
        planes=len(rig.layout.planes),
    )
    return result


def _report(results: list, args) -> None:
    first = results[0]
    hosts = sorted({r["host"] for r in results})
    print(
        f"adaptive EPLB peer transport: {len(results)} ranks on {len(hosts)} nodes, "
        f"{args.layers} layers x {args.slots} slots, one expert = {first['expert_bytes']} bytes "
        f"in {first['planes']} planes, handle {first['handle_type']}, add_layer worst "
        f"{max(r['add_layer_ms'] for r in results):.1f} ms"
    )
    print(
        "peer         expert ms med/worst   GB/s   issue us   group ms   ms/layer   GB/s"
        "   ms/expert under GEMM   GEMM slowdown"
    )
    for name in ("intra-node", "cross-node"):
        rows = [r["peers"][name] for r in results if name in r["peers"]]
        if not rows:
            continue
        expert = max(row["expert_ms"] for row in rows)
        worst = max(row["expert_ms_max"] for row in rows)
        group = max(row["group_ms"] for row in rows)
        per_layer = group / args.layers
        print(
            f"{name:<12} {expert:8.4f} / {worst:8.4f}  {first['expert_bytes'] / expert / 1e6:6.0f}"
            f"   {max(row['issue_us'] for row in rows):7.1f}   {group:8.4f}   {per_layer:8.4f}"
            f"   {first['expert_bytes'] / per_layer / 1e6:5.0f}"
            f"   {max(row['under_gemm_ms'] for row in rows):19.4f}"
            f"   {max(row['gemm_slowdown'] for row in rows):+13.1%}"
        )
    peer.report_verify(results)


def main(argv=None) -> int:
    """Time and check the copies on the torchrun environment's ranks; 0 when every byte matches."""
    args = _parse(argv)
    rank, results = peer.launch(_bench, args)
    if rank == 0:
        _report(results, args)
    return 0 if all(r["verify_bad"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
