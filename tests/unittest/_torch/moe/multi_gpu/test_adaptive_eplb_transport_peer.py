# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU byte check of the adaptive EPLB peer transport over MNNVL memory."""

import argparse
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
    LayerBlockLayout,
    PeerSlotTransport,
    SlotPlane,
    dedicated_copy_stream,
)


def parser(description: str) -> argparse.ArgumentParser:
    """The check's flags; the benchmark adds its own."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--layers", type=int, default=4, help="layer blocks per rank")
    parser.add_argument("--slots", type=int, default=4, help="slots per layer and rank (even)")
    parser.add_argument(
        "--ranks-per-node", type=int, help="ranks of one node (default: torchrun's local world)"
    )
    parser.add_argument("--hidden", type=int, default=7168, help="expert hidden size")
    parser.add_argument("--intermediate", type=int, default=3072, help="expert intermediate size")
    return parser


def parse(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    args = parser.parse_args(argv)
    if args.slots < 2 or args.slots % 2:
        parser.error("--slots must be even and at least 2")
    return args


def megamoe_planes(hidden: int, intermediate: int, slots: int) -> list[SlotPlane]:
    """The slot planes of the MegaMoE NVFP4 expert layout, as its backend declares them."""
    from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

    bundle = BundleLayout.create(hidden=hidden, intermediate=intermediate)
    planes = []
    for plane in bundle.planes:
        shape, stride, dtype_name, _, storage_alignment = plane.storage_metadata(slots)
        kernel_alignment = plane.kernel_metadata(slots)[-1]
        alignment = math.lcm(int(plane.alignment), storage_alignment, kernel_alignment)
        dtype = getattr(torch, dtype_name)
        planes.append(SlotPlane(plane.name, tuple(shape), tuple(stride), dtype, alignment))
    return planes


def pattern(rank, layer, plane, args, rows, row_bytes, device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(1 + (rank * args.layers + layer) * 16 + plane)
    return torch.randint(
        0, 256, (rows, row_bytes), dtype=torch.uint8, device=device, generator=generator
    )


def peers(rank: int, world: int, per_node: int) -> dict[str, int]:
    node, local = divmod(rank, per_node)
    found = {}
    if per_node > 1:
        found["intra-node"] = node * per_node + (local + 1) % per_node
    if world > per_node:
        found["cross-node"] = (rank + per_node) % world
    return found


@dataclass
class Rig:
    """One rank's transport, filled with the source halves of its slots."""

    transport: PeerSlotTransport
    layout: LayerBlockLayout
    copy_stream: torch.cuda.Stream
    half: int
    peers: dict[str, int]
    add_layer_ms: float


def make_rig(args, rank, world, per_node, device) -> Rig:
    """The transport over ``args.layers`` blocks of the MegaMoE expert layout.

    The source half of this rank's slots is filled with its pattern.
    """
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world, rank=rank, gpus_per_node=per_node, tp_size=world, moe_ep_size=world
    )
    planes = megamoe_planes(args.hidden, args.intermediate, args.slots)
    layout = LayerBlockLayout(planes, args.slots)
    found = peers(rank, world, per_node)
    if not found:
        raise RuntimeError("the check needs at least two GPUs per node or two nodes")
    transport = PeerSlotTransport(mapping)
    add_ms = []
    for layer in range(args.layers):
        begin = time.perf_counter()
        transport.add_layer(layer, layout)
        add_ms.append((time.perf_counter() - begin) * 1e3)
    half = args.slots // 2
    for layer in range(args.layers):
        for plane, rows in enumerate(transport.local_planes(layer).values()):
            rows[:half].copy_(pattern(rank, layer, plane, args, half, rows.shape[1], device))
    torch.cuda.synchronize()
    dist.barrier()
    stream = dedicated_copy_stream(device)
    assert stream.cuda_stream != torch.cuda.current_stream(device).cuda_stream
    return Rig(transport, layout, stream, half, found, max(add_ms))


def copy_on_a_thread(rig: Rig, copies: list, events: tuple) -> None:
    """Issue the copies as the adaptive runtime's worker thread does: from a thread that never sets
    the CUDA device, waiting on the end event and reading the copy time there."""
    failures = []

    def job() -> None:
        try:
            rig.transport.copy(copies, rig.copy_stream, *events)
            events[1].synchronize()
            events[0].elapsed_time(events[1])
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=job)
    worker.start()
    worker.join()
    if failures:
        raise failures[0]


def verify(rig: Rig, args, device, on_thread: bool = False) -> tuple[int, int]:
    """Copy into every destination slot from the peers in one batch, then compare the rows.

    With ``on_thread`` the destinations are cleared first and a worker thread issues the copies.
    Returns the number of rows compared and of rows that differ from their source's pattern.
    """
    sources = list(rig.peers.values())
    copies = [
        (layer, rig.half + j, sources[(layer + j) % len(sources)], j)
        for layer in range(args.layers)
        for j in range(rig.half)
    ]
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    if on_thread:
        for layer in range(args.layers):
            for rows in rig.transport.local_planes(layer).values():
                rows[rig.half :].zero_()
        torch.cuda.synchronize()
        copy_on_a_thread(rig, copies, events)
    else:
        rig.transport.copy(copies, rig.copy_stream, *events)
    rig.copy_stream.synchronize()
    bad = 0
    for layer, dst, src_rank, src in copies:
        for plane, rows in enumerate(rig.transport.local_planes(layer).values()):
            expected = pattern(src_rank, layer, plane, args, rig.half, rows.shape[1], device)
            bad += int(not torch.equal(rows[dst], expected[src]))
    return len(copies) * len(rig.layout.planes), bad


def close_rig(rig: Rig) -> None:
    torch.cuda.synchronize()
    dist.barrier()


def launch(
    check: Callable[[argparse.Namespace, int, int, int, torch.device], dict],
    args: argparse.Namespace,
) -> tuple[int, list[dict]]:
    """Run ``check`` on the torchrun environment's ranks; this rank and every rank's result."""
    os.environ["TLLM_DISABLE_MPI"] = "1"
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    per_node = args.ranks_per_node or int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    try:
        result = check(args, rank, world, per_node, device)
        results = [None] * world
        dist.all_gather_object(results, result)
    finally:
        dist.destroy_process_group()
    return rank, results


def report_verify(results: list[dict]) -> None:
    for r in results:
        where = ""
        if r["cross_peer"] is not None:
            where = f", cross-node peer {r['cross_peer']} on {results[r['cross_peer']]['host']}"
        thread = r.get("thread_bad")  # the copies from a worker thread (the benchmark makes none)
        verdict = "VERIFY OK" if r["verify_bad"] == 0 and not thread else "VERIFY FAIL"
        threaded = "" if thread is None else f", {thread} from a worker thread"
        print(
            f"{verdict} rank {r['rank']} on {r['host']}: {r['verify_rows']} rows, "
            f"{r['verify_bad']} differ{threaded}{where}"
        )


def check(args, rank: int, world: int, per_node: int, device: torch.device) -> dict:
    """Set up, copy and compare on this rank."""
    rig = make_rig(args, rank, world, per_node, device)
    rows, bad = verify(rig, args, device)
    dist.barrier()
    _, thread_bad = verify(rig, args, device, on_thread=True)
    close_rig(rig)
    return dict(
        rank=rank,
        host=socket.gethostname(),
        verify_rows=rows,
        verify_bad=bad,
        thread_bad=thread_bad,
        cross_peer=rig.peers.get("cross-node"),
    )


def run(argv: list[str] | None = None) -> int:
    """Run the check on the torchrun environment's ranks; 0 when every copied byte matches."""
    args = parse(parser(__doc__.splitlines()[0]), argv)
    rank, results = launch(check, args)
    if rank == 0:
        report_verify(results)
    return 0 if all(r["verify_bad"] == r["thread_bad"] == 0 for r in results) else 1


@pytest.mark.gpu4
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_peer_copies_between_four_local_ranks():
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    command += ["--nproc-per-node", "4", __file__, "--layers", "2"]
    env = {**os.environ, "TLLM_DISABLE_MPI": "1"}
    done = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert done.stdout.count("VERIFY OK") == 4, done.stdout[-4000:]


if __name__ == "__main__":
    sys.exit(run())
