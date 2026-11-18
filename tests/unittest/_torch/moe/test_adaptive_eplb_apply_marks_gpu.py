# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU diagnostic of the main thread's publish of adaptive EPLB route tables.

Runs with TLLM_ADAPTIVE_EPLB_APPLY_MARKS=1.

Four ranks on one node, one GPU each, run the real runtime (worker thread, CUDA events, pinned host
tables, the copy stream) through install / retire / unchanged windows while the GPU runs a stand-in
forward every step, and log the host time of every step of each publish. The decisions are scripted
and every rank decides alike, so a rank all-gathers by itself; the decision process's answer takes
the time of a blocking pipe round trip (the worker thread releases the interpreter meanwhile).
Prints one ``APPLY_MARKS_SUMMARY`` line per rank: per step, the median and maximum microseconds.
"""

import json
import os
import pickle
import statistics
import sys
import time

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

_WORLD_SIZE = 4
_LAYERS, _EXPERTS, _SLOTS = 61, 48, 64  # per rank 12 experts in 16 slots, as DeepSeek-V4 at EP32
_FORWARDS, _OPEN_AFTER = 8, 4
_WINDOWS = 24
_EXPERT_BYTES = 37158924  # one DeepSeek-V4 NVFP4 expert
_RPC_S = 0.0004  # the decision process's answer

# The workers run this module's functions by value (the module is not importable by name there).
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(path):
    sys.path[:] = path


def _rank_run():
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        ExpertStatistics,
        statistics_buffers,
    )
    from tensorrt_llm._utils import mpi_rank

    rank = mpi_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    os.environ["TLLM_ADAPTIVE_EPLB_APPLY_MARKS"] = "1"
    marks = []
    log = rt.AdaptiveRuntime._log_apply_marks

    def keep(update, created, steps):
        marks.append((len(update.tables), created, [(n, t) for n, t in steps]))
        log(update, created, steps)

    rt.AdaptiveRuntime._log_apply_marks = staticmethod(keep)
    buffers = statistics_buffers(_LAYERS, _EXPERTS, device)
    flag = torch.ones(1, dtype=torch.int32, device=device)

    class Layer:
        expert_count, num_slots = _EXPERTS, _SLOTS

        def __init__(self, index):
            self.statistics = ExpertStatistics.of(buffers, index, flag)

    table = 2 * (_EXPERTS + _SLOTS)

    def update_of(window):
        kind = window % 3  # 0 unchanged, 1 install (8 layers, one copied expert), 2 retire
        if kind == 0:
            update = rt.LayoutUpdate(window)
        else:
            tables = {layer: np.full(table, window, np.int32) for layer in range(8)}
            moved = [0] if kind == 1 else []
            items = {0: [(12, (rank + 1) % _WORLD_SIZE, 0)]} if kind == 1 else {}
            update = rt.LayoutUpdate(window, copy_items=items, tables=tables, moved=moved)
        update.digest = update.fingerprint()
        return update

    class Decision:
        stats = torch.zeros(_WORLD_SIZE, rt.window_stats_size(_LAYERS, _EXPERTS))

        def ask(self, message):
            time.sleep(_RPC_S)
            if message[0] == "window":
                return ("update", update_of(message[1]))
            apply_call = int(message[2][:, rt.ROW.index("copied_call")].max()) + 1
            open_call = apply_call + _OPEN_AFTER
            return ("schedule", message[1] + 1, open_call, apply_call, None, _FORWARDS)

        def close(self):
            pass

    src = torch.empty(_EXPERT_BYTES, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)

    class Transport:
        def copy(self, experts, stream, start, end):
            with torch.cuda.stream(stream):
                start.record()
                for _ in experts:
                    dst.copy_(src, non_blocking=True)
                end.record()

    def gather(send, rows, leave):  # every rank decides alike: its own row stands for all
        for row in rows:
            row.copy_(send)

    tables = torch.zeros(_LAYERS, table, dtype=torch.int32, device=device)
    runtime = rt.AdaptiveRuntime(
        [Layer(i) for i in range(_LAYERS)],
        Transport(),
        torch.cuda.Stream(device),
        Decision(),
        gather,
        device,
        tables,
        buffers,
    )
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    boundary_us = []
    while runtime.window < _WINDOWS and runtime.call < 40 * _WINDOWS:
        start = time.perf_counter_ns()
        runtime.boundary()
        boundary_us.append((time.perf_counter_ns() - start) / 1e3)
        for _ in range(8):  # the stand-in forward, then the step's wait for its results
            a = (a @ a).clamp_(-1, 1)
        torch.cuda.current_stream().synchronize()
    runtime.worker.jobs.join()
    runtime.close()
    assert runtime.failure is None, runtime.failure
    steps = {}
    for _, _, step_marks in marks:
        for i in range(1, len(step_marks)):
            spent = (step_marks[i][1] - step_marks[i - 1][1]) / 1e3
            steps.setdefault(step_marks[i][0], []).append(spent)
    summary = dict(
        rank=rank,
        windows=runtime.window,
        calls=runtime.call + 1,
        publishes=len(marks),
        layers=[m[0] for m in marks],
        new_event_pairs=sum(1 for m in marks if m[1]),
        steps={k: dict(median=statistics.median(v), max=max(v)) for k, v in steps.items()},
        total_median=statistics.median(sum(steps[k][i] for k in steps) for i in range(len(marks))),
        boundary_us=dict(
            mean=statistics.mean(boundary_us),
            p99=sorted(boundary_us)[int(0.99 * len(boundary_us))],
            max=max(boundary_us),
        ),
    )
    print("APPLY_MARKS_SUMMARY " + json.dumps(summary), flush=True)
    return summary


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
def test_publish_marks_on_four_ranks():
    from tensorrt_llm._utils import mpi_world_size

    if mpi_world_size() > 1:
        pytest.skip("Spawns its own worker processes")
    with MPIPoolExecutor(
        max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        futures = [executor.submit(_rank_run) for _ in range(_WORLD_SIZE)]
        summaries = [future.result() for future in futures]
    for summary in summaries:
        print("APPLY_MARKS_SUMMARY " + json.dumps(summary), flush=True)
        # every install and retire window publishes (unchanged windows never reach the main thread)
        assert summary["publishes"] >= 2 * (_WINDOWS // 3) - 1
        names = {"host_tables", "event_pair", "begin_record", "table_copy", "end_record", "event"}
        assert set(summary["steps"]) == names
