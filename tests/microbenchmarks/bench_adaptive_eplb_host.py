# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main-thread cost of adaptive EPLB per executor step on the host, and its worker thread's CPU.

Measures the idle ``boundary`` and prints a ``cycle:`` line: the mean per step, entry to entry,
with windows opening, closing, deciding and installing inside, and the request progress of real
``LlmRequest`` objects read while a window closes::

    python3 tests/microbenchmarks/bench_adaptive_eplb_host.py --requests 256 --calls 20000
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import statistics_buffers
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import ExpertGeometry, TimeModel
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, generation_progress
from tensorrt_llm.sampling_params import SamplingParams

E, R, LOCAL, L = 384, 32, 16, 61


def requests(n: int) -> list[LlmRequest]:
    config = tensorrt_llm.bindings.SamplingConfig(SamplingParams()._get_sampling_config())
    made = [
        LlmRequest(
            request_id=i,
            max_new_tokens=1024,
            input_tokens=[1] * 128,
            sampling_config=config,
            is_streaming=False,
        )
        for i in range(n)
    ]
    for i, request in enumerate(made):
        request.py_decoding_iter = 10 + i % 7
    return made


def quantiles(samples_us: list[float]) -> str:
    q = statistics.quantiles(samples_us, n=100)
    return f"median {q[49]:.1f} us, p90 {q[89]:.1f} us, p99 {q[98]:.1f} us, max {max(samples_us):.1f} us"


def bench_cycle(n: int, calls: int, step_s: float) -> None:
    """``calls`` decode steps of the runtime with windows opening, closing, deciding and installing.

    A real decision process decides; every rank's statistics are this one's (the gathers copy them),
    copies complete at once, and every window reads rows that move the hot experts around so that
    layouts change. The steps are paced ``step_s`` apart (the forward's time, outside the
    measurement) so that decisions land within a few windows.
    """
    geometry = ExpertGeometry(E, R * LOCAL, R)
    homes = np.tile(geometry.home_layout(), (L, 1))
    model = TimeModel(4.6, 2.06, 0.0086, 1.02, 0.9, 256)
    decision = rt.DecisionProcess(rt.DecisionConfig(geometry, model, homes, 0, 4))
    buffers = statistics_buffers(L, E, "cpu")
    layers = [SimpleNamespace(expert_count=E, num_slots=R * LOCAL) for _ in range(L)]
    transport = SimpleNamespace(copy=lambda experts, stream, start, end: None)
    event = SimpleNamespace(
        record=lambda *a: None,
        query=lambda: True,
        synchronize=lambda: None,
        elapsed_time=lambda e: 0.0,
    )
    counts = dict(opened=0, closed=0, published=0)
    rows = torch.where(torch.arange(E) < 8, 400.0, 1.0).expand(L, E).clone()

    def gather(send, out) -> None:
        for row in out:
            row.copy_(send)

    fakes = dict(
        Event=lambda **kw: event,
        Stream=lambda *a, **kw: SimpleNamespace(),
        stream=lambda s: contextlib.nullcontext(),
    )
    with contextlib.ExitStack() as stack:
        for name, fake in fakes.items():
            stack.enter_context(mock.patch.object(torch.cuda, name, fake))
        tables = torch.zeros(L, 2 * (E + R * LOCAL), dtype=torch.int32)
        runtime = rt.AdaptiveRuntime(
            layers, transport, None, decision, gather, "cpu", tables, buffers
        )
        running = requests(n)
        open_window = runtime._open_window
        close_window = runtime._close_window

        def open_with_rows() -> None:  # the kernel's part: a window's rows, the hot experts moved
            open_window()
            counts["opened"] += 1
            forwards = runtime.window_forwards  # the window being opened
            buffers["rows"].copy_(rows.roll(counts["opened"] * 8, dims=1) * forwards)
            buffers["forwards"].fill_(forwards)

        def close_and_count() -> None:
            counts["closed"] += 1
            close_window()

        def listen(event: str, **data) -> None:
            if event == "applied":
                counts["published"] += 1

        runtime._open_window = open_with_rows
        runtime._close_window = close_and_count
        runtime._event = listen
        try:
            runtime.open_at = None  # first, calls with no window: the idle boundary
            samples = []
            for _ in range(min(calls, 2000)):
                t0 = time.perf_counter()
                runtime.boundary()
                samples.append((time.perf_counter() - t0) * 1e6)
            print(f"boundary idle, {L} layers: {quantiles(samples)}")
            runtime.open_at = runtime.call + 1
            entries = []
            pauses = 0.0
            for _ in range(calls):
                entries.append(time.perf_counter())
                runtime.boundary()
                if runtime.wants_request_progress:  # the executor's part of a closing window
                    runtime.add_request_progress(
                        len(running), lambda start, stop: generation_progress(running, start, stop)
                    )
                t = time.perf_counter()  # the rest of the step: paced, so that decisions land
                time.sleep(step_s)
                pauses += time.perf_counter() - t
            entries.append(time.perf_counter())
            main = (entries[-1] - entries[0] - pauses) / calls * 1e6
            deadline = time.monotonic() + 10  # let the last window's install land, for the count
            while runtime.worker.jobs.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.001)
                runtime.boundary()
            worker_cpu_ns = time.clock_gettime_ns(time.pthread_getcpuclockid(runtime.worker.ident))
        finally:
            decision.close()
    print(
        f"cycle: {calls} steps, windows opened {counts['opened']}, closed {counts['closed']}, "
        f"published {counts['published']}; main thread {main:.1f} us per step "
        f"(mean, entry to entry), background {worker_cpu_ns / 1e3 / calls:.1f} us cpu per step"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--calls", type=int, default=20000)
    parser.add_argument(
        "--step-ms", type=float, default=1.0, help="pace between steps (not measured)"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(1)  # the main thread's copies: on the GPU they are memcpys, not OMP work
    bench_cycle(args.requests, args.calls, args.step_ms / 1e3)


if __name__ == "__main__":
    main()
