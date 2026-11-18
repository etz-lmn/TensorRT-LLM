# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the main thread's share of adaptive EPLB windows.

The main thread runs one boundary per executor step and must stay within a small per-step budget,
so an update that changes nothing never reaches it, and a boundary hands jobs to the worker thread
only after its own work (a woken worker competes with it for the interpreter).
"""

import threading

import numpy as np
import pytest
from _torch.moe import test_adaptive_eplb_runtime as base
from _torch.moe.test_adaptive_eplb_runtime import (
    FORWARDS,
    TABLE,
    FakeDecision,
    R,
    make_ranks,
    run,
    script,
)

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt

host_cuda = base.host_cuda  # the runtime tests' fixture: CUDA events and streams on the host
pytestmark = [pytest.mark.cpu_only, pytest.mark.usefixtures("host_cuda")]


def empty(window):
    update = rt.LayoutUpdate(window)
    update.digest = update.fingerprint()
    return update


def publish_every_other(window):
    """Even windows change nothing; window 1 publishes layer 0, window 3 layers 0 and 1, ..."""
    if window % 2 == 0:
        return empty(window)
    layers = (0,) if window % 4 == 1 else (0, 1)
    tables = {layer: np.full(TABLE, window, np.int32) for layer in layers}
    update = rt.LayoutUpdate(window, tables=tables)
    update.digest = update.fingerprint()
    return update


def main_thread_log(runtime, monkeypatch, log):
    """Record the main thread's copies-done, publishes and hand-offs to the worker thread."""
    for name in ("_copies_done", "_apply_tables"):
        method = getattr(runtime, name)

        def wrapped(*args, _method=method, _name=name, **kwargs):
            if threading.current_thread() is threading.main_thread():
                log.append(_name)
            return _method(*args, **kwargs)

        monkeypatch.setattr(runtime, name, wrapped)
    put = runtime.worker.jobs.put

    def counted_put(job, *args, **kwargs):
        if threading.current_thread() is threading.main_thread() and job is not None:
            log.append("hand-off")
        return put(job, *args, **kwargs)

    monkeypatch.setattr(runtime.worker.jobs, "put", counted_put)


def run_windows(ranks, clock, windows):
    while not all(runtime.window >= windows for runtime in ranks):
        assert clock[0] < 40 * windows, [runtime.window for runtime in ranks]
        run(ranks, clock, 1)


def test_an_update_that_changes_nothing_never_reaches_the_main_threads_apply(monkeypatch):
    """Empty updates are settled on the worker thread, and the windows keep their schedule."""
    clock = [0]
    decisions = [FakeDecision(rank, empty) for rank in range(R)]
    ranks = make_ranks(decisions, clock)
    logs = [[] for _ in ranks]
    for runtime, log in zip(ranks, logs):
        main_thread_log(runtime, monkeypatch, log)
    run_windows(ranks, clock, 3)
    for runtime, decision, log in zip(ranks, decisions, logs):
        assert "_copies_done" not in log and "_apply_tables" not in log
        copied = [m[1] for m in decision.sent if m[0] == "copied"]
        assert copied[:3] == [0, 1, 2]  # the decision process still schedules every window
        assert runtime.installs == 0 and not runtime.tables.any() and runtime.failure is None


def test_a_boundary_hands_jobs_to_the_worker_only_after_its_own_work(monkeypatch):
    """Copies done and a publish in one boundary: the decision process's job goes out after it."""
    runtime = make_ranks([FakeDecision(rank) for rank in range(R)], [0])[0]
    runtime._event = lambda event, **data: None  # a bare publish: no decision recorded
    order = []
    publish = runtime._apply_tables

    def recorded_publish(*args):
        order.append("publish")
        return publish(*args)

    monkeypatch.setattr(runtime, "_apply_tables", recorded_publish)
    # Recorded, not run: the scripted rows below were never decided by this rank's decision process.
    monkeypatch.setattr(runtime.worker.jobs, "put", lambda job, *a, **k: order.append("hand-off"))
    update = script(0)(0)
    rows = np.array([[8, 1, 0.25, update.digest]] * R)  # every rank copied at call 8: apply at 9
    runtime.worker.inbox.append(("copied", update, rows))
    runtime.call, runtime.open_at = 8, None
    runtime.boundary()  # call 9
    assert runtime.failure is None
    assert order == ["publish", "hand-off"]


def test_the_main_threads_work_per_window_does_not_depend_on_the_update(monkeypatch):
    """Per window the main thread hands off its gather and its install; a publishing window adds
    one copies-done hand-off, one copies-done and one publish, whatever the number of layers."""
    clock = [0]
    decisions = [FakeDecision(rank, publish_every_other) for rank in range(R)]
    ranks = make_ranks(decisions, clock)
    logs = [[] for _ in ranks]
    for runtime, log in zip(ranks, logs):
        main_thread_log(runtime, monkeypatch, log)
    run_windows(ranks, clock, 6)
    for runtime, decision, log in zip(ranks, decisions, logs):
        settled = sorted({m[1] for m in decision.sent if m[0] == "copied"})
        publishing = [w for w in settled if w % 2]
        assert len(settled) >= 5 and len(publishing) >= 2
        assert log.count("_copies_done") == log.count("_apply_tables") == len(publishing)
        assert log.count("hand-off") == runtime.closed_windows + len(settled) + len(publishing)
        assert runtime.installed_layers == sum(1 if w % 4 == 1 else 2 for w in publishing)
        assert runtime.gathers_posted == 2 * len(settled) + (runtime.closed_windows - len(settled))
        assert runtime.window_forwards == FORWARDS
