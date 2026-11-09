# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime of adaptive EPLB: statistics windows, the decision process and installs.

The main thread advances the runtime once per executor step and never waits; a worker thread runs
the blocking work (the all-gathers, the decision process's messages, the expert copies). Every rank
decides alike in its own decision process. An install copies new replicas into free slots, and its
route tables take effect from the call after every rank's copies finished; ranks whose decisions
differ stop the run before that. The decision process imports this module with only torch, numpy
and this package available, so import other ``tensorrt_llm`` modules inside functions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import mmap
import os
import queue
import socket
import subprocess
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from .policy import (
    EvaluationScheduler,
    ExpertGeometry,
    TimeModel,
    active_slots,
    agreed_timings,
    choose_layers_to_install,
    copy_ms_per_layer,
    evaluate_gains,
    plan_layout,
)

if TYPE_CHECKING:
    from .balancer import AdaptiveMoeLayerLoadBalancer
    from .transport import PeerSlotTransport

    Gather = Callable[[torch.Tensor, list[torch.Tensor]], None]
    GenerationProgress = Callable[[int, int], tuple[int, int, int]]

_UINT32_MAX = (1 << 32) - 1


def solve_shares(load_rows: np.ndarray, layout: np.ndarray, geometry: ExpertGeometry) -> np.ndarray:
    """``[num_slots]`` shares of ``layout`` that minimize the busiest GPU's ``load_rows``."""
    load = np.ascontiguousarray(load_rows, dtype=np.float64)
    slots = np.asarray(layout)
    if not np.issubdtype(slots.dtype, np.integer):
        raise ValueError("layout must hold integers")
    slots = torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int64))
    solve = torch.ops.trtllm.adaptive_eplb_solve_shares
    return solve(torch.from_numpy(load), slots, geometry.ep_size).numpy()


class RouteTables(NamedTuple):
    """Route tables of one layout: copies (``counts``) and first table entry (``starts``) of every
    expert, the slot of every entry, and the Q32 cumulative shares ``cumulative_shares`` of replicated experts."""

    counts: list[int]
    starts: list[int]
    slots: list[int]
    cumulative_shares: list[int]

    def packed(self) -> list[int]:
        """``counts | starts | slots | cumulative_shares`` as int32 values."""
        cumulative_shares = [
            value - (1 << 32) if value >> 31 else value for value in self.cumulative_shares
        ]
        return self.counts + self.starts + self.slots + cumulative_shares


def build_route_tables(
    layout: Sequence[int], shares: Sequence[float] | None, num_experts: int, num_slots: int
) -> RouteTables:
    """Route tables of a ``[num_slots]`` layout and the ``shares`` of its slots; without
    ``shares``, every expert's rows split evenly over its copies."""
    layout = [int(e) for e in (layout.tolist() if hasattr(layout, "tolist") else layout)]
    shares = [1.0] * num_slots if shares is None else shares
    shares = [float(s) for s in (shares.tolist() if hasattr(shares, "tolist") else shares)]
    expert_slots: list[list[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(layout):
        if expert >= 0:
            expert_slots[expert].append(slot)
    counts = [len(copies) for copies in expert_slots]
    if 0 in counts:
        raise ValueError(f"expert {counts.index(0)} has no copy")
    starts = list(itertools.accumulate(counts, initial=0))[:-1]
    slots = [slot for copies in expert_slots for slot in copies] + [-1] * layout.count(-1)
    cumulative_shares = [_UINT32_MAX] * num_slots
    for copies, start in zip(expert_slots, starts):
        weights = [max(0.0, shares[slot]) for slot in copies]
        total = sum(weights)
        if total <= 0.0:
            weights, total = [1.0] * len(copies), float(len(copies))
        acc = 0.0
        bound = 0
        for k in range(len(copies) - 1):
            acc += weights[k]
            bound = min(max(int(acc / total * 4294967296.0 + 0.5), bound), _UINT32_MAX)
            cumulative_shares[start + k] = bound
    return RouteTables(counts, starts, slots, cumulative_shares)


_DECISION_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_main.py")
ROWS_PER_EXPERT = 512
"""int: Rows a statistics window collects per expert, so that an expert's rows per forward carry a
relative standard error of about 1 / sqrt(512) = 4.4 % at any batch size or draft length: a window
lasts as many forwards as the previous window's rows per forward need to reach it. The first window
lasts one forward; with fewer rows than that it only measures the rows per forward."""
REQUESTS_PER_CALL = 64
"""int: Generation requests whose progress one call reads (about 20 us of main-thread time)."""
FIELDS = (
    "remaining",
    "requests",
    "accepted",
    "stats_copy_ms",
    "applied",
    "applied_layers",
    "applied_ms",
)
"""tuple: The values at the end of a rank's window statistics, the float32 array the ranks
all-gather, which holds ``rows [L, E] | forwards [L] | applied window [L] | FIELDS``.

``remaining``, ``requests`` and ``accepted`` sum the tokens the generation requests still have to
produce, the requests, and the tokens they accept per step. ``stats_copy_ms`` is the GPU time of
copying the statistics; ``applied``, ``applied_layers`` and ``applied_ms`` are the window, layer
count and GPU time of the last route tables applied.
"""
ROW = ("copied_call", "moved", "copy_ms", "digest")
"""tuple: The float64 values a rank all-gathers after its copies of an update ran: the call they
finished in, the layers it moved, their copy time and the update's digest."""


def window_forwards(rows_per_forward: float, experts: int) -> int:
    """Forwards a window needs to collect :data:`ROWS_PER_EXPERT` rows per expert."""
    return max(1, math.ceil(ROWS_PER_EXPERT * experts / rows_per_forward))


def window_stats_size(layers: int, experts: int) -> int:
    return layers * (experts + 2) + len(FIELDS)


@dataclass(frozen=True)
class DecisionConfig:
    """Inputs of the decision process; ``home_layouts`` is ``[L, num_slots]`` int64."""

    geometry: ExpertGeometry
    model: TimeModel
    home_layouts: np.ndarray
    ep_rank: int
    ranks_per_node: int
    stats_fd: int = -1


@dataclass
class LayoutUpdate:
    """The decision of one window.

    It holds the new route tables of the chosen layers, the layers whose layout changes
    (``moved``, copied as one timed batch) and the expert copies this rank makes for them
    (``copy_items``). ``digest`` lets the ranks check that they all decided the same.
    """

    window: int
    copy_items: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    tables: dict[int, np.ndarray] = field(default_factory=dict)
    moved: list[int] = field(default_factory=list)
    digest: int = 0

    def fingerprint(self) -> int:
        """48-bit digest (exact in a float64) of the tables and moved layers every rank shares."""
        digest = hashlib.blake2b(digest_size=6)
        for layer in sorted(self.tables):
            digest.update(np.int64(layer).tobytes() + self.tables[layer].tobytes())
        digest.update(np.asarray(self.moved, dtype=np.int64).tobytes())
        return int.from_bytes(digest.digest(), "little")


def copy_items(
    old: np.ndarray, new: np.ndarray, geometry: ExpertGeometry, rank: int, ranks_per_node: int
) -> list[tuple[int, int, int]]:
    """``(dst_local_slot, src_rank, src_local_slot)`` of every expert ``rank`` copies from ``old``
    to ``new``, from its own node first. Raises ``ValueError`` when a destination is not free."""
    local = geometry.local_slots
    node = rank // ranks_per_node
    items = []
    for slot in range(rank * local, (rank + 1) * local):
        expert = int(new[slot])
        if expert < 0 or old[slot] == expert:
            continue
        if old[slot] >= 0:
            raise ValueError(f"slot {slot} holds expert {old[slot]}, not free for expert {expert}")
        holders = np.flatnonzero(old == expert)
        src = int(min(holders, key=lambda s: (s // local // ranks_per_node != node, s)))
        items.append((slot - rank * local, src // local, src % local))
    return items


def _timings(by_group: dict[int, tuple[int, float]]) -> list[tuple[int, int, float]]:
    return [(group, layers, ms) for group, (layers, ms) in sorted(by_group.items())]


class _Decider:
    """State of the decision process: the committed layouts and every timing of the run."""

    def __init__(self, config: DecisionConfig, stats: np.ndarray) -> None:
        self.config = config
        self.stats = stats
        self.layout = np.array(config.home_layouts, dtype=np.int64)
        self.shares = (self.layout >= 0).astype(np.float64)
        self.committed = np.full(len(self.layout), -1)
        self.copy_timings: dict[int, tuple[int, float]] = {}
        self.install_timings: dict[int, tuple[int, float]] = {}
        self.stats_copies: list[float] = []
        self.evaluations = EvaluationScheduler(config.model, config.geometry)
        self.active = active_slots(config.geometry)
        self.forwards = 1
        self.next_forwards: int | None = None
        self.open_call = 0
        self.awaiting_copy: dict[int, tuple] = {}

    def handle(self, message: tuple) -> tuple:
        if message[0] == "window":
            return ("update", self.decide(message[1]))
        return self.schedule(*message[1:])

    def schedule(self, window: int, rows: np.ndarray) -> tuple:
        """The next window's open call once every rank's copies of ``window`` ran."""
        col = {name: rows[:, i] for i, name in enumerate(ROW)}
        apply_call = int(col["copied_call"].max()) + 1
        per_rank = [[(window, int(n), float(ms))] for n, ms in zip(col["moved"], col["copy_ms"])]
        self.copy_timings.update((g, (n, ms)) for g, n, ms in agreed_timings(per_rank) if n > 0)
        close, growth, steps_left = self.awaiting_copy.pop(window)
        self.forwards = self.next_forwards or self.forwards
        plan = self.evaluations.plan(
            apply_call,
            close,
            self.forwards,
            self.stats_copies,
            _timings(self.install_timings),
            growth,
            steps_left,
        )
        self.open_call = apply_call + plan.open_after
        return ("schedule", window + 1, self.open_call, apply_call, plan, self.forwards)

    def decide(self, window: int) -> LayoutUpdate:
        cfg = self.config
        stats = self.stats
        geometry = cfg.geometry
        layers, slots = self.layout.shape
        experts = geometry.num_experts
        close = self.open_call + self.forwards
        tail = {name: stats[:, layers * (experts + 2) + i] for i, name in enumerate(FIELDS)}
        applied = [
            [(int(g), int(n), float(ms))] if g >= 0 else []
            for g, n, ms in zip(tail["applied"], tail["applied_layers"], tail["applied_ms"])
        ]
        self.install_timings.update((g, (n, ms)) for g, n, ms in agreed_timings(applied))
        if (tail["stats_copy_ms"] > 0).all():
            self.stats_copies.append(float(tail["stats_copy_ms"].max()))
        update = LayoutUpdate(window)
        forwards = stats[:, layers * experts : layers * (experts + 1)].astype(np.float64)
        if forwards.min() < 1:
            # A rank counted no forward of some layer: this window gives no rows.
            self.awaiting_copy[window] = (close, None, None)
            return update
        # Rows per forward of every expert: every rank's rows over its own forwards, summed.
        routed = stats[:, : layers * experts].astype(np.float64).reshape(len(stats), layers, -1)
        rows = (routed / forwards[:, :, None]).sum(0)
        per_forward = float(rows.sum(1).mean())
        counted = float(routed.sum((0, 2)).mean())
        sizing = self.next_forwards is None and counted < ROWS_PER_EXPERT * experts
        if per_forward > 0.0:
            self.next_forwards = window_forwards(per_forward, experts)
        if sizing:
            self.awaiting_copy[window] = (close, None, None)
            return update
        total = {name: float(values.astype(np.float64).sum()) for name, values in tail.items()}
        steps_left = None
        if min(total["accepted"], total["requests"]) > 0:
            accepted_per_step = total["accepted"] / total["requests"]
            steps_left = total["remaining"] / accepted_per_step / total["requests"]
        applied_window = stats[:, layers * (experts + 1) : layers * (experts + 2)].min(0)
        planned = np.flatnonzero(self.committed <= applied_window)
        new_layout = np.empty((len(planned), slots), dtype=np.int64)
        new_shares = np.empty((len(planned), slots))
        for i, layer in enumerate(planned.tolist()):
            homes = cfg.home_layouts[layer]
            current = self.layout[layer]
            new_layout[i] = plan_layout(rows[layer], geometry, homes, current, self.active).layout
            new_shares[i] = solve_shares(rows[layer], new_layout[i], geometry)
        current = (rows[planned], self.layout[planned], self.shares[planned])
        gains = evaluate_gains(*current, new_layout, new_shares, cfg.model, geometry)
        previous_load = self.evaluations.previous_load(planned.tolist(), current[0])
        if previous_load is not None:
            # A candidate must also gain on its layer's previous window.
            previous_gains = evaluate_gains(
                previous_load, *current[1:], new_layout, new_shares, cfg.model, geometry
            )
            gains = np.minimum(gains, previous_gains)
        gain = dict(zip(planned.tolist(), gains.tolist()))
        copy_ms = copy_ms_per_layer(_timings(self.copy_timings))
        chosen = choose_layers_to_install(gain, steps_left, copy_ms or 0.0)
        if copy_ms is None:
            # No copy has been timed yet: install the best layer alone to time one.
            chosen = chosen[:1]
        growth = self.evaluations.read_cost(
            range(layers), rows, self.layout, self.shares, self.open_call
        )
        self.awaiting_copy[window] = (close, growth, steps_left)
        moved = []
        for i, layer in enumerate(planned.tolist()):
            if layer not in chosen:
                continue
            tables = build_route_tables(new_layout[i], new_shares[i], experts, slots).packed()
            update.tables[layer] = np.array(tables, dtype=np.int32)
            fresh = (new_layout[i] >= 0) & (new_layout[i] != self.layout[layer])
            if fresh.any():
                moved.append(layer)
                update.copy_items[layer] = copy_items(
                    self.layout[layer], new_layout[i], geometry, cfg.ep_rank, cfg.ranks_per_node
                )
            self.layout[layer] = new_layout[i]
            self.shares[layer] = new_shares[i]
            self.committed[layer] = window
        update.moved = sorted(moved)
        update.digest = update.fingerprint()
        return update


def serve(conn: Connection) -> None:
    """Answer every message on ``conn`` until it closes; a failure is sent back and ends it."""
    config = conn.recv()
    stats = np.frombuffer(mmap.mmap(config.stats_fd, 0), np.float32)
    decider = _Decider(config, stats.reshape(config.geometry.ep_size, -1))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        try:
            conn.send(decider.handle(message))
        except Exception:  # the main thread raises it
            conn.send(("error", traceback.format_exc()))
            return


class DecisionProcess:
    """The decision process of one rank: a fresh interpreter without MPI or a GPU.

    It shares ``stats`` (``[ep_size, window_stats_size]`` float32, which the all-gather fills) with
    this process and runs ``decision_main.py``, which imports numpy, torch and this module but not
    ``tensorrt_llm``, and loads the torch op libraries this process loaded.
    """

    def __init__(self, config: DecisionConfig) -> None:
        size = window_stats_size(len(config.home_layouts), config.geometry.num_experts)
        nbytes = config.geometry.ep_size * size * 4
        fd = os.memfd_create("adaptive-eplb-windows")
        os.ftruncate(fd, nbytes)
        self._map = mmap.mmap(fd, nbytes)
        self.stats = torch.frombuffer(self._map, dtype=torch.float32).view(-1, size)
        parent, child = socket.socketpair()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PMIX", "PMI_", "OMPI_"))}
        env.update(TLLM_DISABLE_MPI="1", TRT_LLM_NO_LIB_INIT="1", CUDA_VISIBLE_DEVICES="")
        libraries = sorted(torch.ops.loaded_libraries)
        argv = [sys.executable, _DECISION_MAIN, str(child.fileno()), *libraries]
        self.process = subprocess.Popen(argv, pass_fds=(child.fileno(), fd), env=env)
        child.close()
        os.close(fd)
        self._conn = Connection(parent.detach())
        self._conn.send(dataclasses.replace(config, stats_fd=fd))

    def submit(self, message: tuple) -> None:
        self._conn.send(message)

    def recv(self) -> tuple:
        """Wait for the process's next answer; raise ``RuntimeError`` once it failed or exited."""
        try:
            message = self._conn.recv()
        except EOFError:
            raise RuntimeError(f"adaptive EPLB decision process exited ({self.process.poll()})")
        if message[0] == "error":
            raise RuntimeError(f"adaptive EPLB decision process failed:\n{message[1]}")
        return message

    def close(self) -> None:
        """Close the pipe and end the process, killing it if it does not exit in time."""
        self._conn.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class _WorkerThread(threading.Thread):
    """Runs blocking jobs in order; results go to ``inbox``, a failure is raised by the main thread.

    ``jobs.join()`` waits until every posted job ran.
    """

    def __init__(self) -> None:
        super().__init__(name="adaptive-eplb-worker", daemon=True)
        self.jobs: queue.Queue = queue.Queue()
        self.inbox: deque[tuple] = deque()
        self.failure: BaseException | None = None
        self.start()

    def post(self, job: Callable[[], tuple | None]) -> None:
        self.jobs.put(job)

    def run(self) -> None:
        for job in iter(self.jobs.get, None):
            try:
                result = job()
            except BaseException as exc:  # raised on the main thread
                self.failure = exc
                return
            if result is not None:
                self.inbox.append(result)
            self.jobs.task_done()

    def close(self) -> None:
        self.jobs.put(None)
        self.join(timeout=30)


class AdaptiveRuntime:
    """Statistics windows, decisions and installs of all balanced layers of one rank.

    ``gather(send, rows)`` all-gathers ``send`` (the window statistics of this rank, or a float64
    :data:`ROW`) into ``rows``, one per EP rank; ``tables`` are the layers' ``[L, 2 * (experts + slots)]`` route
    tables and ``statistics`` their :func:`~.balancer.statistics_buffers`.

    One update is in flight at a time: its route tables apply before the next window opens, and the
    worker waits for the statistics copy of that window's close before it gathers it. The pinned
    host tables, the applied windows and the copies into free slots rely on that order.
    """

    def __init__(
        self,
        layers: Sequence[AdaptiveMoeLayerLoadBalancer],
        transport: PeerSlotTransport,
        copy_stream: torch.cuda.Stream,
        decision: DecisionProcess,
        gather: Gather,
        device: torch.device,
        tables: torch.Tensor,
        statistics: dict[str, torch.Tensor],
    ) -> None:
        # Imported here: the decision process imports this module without the rest of the package.
        from tensorrt_llm._utils import prefer_pinned

        self.layers = layers
        self.transport = transport
        self.copy_stream = copy_stream
        self.decision = decision
        self.gather = gather
        self.worker = _WorkerThread()
        self.call = -1
        self.window = 0
        self.open_at: int | None = 0
        self.close_at: int | None = None
        self.window_opened_call = -1
        self.window_forwards = 1
        self.applied_window = np.full(len(layers), -1.0)
        self.request_totals = np.zeros(3)  # the FIELDS remaining, requests and accepted
        self.wants_request_progress = False
        self._requests_read = 0
        self.last_applied = (-1.0, 0.0, 0.0)
        self.pending_updates: deque[tuple[int, LayoutUpdate]] = deque()
        self._apply_events: deque = deque()
        self.closed_windows = 0
        self.installs = 0
        self.installed_layers = 0
        # Timing event pairs of the publishes: one comes back here once the worker read its time
        # (creating two CUDA events costs the main thread more than the rest of a publish).
        self._apply_event_pairs: deque = deque()
        self._storage = statistics["storage"]
        pin = torch.device(device).type == "cuda" and prefer_pinned()
        self._own_stats = torch.zeros(decision.stats.shape[1], pin_memory=pin)
        # The window's statistics as copied, read on the worker thread through the same views.
        self._own_raw = torch.zeros_like(self._storage, device="cpu", pin_memory=pin)
        rows, forwards = statistics["rows"], statistics["forwards"]
        rows_bytes = rows.numel() * rows.element_size()
        self._raw_rows = self._own_raw[:rows_bytes].view(rows.dtype).numpy()
        at = forwards.storage_offset() * forwards.element_size() - self._storage.storage_offset()
        self._raw_forwards = self._own_raw[at:].view(forwards.dtype).numpy()
        self._gathered = list(decision.stats.unbind(0))
        self.tables = tables
        self._host_tables = torch.empty_like(tables, device="cpu", pin_memory=pin)
        self._host_tables.copy_(tables)
        self._host_tables_np = self._host_tables.numpy()  # the host mirror, written row by row
        self._stats_copy_begin = torch.cuda.Event(enable_timing=True)
        self._stats_copy_end = torch.cuda.Event(enable_timing=True)
        self._pending_stats_copy: torch.cuda.Event | None = None
        self._copy_begin = torch.cuda.Event(enable_timing=True, blocking=True)
        self._copy_end = torch.cuda.Event(enable_timing=True, blocking=True)

    def close(self) -> None:
        """End the worker thread and the decision process."""
        self.worker.close()
        self.decision.close()

    def counters(self) -> dict[str, int]:
        """The windows closed, the installs applied (updates with new route tables) and the layers
        they installed, since the runtime started."""
        return dict(
            windows=self.closed_windows,
            installs=self.installs,
            installedLayers=self.installed_layers,
        )

    def add_request_progress(self, num_requests: int, progress: GenerationProgress) -> None:
        """Add the progress of up to :data:`REQUESTS_PER_CALL` more of the ``num_requests``
        generation requests to the window's request totals: ``progress(start, stop)`` gives the
        tokens the requests ``[start, stop)`` still have to produce, how many of them are requests
        and the tokens they accept per step."""
        start = self._requests_read
        stop = start + REQUESTS_PER_CALL
        self.request_totals += progress(start, stop)
        self._requests_read = stop
        self.wants_request_progress = stop < num_requests

    def boundary(self) -> None:
        """Advance windows, decisions, copies and route tables by one call; never waits."""
        self.call += 1
        call = self.call
        worker = self.worker
        if worker.failure is not None:
            raise RuntimeError("adaptive EPLB worker thread failed") from worker.failure
        # The messages first: a window scheduled to open at this call opens at it.
        while worker.inbox:
            message = worker.inbox.popleft()
            if message[0] == "schedule":
                _, self.window, self.open_at, apply_call, plan, self.window_forwards = message
                self._event(
                    "scheduled",
                    window=self.window,
                    apply_call=apply_call,
                    next_open=self.open_at,
                    plan=plan,
                    forwards=self.window_forwards,
                )
            elif message[0] == "copied":
                self._copies_done(*message[1:])
            else:
                update = message[1]
                self._event(
                    "decision",
                    window=update.window,
                    call=call,
                    moved=update.moved,
                    tables=update.tables,
                    copy_items=update.copy_items,
                )
                worker.post(lambda update=update: self._install(update))
        if self.pending_updates and self.pending_updates[0][0] <= call:
            self._apply_tables(call, self.pending_updates.popleft()[1])
        if self.close_at is None and self.open_at is not None and call >= self.open_at:
            assert not self.pending_updates, "an update must apply before the next window opens"
            self._open_window()
            self.open_at = None
            self.close_at = call + self.window_forwards
            self.window_opened_call = call
        elif call == self.close_at:
            self._close_window()
        if self._pending_stats_copy is not None and not self.wants_request_progress:
            copied = self._pending_stats_copy
            self._pending_stats_copy = None
            totals = self.request_totals.copy()
            window = self.window
            worker.post(lambda: self._gather_window(copied, totals, window))

    def _event(self, event: str, **data: object) -> None:
        """Log an event of the windows, decisions and installs at debug level: its scalars and
        layer lists (tables, copies and statistics are not printed)."""
        from tensorrt_llm.logger import logger

        shown = ", ".join(
            f"{key}={value}"
            for key, value in data.items()
            if not isinstance(value, (dict, np.ndarray))
        )
        logger.debug(f"adaptive EPLB {event}: {shown}")

    def _open_window(self) -> None:
        """Zero the statistics."""
        self._storage.zero_()

    def _close_window(self) -> None:
        """Copy the statistics into this rank's window statistics, timing the copies."""
        assert not self.pending_updates, "an update must apply before its next window closes"
        self.closed_windows += 1
        self._stats_copy_begin.record()
        self._own_raw.copy_(self._storage, non_blocking=True)
        self._stats_copy_end.record()
        self._pending_stats_copy = self._stats_copy_end
        self.close_at = None
        self.wants_request_progress = True
        self._requests_read = 0
        self.request_totals[:] = 0

    def _gather_window(self, copied: torch.cuda.Event, totals: np.ndarray, window: int) -> tuple:
        """Worker thread: complete this rank's window statistics, all-gather them and get the
        decision."""
        copied.synchronize()
        while self._apply_events and self._apply_events[0][3].query():
            applied, layers, begin, end = self._apply_events.popleft()
            self.last_applied = (float(applied), float(layers), begin.elapsed_time(end))
            self._apply_event_pairs.append((begin, end))
        n = len(self.layers)
        experts = self.layers[0].expert_count
        own = self._own_stats.numpy()
        own[: n * experts] = self._raw_rows
        own[n * experts : n * (experts + 1)] = self._raw_forwards
        own[n * (experts + 1) : n * (experts + 2)] = self.applied_window
        stats_copy_ms = self._stats_copy_begin.elapsed_time(self._stats_copy_end)
        own[n * (experts + 2) :] = (*totals, stats_copy_ms, *self.last_applied)
        self.gather(self._own_stats, self._gathered)
        self._event(
            "window",
            window=window,
            open_call=self.window_opened_call,
            stats=self.decision.stats.numpy(),
        )
        self.decision.submit(("window", window))
        return self.decision.recv()

    def _copies_done(self, update: LayoutUpdate, rows: np.ndarray) -> None:
        """Every rank's copies of ``update`` ran: check that all ranks decided alike."""
        digests = rows[:, ROW.index("digest")]
        if (digests != update.digest).any():
            raise RuntimeError(
                f"adaptive EPLB ranks decided window {update.window} differently: layout digests "
                f"{[hex(int(d)) for d in digests]}"
            )
        apply_call = int(rows[:, ROW.index("copied_call")].max()) + 1
        assert not self.pending_updates, "one update is in flight at a time"
        self.pending_updates.append((apply_call, update))
        self._event("copied", window=update.window, apply_call=apply_call, rows=rows)
        window = update.window

        def schedule() -> tuple:
            self.decision.submit(("copied", window, rows))
            return self.decision.recv()

        self.worker.post(schedule)

    def _install(self, update: LayoutUpdate) -> tuple:
        """Worker thread: copy the update's experts as one batch, then all-gather the completion."""
        ms = 0.0
        if update.moved:
            experts = [
                (layer, *item) for layer in update.moved for item in update.copy_items[layer]
            ]
            self.transport.copy(experts, self.copy_stream, self._copy_begin, self._copy_end)
            self._copy_end.synchronize()
            ms = self._copy_begin.elapsed_time(self._copy_end)
        row = (self.call, len(update.moved), ms, update.digest)
        rows = torch.zeros(len(self._gathered), len(ROW), dtype=torch.float64)
        self.gather(torch.tensor(row, dtype=torch.float64), list(rows))
        return ("copied", update, rows.numpy())

    def _apply_tables(self, call: int, update: LayoutUpdate) -> None:
        """Install the update's route tables with one timed copy of the span of layers it changes
        (the other layers in it hold the same tables already; none for an update without tables);
        later forwards route by them."""
        # The previous copy from the pinned host tables finished before the window after it closed.
        assert not self._apply_events, "the previous route tables' copy is still in flight"
        host_tables = self._host_tables_np
        for layer, tables in update.tables.items():
            host_tables[layer] = tables
            self.applied_window[layer] = update.window
        if self._apply_event_pairs:
            begin, end = self._apply_event_pairs.popleft()
        else:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        begin.record()
        if update.tables:
            first, end_layer = min(update.tables), max(update.tables) + 1
            self.tables[first:end_layer].copy_(
                self._host_tables[first:end_layer], non_blocking=True
            )
        end.record()
        self._apply_events.append((update.window, len(update.tables), begin, end))
        if update.tables:
            self.installs += 1
            self.installed_layers += len(update.tables)
        self._event("applied", window=update.window, call=call, layers=sorted(update.tables))
