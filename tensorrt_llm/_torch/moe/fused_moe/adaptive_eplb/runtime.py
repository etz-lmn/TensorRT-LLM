# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonblocking windows; tables apply only after every rank copies into free slots."""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import math
import mmap
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

import numpy as np
import torch

from .policy import (
    EvaluationScheduler,
    ExpertGeometry,
    TimeModel,
    active_slots,
    agreed_timings,
    build_route_tables,
    choose_layers_to_install,
    evaluate_gains,
    plan_layout,
    snapshot_summary,
    solve_shares,
)

if TYPE_CHECKING:
    from .balancer import AdaptiveMoeLayerLoadBalancer
    from .transport import PeerSlotTransport

    Gather = Callable[[torch.Tensor, list[torch.Tensor], threading.Event], None]
    GenerationProgress = Callable[[int, int], tuple[int, int, int]]

_DECISION_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_main.py")
ROWS_PER_EXPERT = 512
"""Per-expert target: relative sampling error about 1/sqrt(512); the first window measures one forward."""
CLOSE_WAIT_S = 30.0
"""float: Seconds a close waits for the worker thread, again after telling it to leave."""
GATHER_SLICE_S = 1.0
"""float: Seconds of one wait in an all-gather between looks at whether to leave it."""
FIELDS = (
    "remaining",
    "requests",
    "accepted",
    "stats_copy_ms",
    "applied",
    "applied_layers",
    "applied_ms",
)
"""All-gather float32 tail after rows[L,E], forwards[L], applied-window[L]: remaining output tokens,
requests, accepted tokens/step, stats D2H GPU ms, and last applied window/layer count/GPU ms."""
ROW = ("copied_call", "moved", "copy_ms", "digest")
"""Float64 copy-completion all-gather: completed call, moved layers, copy ms, digest."""


def window_forwards(rows_per_forward: float, experts: int) -> int:
    """Forwards a window needs to collect :data:`ROWS_PER_EXPERT` rows per expert."""
    return max(1, math.ceil(ROWS_PER_EXPERT * experts / rows_per_forward))


def window_stats_size(layers: int, experts: int) -> int:
    return layers * (experts + 2) + len(FIELDS)


@dataclass(frozen=True)
class DecisionConfig:
    """Home layouts [L, slots] and startup copy ms/replica, retained among the timed samples."""

    geometry: ExpertGeometry
    model: TimeModel
    home_layouts: np.ndarray
    ep_rank: int
    ranks_per_node: int
    stats_fd: int = -1
    copy_ms: float | None = None


@dataclass
class LayoutUpdate:
    """One window's tables, moved layers, per-rank copies and shared agreement digest."""

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


class _Decider:
    """State of the decision process: the committed layouts and every timing of the run."""

    def __init__(self, config: DecisionConfig, stats: np.ndarray) -> None:
        self.measure_decisions = os.environ.get("GENEPLB_DECISION_SUMMARY") == "1"
        objective = os.environ.get("GENEPLB_EXPERIMENT_SHARE_OBJECTIVE")
        if objective not in (None, "rows", "time", "builtin-time") or (
            objective is not None and not self.measure_decisions
        ):
            raise ValueError(
                "Share-objective experiment requires explicit rows/time and decision summary"
            )
        identity = os.environ.get("GENEPLB_EXPERIMENT_BUILTIN_MODEL_IDENTITY")
        if identity is not None or objective == "builtin-time":
            fields = (
                "per_active_slot_us",
                "per_extra_tile_us",
                "per_row_us",
                "hot_slot_extra_tile_us",
                "scale",
                "route_us",
                "tile_rows",
            )
            actual = {key: getattr(config.model, key) for key in fields}
            if (
                objective not in ("rows", "builtin-time")
                or config.model.scales
                or config.model.time_shares
                or identity != json.dumps(actual, sort_keys=True, separators=(",", ":"))
            ):
                raise ValueError("Builtin share experiment model identity differs")
        if objective == "builtin-time":
            config = dataclasses.replace(
                config, model=dataclasses.replace(config.model, time_shares=True)
            )
        elif objective == "rows":
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(
                    config.model,
                    time_shares=False,
                    scales=tuple((*entry[:-1], False) for entry in config.model.scales),
                ),
            )
        elif objective == "time" and (
            not config.model.time_shares or any(not entry[-1] for entry in config.model.scales)
        ):
            raise ValueError("Time-objective experiment requires an already priced time model")
        self.config = config
        self.stats = stats
        self.layout = np.array(config.home_layouts, dtype=np.int64)
        self.shares = (self.layout >= 0).astype(np.float64)
        self.committed = np.full(len(self.layout), -1)
        self.install_timings: dict[int, tuple[int, float]] = {}
        self.stats_copies: list[float] = []
        self.evaluations = EvaluationScheduler(config.model, config.geometry)
        self.active = active_slots(config.geometry)
        # (apply call, ms per replica) of the timed copies; the start-up one, at call inf, always counts.
        self.copy_samples = [] if config.copy_ms is None else [(math.inf, config.copy_ms)]
        self.forwards = 1
        self.next_forwards: int | None = None
        self.open_call = 0
        self.awaiting_copy: dict[int, tuple] = {}
        # (window, rows per forward [L, E], the requests they stand for) of the last two windows.
        self.kept: deque[tuple[int, np.ndarray, float]] = deque(maxlen=2)
        self.close_requests = self.largest_change = 0.0

    def handle(self, message: tuple) -> tuple:
        if message[0] == "window":
            return ("update", self.decide(message[1]))
        return self.schedule(*message[1:])

    def schedule(self, window: int, rows: np.ndarray) -> tuple:
        """The next window's open call once every rank's copies of ``window`` ran."""
        col = {name: rows[:, i] for i, name in enumerate(ROW)}
        apply_call = int(col["copied_call"].max()) + 1
        per_rank = [[(window, int(n), float(ms))] for n, ms in zip(col["moved"], col["copy_ms"])]
        # got: the replicas the update's busiest receiving GPU copied
        close, growth, steps_left, got = self.awaiting_copy.pop(window)
        self.copy_samples += [(apply_call, ms / got) for _, n, ms in agreed_timings(per_rank) if n]
        self.forwards = self.next_forwards or self.forwards
        summary = {} if self.measure_decisions else None
        snapshot_summary(summary, copy_feedback=rows, max_receiver_copies=got)
        plan = self.evaluations.plan(
            apply_call,
            close,
            self.forwards,
            self.stats_copies,
            [(group, *timing) for group, timing in sorted(self.install_timings.items())],
            growth,
            steps_left,
            summary=summary,
        )
        self.open_call = apply_call + plan.open_after
        message = ("schedule", window + 1, self.open_call, apply_call, plan, self.forwards)
        return (*message, summary) if summary is not None else message

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
        summary = (
            {"status": "invalid_forward", "share_solve": None} if self.measure_decisions else None
        )
        if summary is not None:
            update.summary = {"window": window, "diagnostics": summary}
        self.awaiting_copy[window] = (close, None, None, 0)  # until the window gives a decision
        forwards = stats[:, layers * experts : layers * (experts + 1)].astype(np.float64)
        if forwards.min() < 1:  # a rank counted no forward of some layer: this window gives no rows
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
            if summary is not None:
                summary["status"] = "initial_sizing"
            return update
        total = {name: float(values.astype(np.float64).sum()) for name, values in tail.items()}
        steps_left = None
        if min(total["accepted"], total["requests"]) > 0:
            per_step = total["accepted"] / total["requests"]  # tokens a request accepts per step
            steps_left = total["remaining"] / per_step / total["requests"]
        snapshot_summary(
            summary,
            steps_left=steps_left,
            planned_open_call=self.open_call,
            progress_totals={key: total[key] for key in ("remaining", "requests", "accepted")},
        )
        # A batch that changed between two closes more than it ever did (a steady batch never does, a
        # draining one keeps doing so) is priced where the new layouts serve it: the requests at this
        # close carried a window ahead at that rate, over their mean across the window.
        requests, before = total["requests"], self.close_requests
        change = abs(requests / before - 1.0) if min(requests, before) > 0.0 else 0.0
        self.close_requests, ahead = requests, None
        if change > self.largest_change:
            ahead = requests * requests / before
            rows = rows * (ahead / ((before + requests) / 2.0))
        self.largest_change = max(self.largest_change, change)
        requests = ahead or ((before + requests) / 2.0 if before > 0.0 else requests)
        applied_window = stats[:, layers * (experts + 1) : layers * (experts + 2)].min(0)
        planned = np.flatnonzero(self.committed <= applied_window)
        # The busiest rank's rows per forward size the CUDA graph bucket the steps pad to.
        query_rows = float((routed.sum(2) / forwards).mean(1).max())
        model = cfg.model.at(query_rows)
        if summary is not None:
            completed = {"rows": 0, "time": 0}
            names = (
                "per_active_slot_us",
                "per_extra_tile_us",
                "per_row_us",
                "hot_slot_extra_tile_us",
                "scale",
                "route_us",
                "tile_rows",
            )
            summary.update(
                status="evaluated",
                share_solve=dict(
                    query_rows=query_rows,
                    time_shares=bool(model.time_shares),
                    model={n: getattr(model, n) for n in names},
                    completed_calls=completed,
                ),
            )
        self.evaluations.model = model
        new_layout = np.empty((len(planned), slots), dtype=np.int64)
        new_shares, resolved = np.empty((2, len(planned), slots))
        home, installed = cfg.home_layouts[planned], self.layout[planned]
        # This window's rows, replaced below by an earlier window's rows where one is kept.
        previous_load = rows[planned]
        for i, layer in enumerate(planned.tolist()):
            plan = plan_layout(rows[layer], geometry, home[i], installed[i], self.active)
            new_layout[i] = plan.layout
            new_shares[i] = solve_shares(rows[layer], new_layout[i], geometry, model)
            if summary is not None:
                completed["time" if model.time_shares else "rows"] += 1
            resolved[i] = solve_shares(rows[layer], installed[i], geometry, model)
            if summary is not None:
                completed["time" if model.time_shares else "rows"] += 1
            # The latest kept window that neither the installed layout (planned on the window the
            # layer was committed in) nor the new one (planned on this window) was planned on, at
            # this window's batch; a layer without one keeps only this window's gains.
            seen = next((x for x in reversed(self.kept) if x[0] != self.committed[layer]), None)
            if seen is not None:
                previous_load[i] = seen[1][layer] * (ahead / seen[2] if ahead and seen[2] else 1.0)
        # Shares solved again on the installed layouts replace the old ones where they gain on both
        # windows: the layers that do not install get them without a copy, the candidates gain on them.
        refresh = np.full(len(planned), True)
        for label, load in zip(("refresh_now", "refresh_before"), (rows[planned], previous_load)):
            old = (load, installed, self.shares[planned])
            refresh_gain = evaluate_gains(
                *old, installed, resolved, model, geometry, summary, label
            )
            snapshot_summary(summary, **{label + "_ms": refresh_gain})
            refresh &= refresh_gain > 0.0
        current = (rows[planned], installed, np.where(refresh[:, None], resolved, old[2]))
        gains = evaluate_gains(
            *current, new_layout, new_shares, model, geometry, summary, "candidate_now"
        )
        # A layer's copy: its busiest receiving GPU's replicas at the lower median of the copy times per
        # replica within the steps left (one slow copy prices nothing out; the start-up one counts).
        fresh = (new_layout >= 0) & (new_layout != installed)
        copied = fresh.reshape(-1, geometry.ep_size, geometry.local_slots).sum(2)
        recent = [ms for call, ms in self.copy_samples if call >= close - (steps_left or 0.0)]
        per_replica = sorted(recent)[(len(recent) - 1) // 2] if recent else None
        # The home layout is a candidate too: retiring the replicas copies nothing, so the plan stays
        # only where it gains more than the home layout, net of a layer's copy over the steps left.
        home_shares = (home >= 0).astype(np.float64)
        home_gains = evaluate_gains(
            *current, home, home_shares, model, geometry, summary, "home_now"
        )
        left, copy = steps_left or 0.0, (per_replica or 0.0) * copied.max(1, initial=0)
        snapshot_summary(
            summary,
            candidate_now_ms=gains,
            home_now_ms=home_gains,
            per_replica_ms=per_replica,
            copy_ms=copy,
        )
        to_home = (home_gains > 0.0) & (home_gains * left >= gains * left - copy)
        # Both candidates must also gain on that window, and the plan beat the home layout there.
        earlier = (previous_load, *current[1:])
        plan_gains = evaluate_gains(
            *earlier, new_layout, new_shares, model, geometry, summary, "candidate_before"
        )
        back_gains = evaluate_gains(
            *earlier, home, home_shares, model, geometry, summary, "home_before"
        )
        to_home |= (home_gains > 0.0) & (back_gains * left >= plan_gains * left - copy)
        snapshot_summary(summary, candidate_before_ms=plan_gains, home_before_ms=back_gains)
        gains, home_gains = np.minimum(gains, plan_gains), np.minimum(home_gains, back_gains)
        snapshot_summary(summary, candidate_gain_ms=gains, home_gain_ms=home_gains)
        new_layout[to_home], new_shares[to_home] = home[to_home], home_shares[to_home]
        gains[to_home] = home_gains[to_home]
        gain = dict(zip(planned.tolist(), gains.tolist()))
        retire = [layer for layer in planned[to_home].tolist() if gain.pop(layer) * left > 0.0]
        chosen = choose_layers_to_install(gain, steps_left, dict(zip(planned.tolist(), copy)))
        snapshot_summary(summary, payback_eligible=chosen.copy())
        if per_replica is None:
            # No copy has been timed yet: install the best layer alone to time one.
            chosen = chosen[:1]
        chosen += retire  # a retirement copies nothing: it pays as soon as it gains
        snapshot_summary(
            summary,
            layers=planned,
            refresh=refresh,
            to_home=to_home,
            admission_gain_ms=gains,
            chosen=chosen,
            retire=retire,
            copy_price_missing=per_replica is None,
        )
        growth = self.evaluations.read_cost(
            range(layers), rows, self.layout, self.shares, self.open_call
        )
        self.kept.append((window, rows, requests))
        moved, received = [], np.zeros(geometry.ep_size)
        for i, layer in enumerate(planned.tolist()):
            if layer not in chosen:
                if not refresh[i]:
                    continue
                new_layout[i], new_shares[i] = installed[i], current[2][i]
            tables = build_route_tables(new_layout[i], new_shares[i], experts, slots).packed()
            update.tables[layer] = np.array(tables, dtype=np.int32)
            fresh = (new_layout[i] >= 0) & (new_layout[i] != self.layout[layer])
            if fresh.any():
                moved.append(layer)
                received += copied[i]
                update.copy_items[layer] = copy_items(
                    self.layout[layer], new_layout[i], geometry, cfg.ep_rank, cfg.ranks_per_node
                )
            self.layout[layer] = new_layout[i]
            self.shares[layer] = new_shares[i]
            self.committed[layer] = window
        update.moved = sorted(moved)
        self.awaiting_copy[window] = (close, growth, steps_left, int(received.max()))
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
    """One rank's fresh interpreter without MPI/GPU; shares gathered stats and loads our ops."""

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

    def ask(self, message: tuple) -> tuple:
        """Send ``message`` and wait for the answer; ``RuntimeError`` once the process failed or exited."""
        self._conn.send(message)
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
    """Ordered blocking jobs; inbox carries results/errors, and jobs.join waits for completion."""

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


class _ApplyProfile:
    """Diagnostic clocks only; residuals include bookkeeping, never subtract from the gate."""

    def __init__(self, runtime, call, window):
        self.wall = time.perf_counter_ns()
        self.cpu = time.thread_time_ns()
        self.timeline = getattr(getattr(runtime, "_geneplb_readout", None), "timeline", None)
        self.record = dict(
            phase="apply_segments",
            edge="event",
            window=window,
            call=call,
            thread=threading.get_native_id(),
            segments=[],
            failed=False,
            storage=getattr(runtime, "_apply_profile_metadata", None),
        )

    @contextmanager
    def section(self, name):
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        try:
            yield
        except BaseException:
            self.record["failed"] = True
            raise
        finally:
            end_cpu = time.thread_time_ns()
            end_wall = time.perf_counter_ns()
            self.record["segments"].append(
                dict(name=name, wall_ns=end_wall - wall, thread_cpu_ns=end_cpu - cpu)
            )

    def finish(self):
        self.record["ns"] = time.perf_counter_ns()
        self.record["total_wall_ns"] = self.record["ns"] - self.wall
        self.record["total_thread_cpu_ns"] = time.thread_time_ns() - self.cpu
        for kind in ("wall", "thread_cpu"):
            self.record[f"bookkeeping_{kind}_ns"] = self.record[f"total_{kind}_ns"] - sum(
                item[f"{kind}_ns"] for item in self.record["segments"]
            )
        sink_wall, sink_cpu = time.perf_counter_ns(), time.thread_time_ns()
        if self.timeline is not None:
            self.timeline.records.append(self.record)
        self.record["sink_thread_cpu_ns"] = time.thread_time_ns() - sink_cpu
        self.record["sink_wall_ns"] = time.perf_counter_ns() - sink_wall


class AdaptiveRuntime:
    """One rank's windows: gather statistics/ROW from every rank, then apply one update at a time.
    Statistics copies finish before gather; tables apply before the next window opens.
    Errors stop further windows/installs, retaining valid tables and the server; copies use free slots.
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
        self.leave = threading.Event()
        self.gathers_posted = self._gathers_run = 0  # all-gathers posted to / run by the worker
        self.worker = _WorkerThread()
        # Closed before the interpreter finalizes even without a shutdown: a daemon worker that
        # returns from a collective while the interpreter finalizes ends inside the C++ call, and
        # that aborts the process.
        self._closed = False
        atexit.register(self.close)
        self.call = -1
        self.window = 0
        self.open_at: int | None = 0
        self.close_at: int | None = None
        self.window_opened_call = -1
        self.window_forwards = 1
        self.applied_window = np.full(len(layers), -1.0)
        self.wants_request_progress = False
        self._progress: tuple[int, GenerationProgress] | None = None
        self.last_applied = (-1.0, 0.0, 0.0)
        self.pending_updates: deque[tuple[int, LayoutUpdate]] = deque()
        self._apply_events: deque = deque()
        self.closed_windows = 0
        self.installs = 0
        self.installed_layers = 0
        self.failure: BaseException | None = None
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
        self._apply_profile_metadata = dict(
            prefer_pinned=prefer_pinned(),
            pin_requested=pin,
            tensors={
                name: dict(
                    device=str(tensor.device),
                    dtype=str(tensor.dtype),
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    is_pinned=tensor.is_pinned(),
                )
                for name, tensor in (("host", self._host_tables), ("device", self.tables))
            },
        )
        self._stats_copy_begin = torch.cuda.Event(enable_timing=True)
        self._stats_copy_end = torch.cuda.Event(enable_timing=True)
        self._copy_begin = torch.cuda.Event(enable_timing=True, blocking=True)
        self._copy_end = torch.cuda.Event(enable_timing=True, blocking=True)

    def close(self, gathers: int = 0) -> None:
        """End the worker thread after its jobs, then the decision process; once, from the balancer's
        shutdown (``gathers``: the most all-gathers any EP rank posted) or at interpreter exit."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        self.worker.jobs.put(None)
        self.worker.join(timeout=CLOSE_WAIT_S)
        # Zeros for the all-gathers its worker missed: a window's statistics, then an install's ROW.
        for index in range(self._gathers_run, 0 if self.worker.is_alive() else gathers):
            row = torch.zeros(len(ROW), dtype=torch.float64)
            send = row if index % 2 else torch.zeros_like(self._own_stats)
            self.gather(send, [torch.empty_like(send) for _ in self._gathered], self.leave)
        if self.worker.is_alive():
            self.leave.set()
            self.worker.join(timeout=CLOSE_WAIT_S)
        if self.worker.is_alive():
            from tensorrt_llm.logger import logger

            logger.warning("adaptive EPLB: the worker thread did not end at its close")
        self.decision.close()

    def counters(self) -> dict[str, int]:
        """Counts of closed windows, applied route-table updates and installed layers."""
        return dict(
            windows=self.closed_windows,
            installs=self.installs,
            installedLayers=self.installed_layers,
        )

    def add_request_progress(self, num_requests: int, progress: GenerationProgress) -> None:
        """Keep the latest pre-close progress callback: remaining tokens, requests and accepted/step
        for requests [start, stop); the worker reads it after close."""
        self._progress = (num_requests, progress)
        self.wants_request_progress = False

    def boundary(self) -> None:
        """Advance windows, decisions, copies and route tables by one call; never waits. After a
        failure it only drops the worker's messages (:attr:`failure`)."""
        if self.failure is not None:
            self.worker.inbox.clear()
            return
        try:
            self._boundary()
        except Exception as error:  # noqa: BLE001 - an optional optimization never stops serving
            self._stop(error)

    def _stop(self, error: BaseException) -> None:
        """Stop after a failure: keep the route tables in use, apply nothing further, log once."""
        from tensorrt_llm.logger import logger

        self.failure = error
        self.pending_updates.clear()
        self.worker.inbox.clear()
        self.open_at = self.close_at = None
        self.wants_request_progress = False
        logger.error(
            f"adaptive EPLB stopped after {error!r}; the expert layouts in use stay as they are"
        )

    def _boundary(self) -> None:
        self.call += 1
        call = self.call
        worker = self.worker
        if worker.failure is not None:
            raise RuntimeError("adaptive EPLB worker thread failed") from worker.failure
        # The messages first: a window scheduled to open at this call opens at it.
        while worker.inbox:
            message = worker.inbox.popleft()
            if message[0] == "schedule":
                _, self.window, self.open_at, apply_call, plan, self.window_forwards = message[:6]
                self._event(
                    "scheduled",
                    window=self.window,
                    apply_call=apply_call,
                    next_open=self.open_at,
                    plan=plan,
                    forwards=self.window_forwards,
                    summary=message[6] if len(message) > 6 else None,
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
                    **({"summary": update.summary} if hasattr(update, "summary") else {}),
                )
                worker.post(lambda update=update: self._install(update))
                self.gathers_posted += 1
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
        if self.close_at == call + 1:
            self.wants_request_progress = True

    def _event(self, event: str, **data: object) -> None:
        """Log window/decision/install scalars and layer lists, excluding tables, copies and stats."""
        from tensorrt_llm.logger import logger

        shown = {k: v for k, v in data.items() if not isinstance(v, (dict, np.ndarray))}
        logger.debug(f"adaptive EPLB {event}: {', '.join(f'{k}={v}' for k, v in shown.items())}")

    def _open_window(self) -> None:
        """Zero the statistics."""
        self._storage.zero_()

    def _close_window(self) -> None:
        """Copy the statistics into this rank's window statistics, timing the copies, and have the
        worker thread gather them at once."""
        assert not self.pending_updates, "an update must apply before its next window closes"
        self.closed_windows += 1
        self._stats_copy_begin.record()
        self._own_raw.copy_(self._storage, non_blocking=True)
        self._stats_copy_end.record()
        self.close_at = None
        progress, window = self._progress, self.window
        self.worker.post(lambda: self._gather_window(progress, window))
        self.gathers_posted += 1

    def _gather_window(self, progress: tuple | None, window: int) -> tuple:
        """Worker thread: read the requests' progress, complete this rank's window statistics,
        all-gather them and get the decision."""
        totals = (0, 0, 0) if progress is None else progress[1](0, progress[0])
        self._stats_copy_end.synchronize()
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
        self._gathers_run += 1
        self.gather(self._own_stats, self._gathered, self.leave)
        self._event(
            "window",
            window=window,
            open_call=self.window_opened_call,
            stats=self.decision.stats.numpy(),
        )
        return self.decision.ask(("window", window))

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
        self.worker.post(lambda: self.decision.ask(("copied", update.window, rows)))

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
        self._gathers_run += 1
        self.gather(torch.tensor(row, dtype=torch.float64), list(rows), self.leave)
        return ("copied", update, rows.numpy())

    def _apply_tables(self, call: int, update: LayoutUpdate) -> None:
        """Time one copy of the changed layer span, keeping other tables in it; no copy if empty.
        Subsequent forwards use the installed tables."""
        # The previous copy from the pinned host tables finished before the window after it closed.
        assert not self._apply_events, "the previous route tables' copy is still in flight"
        profile = _ApplyProfile(self, call, update.window)
        try:
            with profile.section("clock_probe"):
                pass
            with profile.section("host"):
                host_tables = self._host_tables_np
                for layer, tables in update.tables.items():
                    host_tables[layer] = tables
                    self.applied_window[layer] = update.window
            with profile.section("event_pool"):
                if self._apply_event_pairs:
                    begin, end = self._apply_event_pairs.popleft()
                else:
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
            with profile.section("begin_record"):
                begin.record()
            with profile.section("h2d"):
                if update.tables:
                    first, stop = min(update.tables), max(update.tables) + 1
                    self.tables[first:stop].copy_(self._host_tables[first:stop], non_blocking=True)
                    self.installs += 1
                    self.installed_layers += len(update.tables)
            with profile.section("end_record"):
                end.record()
            with profile.section("event_callback"):
                self._apply_events.append((update.window, len(update.tables), begin, end))
                self._event(
                    "applied", window=update.window, call=call, layers=sorted(update.tables)
                )
        finally:
            profile.finish()
