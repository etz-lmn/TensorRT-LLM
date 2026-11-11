# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Measurement only: the rows every MoE layer routes to each logical expert in every forward of the
serving loop, from this rank's tokens, live rows only.

Switch: ``TRTLLM_MOE_ROW_COUNT_DIR=<directory>``. Unset or empty, :data:`ENABLED` is false: the MoE
schedulers never call :func:`record` and the decoder runner builds no counter, so nothing is
allocated or launched.

Count. Each MoE layer hands its expert ids ``[rows, top_k]`` (after routing, before EPLB maps them
to slots) to :func:`record`. One launch of ``trtllm::adaptive_eplb_record`` adds them to the layer's
row of a device accumulator, reading the forward's live-row word (:func:`live_rows.current_word`):
a launch over the whole forward counts its leading ``T / P * R`` rows only, so the rows of CUDA graph
padding requests and of attention-DP placeholder requests are left out; a launch over part of the
forward counts every row. The launch is captured with the forward's CUDA graph.

Forwards. The decoder runner calls :meth:`RowCounter.begin` before and :meth:`RowCounter.end` after
each forward that the executor marked with its iteration (:func:`set_iter`), never for a warmup or
dummy forward. ``begin`` zeroes the accumulator on the forward's stream, which wipes whatever an
unrecorded forward (warmup, graph capture, dummy) added; ``end`` enqueues on the same stream a copy
into a pinned host slot and an event. A writer thread waits for the event and appends one JSON line
to ``<directory>/rank<rank>.moe-rows.jsonl``, flushed line by line: a killed process keeps every
line it completed, and a reader drops a last line without its newline. The writer starts at the
first recorded forward: while the engine warms up and captures its CUDA graphs no thread of the
counter exists, since a CUDA call from another thread invalidates a capture in the global mode.

Lines: ``header`` (the geometry), one ``forward`` per recorded forward (the executor iteration, the
CUDA graph key, the live-row word, the host's monotonic clock at the forward's end, per layer slot
the routing launches and the total, the counts as base64 of zlib of little-endian unsigned integers
``[layer slots, experts]`` in row-major order), ``end`` at exit.
"""

import atexit
import base64
import json
import os
import queue
import socket
import threading
import time
import zlib
from typing import Optional, Sequence

import numpy as np
import torch

from tensorrt_llm._torch.utils import get_model_extra_attrs

from . import live_rows

ENV = "TRTLLM_MOE_ROW_COUNT_DIR"
ENABLED = bool(os.environ.get(ENV))
SCHEMA = "trtllm_moe_row_count_v1"
FILE = "rank{rank}.moe-rows.jsonl"
# Key of the engine's counter in the per-forward model extra attrs.
MODEL_EXTRA_ATTR = "moe_row_counter"
# Pinned host slots between a forward's copy and the writer; a forward waits while none is free.
RING = 256
# Seconds between checks of the writer while a forward waits for a slot.
WAIT_S = 1.0

_iteration: Optional[int] = None


def set_iter(iter_id: Optional[int]) -> None:
    """The executor iteration whose forward runs next; None for a warmup pass."""
    global _iteration
    _iteration = iter_id


def take_iter() -> Optional[int]:
    """The iteration set for the forward about to run, once: a second forward of the pass reads None."""
    global _iteration
    iter_id, _iteration = _iteration, None
    return iter_id


def record(layer_idx: int, token_selected_experts: torch.Tensor, first: bool) -> None:
    """Count one routing launch of MoE layer ``layer_idx`` into the running engine's counter."""
    attrs = get_model_extra_attrs()
    counter = attrs.get(MODEL_EXTRA_ATTR) if attrs else None
    if counter is not None:
        counter.record(layer_idx, token_selected_experts, first)


def attach(model: torch.nn.Module, rank: int, device: torch.device) -> "RowCounter":
    """The counter of ``model``'s MoE layers, registered in the model's extra attrs."""
    if os.environ.get(live_rows.KEEP_ENV) == "1":
        raise RuntimeError(
            f"{ENV} counts the live rows only, but {live_rows.KEEP_ENV}=1 keeps every row: "
            "the counts would include the padding rows. Unset one of them."
        )
    from .fused_moe.moe_scheduler import MoEScheduler

    moes = [m for m in model.modules() if isinstance(getattr(m, "scheduler", None), MoEScheduler)]
    experts = {int(m.num_experts) for m in moes}
    top_k = {int(m.routing_method.experts_per_token) for m in moes}
    if len(experts) != 1 or len(top_k) != 1:
        raise RuntimeError(f"{ENV}: MoE layers with experts {experts} and top_k {top_k}")
    counter = RowCounter(
        os.environ[ENV],
        rank,
        sorted({int(m.layer_idx) for m in moes}),
        experts.pop(),
        top_k.pop(),
        device,
    )
    model.extra_attrs[MODEL_EXTRA_ATTR] = counter
    return counter


class RowCounter:
    """One rank's accumulator, its pinned host slots and the writer of its file."""

    def __init__(
        self,
        directory: str,
        rank: int,
        layers: Sequence[int],
        experts: int,
        top_k: int,
        device: torch.device,
        ring: int = RING,
    ) -> None:
        from .fused_moe.adaptive_eplb.policy import ExpertStatistics, statistics_buffers

        try:
            torch.ops.trtllm.adaptive_eplb_record
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(f"{ENV} needs the op trtllm::adaptive_eplb_record") from error
        device = torch.device(device)
        slots = max(layers) + 1
        buffers = statistics_buffers(slots, experts, device)
        self._storage = buffers["storage"]
        self._rows_bytes = buffers["rows"].numel() * 4
        self._launches_at = buffers["forwards"].data_ptr() - self._storage.data_ptr()
        self._shape = (slots, experts)
        enabled = torch.ones(1, dtype=torch.int32, device=device)
        self._layers = {layer: ExpertStatistics.of(buffers, layer, enabled) for layer in layers}
        # A host device copies in order and needs neither pinned memory nor events.
        cuda = device.type == "cuda"
        size = self._storage.numel()
        self._host = [torch.empty(size, dtype=torch.uint8, pin_memory=cuda) for _ in range(ring)]
        self._events = [torch.cuda.Event() if cuda else None for _ in range(ring)]
        self._free: queue.Queue = queue.Queue()
        for slot in range(ring):
            self._free.put(slot)
        self._work: queue.Queue = queue.Queue()
        self.forwards = self.stalls = self.unrecorded = 0
        self._error: Optional[str] = None
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, FILE.format(rank=rank))
        # A directory holds one run: a second writer of the same rank file is refused.
        self._file = open(self.path, "x")
        self._write(
            dict(
                kind="header",
                schema=SCHEMA,
                rank=rank,
                layers=list(layers),
                layer_slots=slots,
                experts=experts,
                top_k=top_k,
                ring=ring,
                host=socket.gethostname(),
                pid=os.getpid(),
                start_unix=time.time(),
            )
        )
        self._thread: Optional[threading.Thread] = None
        atexit.register(self.close)

    def record(self, layer_idx: int, token_selected_experts: torch.Tensor, first: bool) -> None:
        statistics = self._layers.get(layer_idx)
        if statistics is None:
            raise RuntimeError(f"{ENV}: MoE layer {layer_idx} is not one of {sorted(self._layers)}")
        ids = token_selected_experts
        statistics.record(ids, first, live_rows.current_word(ids.device))

    def begin(self) -> None:
        """Before a recorded forward: zero the accumulator on its stream."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{ENV}: a recorded forward is not captured")
        self._storage.zero_()

    def end(self, meta: dict) -> None:
        """After a recorded forward: copy the accumulator to a host slot for the writer."""
        stall = False
        while True:
            if self._error is not None:
                raise RuntimeError(f"{ENV}: the writer of {self.path} stopped: {self._error}")
            try:
                slot = self._free.get(block=stall, timeout=WAIT_S if stall else None)
                break
            except queue.Empty:
                stall = True
        self.stalls += stall
        if self._thread is None:
            self._thread = threading.Thread(target=self._drain, name="moe-row-count", daemon=True)
            self._thread.start()
        self._host[slot].copy_(self._storage, non_blocking=True)
        if self._events[slot] is not None:
            self._events[slot].record()
        self._work.put((slot, self.forwards, dict(meta, t=time.monotonic()), stall))
        self.forwards += 1

    def skipped(self) -> None:
        """A forward of the engine that is not recorded (warmup, dummy, no executor iteration)."""
        self.unrecorded += 1

    def drain(self) -> None:
        """Wait until the writer has written every recorded forward, or has stopped."""
        while self._error is None and self._free.qsize() < len(self._host):
            time.sleep(WAIT_S / 100)

    def close(self) -> None:
        if self._file is None:
            return
        if self._thread is not None:
            self._work.put(None)
            self._thread.join()
        self._write(
            dict(
                kind="end",
                forwards=self.forwards,
                stalls=self.stalls,
                unrecorded_forwards=self.unrecorded,
                error=self._error,
            )
        )
        self._file.close()
        self._file = None

    def _drain(self) -> None:
        while True:
            item = self._work.get()
            if item is None:
                return
            slot, n, meta, stall = item
            try:
                if self._events[slot] is not None:
                    self._events[slot].synchronize()
                self._write(self._line(slot, n, meta, stall))
            except Exception as error:  # noqa: BLE001 - reported by the next end() and the end line
                self._error = f"{type(error).__name__}: {error}"
                return
            finally:
                self._free.put(slot)

    def _line(self, slot: int, n: int, meta: dict, stall: bool) -> dict:
        raw = self._host[slot].numpy()
        rows = raw[: self._rows_bytes].view(np.float32).reshape(self._shape)
        launches = raw[self._launches_at : self._launches_at + 8 * self._shape[0]].view(np.int64)
        counts = rows.astype(np.int64)
        if not np.array_equal(counts, rows):
            raise ValueError(f"forward {n}: counts that are not integers")
        top = int(counts.max(initial=0))
        dtype = "<u1" if top < 1 << 8 else "<u2" if top < 1 << 16 else "<u4"
        payload = zlib.compress(counts.astype(dtype).tobytes(), 1)
        return dict(
            kind="forward",
            n=n,
            **meta,
            launches=launches.tolist(),
            totals=counts.sum(axis=1).tolist(),
            dtype=dtype,
            rows=base64.b64encode(payload).decode("ascii"),
            stall=stall,
        )

    def _write(self, doc: dict) -> None:
        self._file.write(json.dumps(doc, separators=(",", ":")) + "\n")
        self._file.flush()
