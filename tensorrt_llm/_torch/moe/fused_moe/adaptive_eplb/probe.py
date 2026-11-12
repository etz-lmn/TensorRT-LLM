# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement hook (fork measurement branch only): GPU timestamps inside every MoE layer's pass.

Off unless ``TRTLLM_MOE_PROBE_DIR`` names a directory: then nothing of this module is imported, allocated or launched.
On, ``ConfigurableMoE.forward`` stamps the GPU global timer on the current stream right before (``pass_start``) and
right after (``pass_end``) its scheduler's pass, whatever the backend. With ``TRTLLM_MOE_PROBE_POINTS=stages`` a
fused-communication (MegaMoE) layer is also stamped once its routes are ready, before the input quantization
(``routed``), and right before and right after the MegaMoE kernel (``kernel_start``, ``kernel_end``), so its pass splits
into the routing (top-k, the load balancer's route and statistics), the quantization and route staging, the MegaMoE
kernel (dispatch, both GEMMs and the combine) and what follows it. Each stamp is one single-thread kernel, so a CUDA
graph replays the stamps with the forward; each stamp adds a kernel boundary to the stream.

The stamps go into a device ring of ``RING`` forwards by ``LAYERS`` layers. A forward starts when a ``pass_start``
layer index is smaller than the last one; a layer passed several times in one forward (the MTP layer, once per draft
token) keeps its last pass. A thread of the rank reads the forward counter every ``POLL_S`` seconds on its own stream
and writes the ring to ``<dir>/rank<r>.moe-probe.npz`` (a whole file each time, renamed into place) once no forward has
started for ``IDLE_S`` seconds (the run's end, before its shutdown) and at least every ``FLUSH_S`` seconds while
forwards run; the exit writes it once more with ``final`` set. Per row: the forward number and layer 0's
``pass_start`` (ns); per row and layer: the pass's duration (ns, 0 where the pass was not stamped whole), its token
count and, with ``stages``, the four parts of the pass (ns, 0 where a part was not stamped).
"""

import atexit
import os
import threading
import time

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer

RING = 32768
"""Forwards the ring keeps: a sustained decode run of about 25,000 steps fits whole."""
LAYERS = 64
POINTS = ("pass_start", "routed", "kernel_start", "kernel_end", "pass_end")
STAGES = ("routing", "quantize_and_stage", "kernel", "after_kernel")
POINTS_ENV = "TRTLLM_MOE_PROBE_POINTS"
TOKENS = len(POINTS)  # the column of a row's token count
POLL_S, IDLE_S, FLUSH_S = 2.0, 4.0, 120.0

_probe = None


@triton.jit
def _stamp_kernel(ring_ptr, state_ptr, layer, point, tokens, num_layers, ring_size, columns):
    forward = tl.load(state_ptr)
    if point == 0:
        if layer < tl.load(state_ptr + 1):
            forward += 1
            tl.store(state_ptr, forward)
        tl.store(state_ptr + 1, layer)
    row = ((forward % ring_size) * num_layers + layer) * columns
    tl.store(ring_ptr + row + point, globaltimer())
    if point == 0:
        tl.store(ring_ptr + row + columns - 1, tokens)


class MoeProbe:
    """One rank's ring of the stamps per forward and layer, and the thread that writes it."""

    def __init__(self, directory: str, rank: int, stages: bool = False):
        self.path = os.path.join(directory, f"rank{rank}.moe-probe.npz")
        self.stages = stages
        self.ring = torch.zeros(RING, LAYERS, TOKENS + 1, dtype=torch.int64, device="cuda")
        self.state = torch.tensor([0, LAYERS], dtype=torch.int64, device="cuda")
        self.stream = torch.cuda.Stream()
        self.lock = threading.Lock()
        self.layer = None  # the layer whose pass is open (host side), None between passes
        threading.Thread(target=self._flusher, args=(torch.cuda.current_device(),), daemon=True).start()
        atexit.register(self.write, True)

    def _stamp(self, layer: int, point: int, tokens: int) -> None:
        _stamp_kernel[(1,)](self.ring, self.state, layer, point, tokens, LAYERS, RING, TOKENS + 1)

    def stamp(self, layer: int, phase: int, tokens: int) -> None:
        """``phase`` 0 opens the layer's pass (``pass_start``), 1 closes it (``pass_end``)."""
        if not 0 <= layer < LAYERS:
            return
        if phase == 0:
            self.layer = layer
            self._stamp(layer, 0, tokens)
        else:
            self._stamp(layer, len(POINTS) - 1, 0)
            self.layer = None

    def stamp_point(self, point: str) -> None:
        """A stamp inside the open pass (``routed``, ``kernel_start``, ``kernel_end``); none outside a pass or
        without ``stages``."""
        if self.stages and self.layer is not None:
            self._stamp(self.layer, POINTS.index(point), 0)

    def _flusher(self, device: int) -> None:
        torch.cuda.set_device(device)
        seen, written, changed, flushed = 0, 0, time.monotonic(), time.monotonic()
        while True:
            time.sleep(POLL_S)
            with torch.cuda.stream(self.stream):
                forwards = int(self.state[0].to("cpu"))
            now = time.monotonic()
            if forwards != seen:
                seen, changed = forwards, now
            if seen != written and (now - changed >= IDLE_S or now - flushed >= FLUSH_S):
                self.write(False)
                written, flushed = seen, now

    def write(self, final: bool) -> None:
        with self.lock, torch.cuda.stream(self.stream):
            ring, forwards = self.ring.cpu().numpy(), int(self.state[0].cpu())
            number = forwards - (forwards - np.arange(RING)) % RING  # the forward each row holds last
            stamps = ring[:, :, :TOKENS]
            first, last = stamps[:, :, 0], stamps[:, :, -1]
            whole = (first > 0) & (last >= first)
            out = dict(
                forward=number,
                start_ns=first[:, 0],
                tokens=ring[:, :, TOKENS].astype(np.int32),
                duration_ns=np.where(whole, last - first, 0).astype(np.int32),
                forwards=forwards,
                final=final,
                written_ns=time.time_ns(),
                points=np.array(POINTS),
            )
            if self.stages:
                parts = np.diff(stamps, axis=2)
                reached = (stamps[:, :, :-1] > 0) & (stamps[:, :, 1:] > 0) & (parts >= 0)
                out["stage_ns"] = np.where(reached & whole[:, :, None], parts, 0).astype(np.int32)
                out["stages"] = np.array(STAGES)
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp.npz"
            np.savez(tmp, **out)
            os.replace(tmp, self.path)


def probe_for(directory: str, rank: int) -> MoeProbe:
    """The process's probe (made on first use, on the current device)."""
    global _probe
    if _probe is None:
        _probe = MoeProbe(directory, rank, os.environ.get(POINTS_ENV, "pass") == "stages")
    return _probe


def active_probe():
    """The process's probe once a pass made it, else None."""
    return _probe
