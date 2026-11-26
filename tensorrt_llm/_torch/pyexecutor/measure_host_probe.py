# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-only probes of the executor's main thread; not part of any feature.

Three settings, each off when unset or empty, all read once by :meth:`HostProbe.from_env`:

``TLLM_MEASURE_BOUNDARY_INJECT_US`` with ``TLLM_MEASURE_BOUNDARY_INJECT_EVERY``: adaptive EPLB's
per-forward boundary spins on the main thread after its own work, on every call whose number is
a multiple of EVERY, for the next dose of the comma list (``"250,250,4000"``: two such calls of
250 us, then one of 4 ms, repeating). The spin holds the GIL as Python work does. The call numbers
are the runtime's, the same on every EP rank, so every rank spins on the same forward. It measures
whether main-thread time at the boundary lengthens the step, and by how much.

``TLLM_MEASURE_STAGE_STEPS=1``: per pass of the overlap loop, the wall and thread-CPU nanoseconds of
every stage the executor marks (the stages of ``TLLM_EXECUTOR_STAGE_TIMING``, which this turns on)
and the pass's start (``perf_counter_ns``, the clock of the adaptive readout's entries). The loop's
end writes them to ``<ADAPTIVE_READOUT_DIR>/rank<rank>.stage-steps.json``. Wall minus CPU of a
stage is the time the thread was not running in it: a blocking wait, or a wait for the GIL.

``TLLM_MEASURE_SWITCH_INTERVAL_S``: the interpreter's thread switch interval of the executor's
process (``sys.setswitchinterval``; the default is 0.005 s).
"""

import array
import json
import os
import sys
import time

INJECT_US = "TLLM_MEASURE_BOUNDARY_INJECT_US"
INJECT_EVERY = "TLLM_MEASURE_BOUNDARY_INJECT_EVERY"
STAGE_STEPS = "TLLM_MEASURE_STAGE_STEPS"
SWITCH_INTERVAL_S = "TLLM_MEASURE_SWITCH_INTERVAL_S"
OUT_DIR = "ADAPTIVE_READOUT_DIR"
SCHEMA = "trtllm_measure_stage_steps_v1"


def spin(ns: int, clock=time.perf_counter_ns) -> None:
    """Busy-wait ``ns`` nanoseconds on this thread, holding the GIL."""
    end = clock() + ns
    while clock() < end:
        pass


def dose_of(call: int, every: int, doses_us: tuple) -> int:
    """Microseconds call number ``call`` spins: 0 unless ``call`` is a positive multiple of
    ``every``; else the doses in turn."""
    if call <= 0 or call % every:
        return 0
    return doses_us[(call // every - 1) % len(doses_us)]


def inject_boundary(cls, every: int, doses_us: tuple, sleep=spin) -> None:
    """Make ``cls._boundary`` (adaptive EPLB's ``AdaptiveRuntime``) spin its dose after its work, once
    per process: the executor is built twice (a profiling one, then the serving one), and a second
    patch over the first would spin every dose twice."""
    if getattr(cls._boundary, "measure_host_probe", None) is not None:
        return
    original = cls._boundary

    def _boundary(self):
        original(self)
        us = dose_of(self.call, every, doses_us)
        if us:
            sleep(us * 1000)

    _boundary.measure_host_probe = (every, doses_us)
    cls._boundary = _boundary


class StageSteps:
    """Per pass: its start and, per marked stage, the wall and thread-CPU nanoseconds."""

    def __init__(self, clock=time.perf_counter_ns, cpu=time.thread_time_ns):
        self.clock, self.cpu = clock, cpu
        self.names: list = []
        self.index: dict = {}
        self.starts = array.array("q")
        self.passes, self.stages = array.array("i"), array.array("i")
        self.wall, self.thread_cpu = array.array("q"), array.array("q")
        self._t = self._c = None

    def begin(self) -> None:
        self._t, self._c = self.clock(), self.cpu()
        self.starts.append(self._t)

    def mark(self, name: str) -> None:
        if self._t is None:
            return
        t, c = self.clock(), self.cpu()
        if name not in self.index:
            self.index[name] = len(self.names)
            self.names.append(name)
        self.passes.append(len(self.starts) - 1)
        self.stages.append(self.index[name])
        self.wall.append(t - self._t)
        self.thread_cpu.append(c - self._c)
        self._t, self._c = t, c

    def doc(self) -> dict:
        return dict(
            schema=SCHEMA,
            stages=self.names,
            pass_start_ns=list(self.starts),
            pass_index=list(self.passes),
            stage=list(self.stages),
            wall_ns=list(self.wall),
            thread_cpu_ns=list(self.thread_cpu),
        )


class HostProbe:
    """The probes the environment selects for one executor process."""

    def __init__(
        self, every=0, doses_us=(), stage_steps=False, switch_interval_s=None, out_dir=None
    ):
        self.every, self.doses_us = every, tuple(doses_us)
        self.steps = StageSteps() if stage_steps else None
        self.switch_interval_s, self.out_dir = switch_interval_s, out_dir
        self.written = None

    @classmethod
    def from_env(cls, environ=os.environ) -> "HostProbe":
        doses = tuple(int(x) for x in environ.get(INJECT_US, "").split(",") if x.strip())
        every = int(environ.get(INJECT_EVERY, "0") or 0)
        if bool(doses) != (every > 0) or any(d <= 0 for d in doses):
            raise ValueError(f"{INJECT_US} and {INJECT_EVERY} go together, with positive doses")
        steps = environ.get(STAGE_STEPS, "") == "1"
        out_dir = environ.get(OUT_DIR) or None
        if steps and not out_dir:
            raise ValueError(f"{STAGE_STEPS}=1 writes into {OUT_DIR}, which is unset")
        switch = environ.get(SWITCH_INTERVAL_S, "")
        return cls(every, doses, steps, float(switch) if switch else None, out_dir)

    def install(self, runtime_class=None) -> None:
        """Set the switch interval; with doses, patch ``runtime_class`` (by default adaptive EPLB's
        ``AdaptiveRuntime``, imported only then)."""
        if self.switch_interval_s is not None:
            sys.setswitchinterval(self.switch_interval_s)
        if self.every:
            if runtime_class is None:
                from ..moe.fused_moe.adaptive_eplb.runtime import AdaptiveRuntime as runtime_class
            inject_boundary(runtime_class, self.every, self.doses_us)

    def dump(self, rank: int):
        """Write the stage steps of this process once; the path, or None without them."""
        if self.steps is None or self.written is not None:
            return self.written
        path = os.path.join(self.out_dir, f"rank{rank}.stage-steps.json")
        doc = dict(self.steps.doc(), rank=rank, switch_interval_s=sys.getswitchinterval())
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f)
        os.replace(path + ".tmp", path)
        self.written = path
        return path
