# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-only admission gate: admit nothing until every request of the run is queued.

A measurement that submits all of its requests at once, each pinned to an attention DP rank, gets a
schedule that depends only on the outputs once the whole run is in the waiting queue: from then on a
rank admits its next request exactly at the step one of its requests finishes. Requests reach the
waiting queue on host timing, so the gate holds every admission until the queue holds
``TLLM_MEASURE_ADMIT_AFTER_REQUESTS`` requests; ``TLLM_MEASURE_ADMIT_TIMEOUT_S`` seconds after the
first one arrived without that many, the run stops with an error. Unset, the gate is always open.
The waiting queue is replicated on every rank, so every rank opens at the same step.

``TLLM_MEASURE_ADMIT_CAPS`` changes a rank's concurrency within such a run without changing
``max_batch_size``: ``"64@800,32@400,64"`` caps a rank at 64 active requests while it has admitted
fewer than 800, at 32 for its next 400 admissions, then at 64 again. A capped rank admits nothing
until its active requests drain below the cap. The admitted counts come from the routing every
rank computes alike, so the caps follow the outputs only.

``TLLM_MEASURE_ADMIT_BURST_THETA`` admits in synchronized bursts: nothing is admitted until the
active requests of all ranks fall to theta times the sum of their caps; then each rank admits its
quota -- what it lacked of its cap when the burst opened, at most the requests still waiting for
it -- and the burst ends once every rank has admitted its quota or has none left waiting. Requests
that finish during a burst are replaced in the next one. Between bursts every step is a pure decode
step, and the bursts' timing follows the outputs only.

``TLLM_MEASURE_ADMIT_STALL_STEPS`` stops the run when nothing is admitted for that many steps while
requests wait and fewer than the caps are active: a stalled admission fails loudly, never spins.
``MeasuredAdmission`` composes the four for the executor (and for the tests, on the same path).
"""

import dataclasses
import os
import time

AFTER_REQUESTS = "TLLM_MEASURE_ADMIT_AFTER_REQUESTS"
TIMEOUT_S = "TLLM_MEASURE_ADMIT_TIMEOUT_S"
CAPS = "TLLM_MEASURE_ADMIT_CAPS"
BURST_THETA = "TLLM_MEASURE_ADMIT_BURST_THETA"
STALL_STEPS = "TLLM_MEASURE_ADMIT_STALL_STEPS"


def waiting_per_rank(waiting_queue):
    """Requests waiting per pinned attention DP rank (requests without a rank are not counted)."""
    counts = {}
    for item in waiting_queue:
        params = getattr(getattr(item, "request", None), "py_scheduling_params", None)
        rank = getattr(params, "attention_dp_rank", None)
        if rank is not None:
            counts[rank] = counts.get(rank, 0) + 1
    return counts


class AdmissionGate:
    def __init__(self, environ=os.environ, clock=time.monotonic) -> None:
        self.need = int(environ.get(AFTER_REQUESTS, "0"))
        if self.need < 0:
            raise ValueError(f"{AFTER_REQUESTS}={self.need} must be >= 0")
        self.timeout_s = float(environ.get(TIMEOUT_S, "600"))
        self.clock = clock
        self.is_open = self.need == 0
        self.first_seen = None

    def admits(self, queued: int) -> bool:
        """Whether this step may admit, given the waiting queue holds ``queued`` requests."""
        if self.is_open:
            return True
        if queued >= self.need:
            self.is_open = True
            return True
        if queued > 0 and self.first_seen is None:
            self.first_seen = self.clock()
        if self.first_seen is not None and self.clock() - self.first_seen > self.timeout_s:
            raise RuntimeError(
                f"{AFTER_REQUESTS}: {queued} of {self.need} requests queued after "
                f"{self.timeout_s:.0f} s; the run would start on a partial queue"
            )
        return False


class AdmissionCaps:
    """Per-rank caps on active requests by the rank's admitted-request count (``CAPS``)."""

    def __init__(self, environ=os.environ) -> None:
        self.segments = []  # (cap, admitted count at which the cap ends; None: open)
        start = 0
        parts = [p for p in environ.get(CAPS, "").split(",") if p]
        for k, part in enumerate(parts):
            cap, _, count = part.partition("@")
            if int(cap) < 1 or (not count) != (k == len(parts) - 1):
                raise ValueError(f"{CAPS}={environ.get(CAPS)!r}: cap@count,..., the last open")
            start = start + int(count) if count else None
            self.segments.append((int(cap), start))
        self.admitted = {}

    def cap(self, admitted: int) -> int:
        """The cap of a rank that has admitted ``admitted`` requests."""
        return next(cap for cap, end in self.segments if end is None or admitted < end)

    def adjust(self, states, max_active: int):
        """Copies of the rank states whose active counts make the admission's per-rank bound
        ``max_active`` act as each rank's cap; the states themselves when no caps are set."""
        if not self.segments or states is None:
            return states
        return [
            dataclasses.replace(
                s,
                num_active_requests=s.num_active_requests
                + max(0, max_active - self.cap(self.admitted.get(s.rank, 0))),
            )
            for s in states
        ]

    def limits(self, states, max_active: int):
        """Each rank's cap (``max_active`` without caps); None without rank states."""
        if states is None:
            return None
        if not self.segments:
            return [max_active] * len(states)
        return [self.cap(self.admitted.get(s.rank, 0)) for s in states]

    def record(self, new_per_rank) -> None:
        """Count the requests just routed to each rank (``{rank: [items]}``)."""
        for rank, items in new_per_rank.items():
            self.admitted[rank] = self.admitted.get(rank, 0) + len(items)


class AdmissionBursts:
    """Synchronized admission bursts with per-rank quotas (``BURST_THETA``); unset, always open."""

    def __init__(self, environ=os.environ) -> None:
        value = environ.get(BURST_THETA, "")
        self.theta = float(value) if value else None
        if self.theta is not None and not 0.0 < self.theta < 1.0:
            raise ValueError(f"{BURST_THETA}={value!r} must be in (0, 1)")
        self.quota = None  # rank -> admissions left in the open burst

    def admits(self, ranks, active, limits, waiting) -> bool:
        """Whether this step admits, given every rank's id, active requests and cap; ``waiting()``
        returns the requests waiting per rank (called only when a burst opens or is open)."""
        if self.theta is None or limits is None:
            return True
        if self.quota is not None:
            left = waiting()
            self.quota = {r: min(q, left.get(r, 0)) for r, q in self.quota.items()}
            if all(q <= 0 for q in self.quota.values()):
                self.quota = None
        if self.quota is None and sum(active) <= self.theta * sum(limits):
            left = waiting()
            quota = {
                r: min(max(0, c - a), left.get(r, 0)) for r, a, c in zip(ranks, active, limits)
            }
            self.quota = quota if any(q > 0 for q in quota.values()) else None
        return self.quota is not None

    def adjust(self, states, max_active: int):
        """Copies of the rank states that let a rank admit at most its quota left in this burst."""
        if self.theta is None or self.quota is None or states is None:
            return states
        return [
            dataclasses.replace(
                s,
                num_active_requests=max(
                    s.num_active_requests, max_active - self.quota.get(s.rank, 0)
                ),
            )
            for s in states
        ]

    def record(self, new_per_rank) -> None:
        """Count the requests just routed to each rank against its quota."""
        if self.quota is not None:
            for rank, items in new_per_rank.items():
                self.quota[rank] = self.quota.get(rank, 0) - len(items)


class MeasuredAdmission:
    """The measurement's admission step: the queue gate, the per-rank caps, the bursts and the
    stall guard, as the executor applies them each step (``step`` before the admission, ``record``
    with the requests routed to each rank after it)."""

    def __init__(self, environ=os.environ) -> None:
        self.gate = AdmissionGate(environ)
        self.caps = AdmissionCaps(environ)
        self.bursts = AdmissionBursts(environ)
        self.stall_steps = int(environ.get(STALL_STEPS, "0"))
        self.idle = 0

    def step(self, states, waiting_queue, max_active: int):
        """-> (whether this step admits, the rank states the admission reads)."""
        admit = self.gate.admits(len(waiting_queue))
        if states is None or not admit:
            return admit, states
        limits = self.caps.limits(states, max_active)
        active = [s.num_active_requests for s in states]
        if self.stall_steps and len(waiting_queue) and sum(active) < sum(limits):
            self.idle += 1
            if self.idle > self.stall_steps:
                raise RuntimeError(
                    f"{STALL_STEPS}: nothing admitted for {self.idle - 1} steps with "
                    f"{len(waiting_queue)} requests waiting and {sum(active)} of "
                    f"{sum(limits)} active"
                )
        admit = self.bursts.admits(
            [s.rank for s in states], active, limits, lambda: waiting_per_rank(waiting_queue)
        )
        states = self.bursts.adjust(self.caps.adjust(states, max_active), max_active)
        return admit, states

    def record(self, new_per_rank) -> None:
        if any(new_per_rank.values()):
            self.idle = 0
        self.caps.record(new_per_rank)
        self.bursts.record(new_per_rank)


def simulate_bursts(
    lifetimes, ranks, target, theta, step_tokens, prompt, steps, seed=0, per_rank=None
):
    """Pure-decode share and the pure steps' mean active fraction under ``AdmissionBursts`` itself:
    ``ranks`` ranks of ``target`` slots, request lifetimes in steps drawn from ``lifetimes``, a
    rank prefilling at most ``(step_tokens - 4 x active) // prompt`` prompts per step (MTP 3 drafts:
    4 decode tokens per active request); ``per_rank`` requests per rank (None: endless). ->
    (pure share, [active fraction of each pure step], requests left waiting per rank)."""
    import random

    rng = random.Random(seed)
    bursts = AdmissionBursts(environ={BURST_THETA: str(theta)})
    left = {r: [] for r in range(ranks)}
    queue = {r: (per_rank if per_rank is not None else 10**9) for r in range(ranks)}
    pure, fractions = 0, []
    for _ in range(steps):
        active = [len(left[r]) for r in range(ranks)]
        admitted = {r: [] for r in range(ranks)}
        if bursts.admits(list(range(ranks)), active, [target] * ranks, lambda: dict(queue)):
            for r in range(ranks):
                budget = (step_tokens - 4 * active[r]) // prompt
                room = min(target - active[r], bursts.quota[r], budget, queue[r])
                admitted[r] = [rng.choice(lifetimes) for _ in range(max(0, room))]
                queue[r] -= len(admitted[r])
            bursts.record(admitted)
        if any(admitted.values()):
            for r in range(ranks):
                left[r] += admitted[r]
        else:
            pure += 1
            fractions.append(sum(active) / (ranks * target))
        for r in range(ranks):
            left[r] = [v - 1 for v in left[r] if v > 1]
    return pure / steps, fractions, queue
