# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the measurement-only admission gate."""

import dataclasses

import pytest

from tensorrt_llm._torch.pyexecutor.measure_admission_gate import (
    AFTER_REQUESTS,
    BURST_THETA,
    CAPS,
    STALL_STEPS,
    TIMEOUT_S,
    AdmissionBursts,
    AdmissionCaps,
    AdmissionGate,
    MeasuredAdmission,
    simulate_bursts,
)

pytestmark = pytest.mark.cpu_only


def test_unset_gate_is_always_open():
    gate = AdmissionGate(environ={})
    assert gate.admits(0) and gate.admits(5)


def test_gate_opens_once_the_whole_run_is_queued_and_stays_open():
    clock = [0.0]
    gate = AdmissionGate(environ={AFTER_REQUESTS: "3"}, clock=lambda: clock[0])
    assert [gate.admits(n) for n in (0, 1, 2)] == [False] * 3
    assert gate.admits(3)
    assert gate.admits(0)  # admitted requests leave the queue: the gate stays open


def test_a_partial_queue_past_the_timeout_stops_the_run():
    clock = [0.0]
    gate = AdmissionGate(environ={AFTER_REQUESTS: "4", TIMEOUT_S: "10"}, clock=lambda: clock[0])
    assert not gate.admits(0)  # nothing queued yet: the timeout has not started
    clock[0] = 100.0
    assert not gate.admits(1)
    clock[0] = 109.0
    assert not gate.admits(2)
    clock[0] = 111.0
    with pytest.raises(RuntimeError, match="2 of 4 requests queued"):
        gate.admits(2)


@dataclasses.dataclass
class _State:
    rank: int
    num_active_requests: int


def test_caps_follow_each_ranks_admitted_count():
    caps = AdmissionCaps(environ={CAPS: "4@3,2@2,4"})
    assert [caps.cap(n) for n in range(7)] == [4, 4, 4, 2, 2, 4, 4]
    states = [_State(0, 3), _State(1, 1)]
    caps.record({0: [None] * 3, 1: [None]})  # rank 0 is in its capped stretch, rank 1 is not
    assert [s.num_active_requests for s in caps.adjust(states, 4)] == [5, 1]
    assert AdmissionCaps(environ={}).adjust(states, 4) is states


def test_malformed_caps_refused():
    for spec in ("4,2@2", "0", "4@3,2@2"):
        with pytest.raises(ValueError):
            AdmissionCaps(environ={CAPS: spec})


def test_a_burst_admits_each_ranks_quota_and_closes_although_requests_keep_finishing():
    bursts = AdmissionBursts(environ={BURST_THETA: "0.75"})
    limits, plenty = [4, 4], lambda: {0: 99, 1: 99}
    assert bursts.admits([0, 1], [0, 0], limits, plenty)  # the start: quotas 4 and 4
    bursts.record({0: [None] * 4, 1: [None] * 2})
    # Rank 0 is done; a request of rank 0 finished meanwhile: it is not refilled in this burst.
    states = [_State(0, 3), _State(1, 2)]
    assert [s.num_active_requests for s in bursts.adjust(states, 4)] == [4, 2]
    assert bursts.admits([0, 1], [3, 2], limits, plenty)
    bursts.record({1: [None] * 2})
    assert not bursts.admits([0, 1], [3, 4], limits, plenty)  # every quota used (7 > 0.75 x 8)
    assert AdmissionBursts(environ={}).admits([0, 1], [0, 0], limits, plenty)
    with pytest.raises(ValueError):
        AdmissionBursts(environ={BURST_THETA: "1.5"})


def test_a_rank_with_nothing_left_waiting_does_not_hold_a_burst_open():
    # The stall of the first quota version: a rank whose list ran out kept its quota, every other
    # rank had used its own, and no burst ever closed or opened again.
    bursts = AdmissionBursts(environ={BURST_THETA: "0.75"})
    waiting = {0: 5, 1: 0}
    assert bursts.admits([0, 1], [0, 2], [4, 4], lambda: dict(waiting))
    assert bursts.quota == {0: 4, 1: 0}
    bursts.record({0: [None] * 4})
    waiting[0] = 1
    assert not bursts.admits([0, 1], [4, 2], [4, 4], lambda: dict(waiting))


# Quantiles (every 5 %) of the decoding iterations of 51200 natural end-of-sequence requests of
# prompt set A at 64 per GPU (the step-aligned trial's R run): the request lifetimes in steps.
LIFETIMES = [
    4,
    24,
    27,
    29,
    30,
    32,
    34,
    38,
    42,
    49,
    56,
    66,
    79,
    101,
    139,
    200,
    225,
    243,
    265,
    311,
    476,
]


def test_bursts_leave_most_steps_pure_decode_with_the_measured_lifetimes():
    # The simulation drives AdmissionBursts itself (the gate and a model of it once disagreed).
    share, fractions, left = simulate_bursts(
        LIFETIMES, ranks=32, target=64, theta=0.9, step_tokens=4096, prompt=512, steps=3000
    )
    assert share >= 0.7
    assert min(fractions) >= 0.85  # the pure steps run near the full batch


def test_bursts_drain_finite_per_rank_lists():
    _, _, left = simulate_bursts(LIFETIMES, 32, 64, 0.9, 4096, 512, steps=4000, per_rank=1600 // 8)
    assert not any(left.values())


def test_the_executor_admission_path_admits_every_request_in_alternating_bursts():
    # The executor's own pieces: the waiting queue, the strict-rank fetch, the explicit-rank
    # routing and MeasuredAdmission; fake requests finish on a schedule that empties some ranks'
    # lists early (the stall of the first quota version).
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
    from tensorrt_llm._torch.pyexecutor.request_utils import get_from_waiting_queue
    from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import ADPRouter
    from tensorrt_llm._torch.pyexecutor.scheduler.rank_state import RankState
    from tensorrt_llm._torch.pyexecutor.scheduler.waiting_queue import (
        DEFAULT_REQUEST_PRIORITY,
        FCFSWaitingQueue,
    )
    from tensorrt_llm.scheduling_params import SchedulingParams

    ranks, target, total = 4, 8, 4 * 60
    env = {AFTER_REQUESTS: str(total), BURST_THETA: "0.75", STALL_STEPS: "200"}
    admission, queue = MeasuredAdmission(environ=env), FCFSWaitingQueue()
    for k in range(total):
        params = SchedulingParams(attention_dp_rank=k % ranks, attention_dp_relax=False)
        request = SimpleNamespace(py_scheduling_params=params, priority=DEFAULT_REQUEST_PRIORITY)
        queue.add_request(RequestQueueItem(k, request))
    active, admitted, opened, was_open = {r: [] for r in range(ranks)}, 0, 0, False
    for _ in range(5000):
        states = [RankState(rank=r, num_active_requests=len(active[r])) for r in range(ranks)]
        admit, states = admission.step(states, queue, target)
        routed = {r: [] for r in range(ranks)}
        if admit:
            counts = [s.num_active_requests for s in states]
            free = ranks * target - sum(len(v) for v in active.values())
            items = get_from_waiting_queue(queue, free, True, target, list(counts))
            assert not ADPRouter._assign_explicit_dp_ranks(items, routed, counts, target)
        admission.record(routed)
        for r, items in routed.items():
            active[r] += [3 + (item.id % 17) * (1 + r) for item in items]  # rank 3's live longest
        admitted += sum(len(v) for v in routed.values())
        is_open = admission.bursts.quota is not None
        opened += is_open and not was_open
        was_open = is_open
        active = {r: [t - 1 for t in v if t > 1] for r, v in active.items()}
        if admitted == total and not any(active.values()):
            break
    assert admitted == total and not queue
    assert opened > 10  # bursts opened and closed again many times


def test_a_stalled_admission_fails_loudly():
    admission = MeasuredAdmission(environ={STALL_STEPS: "5"})
    queue = [object()]  # one waiting request that is never admitted
    for _ in range(5):
        admission.step([_State(0, 0)], queue, 4)
    with pytest.raises(RuntimeError, match="nothing admitted for 5 steps"):
        admission.step([_State(0, 0)], queue, 4)


def test_unset_admission_leaves_the_executor_admission_unchanged():
    # Every measurement variable unset: each step admits and hands the executor its own rank
    # states (the same objects), whatever the queue, the actives and the records, so the executor
    # admits as it does without this module.
    admission = MeasuredAdmission(environ={})
    states = [_State(0, 4), _State(1, 0)]
    for queue in ([], [object()], [object()] * 9):
        for _ in range(3):
            admit, seen = admission.step(states, queue, 4)
            assert admit and seen is states
            admission.record({0: [], 1: [object()]})
    assert admission.step(None, [object()], 4) == (True, None)
