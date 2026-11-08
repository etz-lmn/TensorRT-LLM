# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adaptive EPLB runtime: statistics windows, the decision process and installs."""

import contextlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import runtime as rt
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.balancer import (
    AdaptiveMoeLoadBalancer,
    ExpertStatistics,
    statistics_buffers,
)
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    EMA_DECAY,
    ExpertGeometry,
    TimeModel,
    even_shares,
)
from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

pytestmark = pytest.mark.cpu_only

E, R, LOCAL, L = 8, 2, 6, 2
GEOMETRY = ExpertGeometry(E, R * LOCAL, R)
MODEL = TimeModel(1.0, 1.0, 1.0, 0.5, 1.0, 128)
HOME = GEOMETRY.home_layout()
TABLE = 2 * (E + R * LOCAL)


def fake_event(enable_timing: bool = False, blocking: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        record=lambda stream=None: None,
        query=lambda: True,
        synchronize=lambda: None,
        elapsed_time=lambda end: 0.0,
    )


STREAM = SimpleNamespace(wait_event=lambda event: None)


@pytest.fixture
def host_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", fake_event)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: STREAM)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class FakeLayer:
    expert_count, num_slots = E, R * LOCAL

    def __init__(self, clock: list[int], buffers: dict, index: int) -> None:
        on = torch.ones(1, dtype=torch.int32)
        self.clock, self.statistics = clock, ExpertStatistics.of(buffers, index, on)

    def home_layout(self) -> list[int]:
        return HOME.tolist()


class FakeTransport:
    """A 0.25 ms batch completing ``delay`` calls on; ``waiting()`` jobs still wait for the clock."""

    def __init__(self, clock: list[int], delay: int = 1) -> None:
        self.clock, self.delay, self.batches, self.due = clock, delay, [], []
        self.waiting = lambda: sum(due > self.clock[0] for due in self.due)

    def copy(self, experts, stream, start, end) -> None:
        due = self.clock[0] + self.delay
        self.batches.append(list(experts))

        def synchronize() -> None:  # the test's settle guards the wait with its own deadline
            self.due.append(due)
            while self.clock[0] < due:
                time.sleep(0.0005)
            self.due.remove(due)

        end.synchronize, start.elapsed_time = synchronize, lambda end: 0.25


class FakeGroup:
    """Blocking all-gathers of in-process worker threads; ``waiting[rank]`` until all posted."""

    def __init__(self) -> None:
        self.posts, self.waiting = {}, {}
        self.ready = threading.Condition()

    def gather(self, rank: int):
        counts = [0]

        def start(send, rows) -> None:
            key = counts[0]
            counts[0] += 1
            with self.ready:
                posts = self.posts.setdefault(key, {})
                posts[rank] = send.clone()
                self.ready.notify_all()
                self.waiting[rank] = (posts, len(rows))
                self.ready.wait_for(lambda: len(posts) == len(rows), timeout=30)
                del self.waiting[rank]
                assert len(posts) == len(rows), f"gather {key}: {sorted(posts)} posted"
                for r, row in enumerate(rows):
                    row.copy_(posts[r])

        return start


class FakeDecision:
    """The decision process in this process, or the updates ``script(window)`` prepares."""

    def __init__(self, rank: int, script=None) -> None:
        self.stats = torch.zeros(R, rt.window_stats_size(L, E))
        homes = np.tile(HOME, (L, 1))
        config = rt.DecisionConfig(GEOMETRY, MODEL, homes, rank, 1)
        self.decider = rt._Decider(config, self.stats.numpy())
        self.script = script
        self.queue = []
        self.sent = []

    def submit(self, message) -> None:
        self.sent.append(message)
        if self.script is None:
            reply = self.decider.handle(message)
        elif message[0] == "window":
            reply = ("update", self.script(message[1]))
        else:
            apply_call = int(message[2][:, rt.ROW.index("copied_call")].max()) + 1
            reply = ("schedule", message[1] + 1, apply_call + 4, apply_call, None)
        self.queue.append(reply)

    def recv(self):
        return self.queue.pop(0)

    def close(self) -> None:
        pass


class Events:
    """A debug listener: the decision records of one rank, completed by the later events."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.records = []
        self.by_window = {}
        self.kinds = []
        self.stats = []
        self.open_calls = {}

    def __call__(self, event: str, data: dict) -> None:
        self.kinds.append(event)
        if event == "window":
            self.stats.append(data["stats"].copy())
            self.open_calls[data["window"]] = data["open_call"]
        elif event == "decision":
            record = dict(
                window=data["window"],
                open_call=self.open_calls[data["window"]],
                solved_call=data["call"],
                tables=sorted(data["tables"]),
                moved=data["moved"],
                effect={},
            )
            self.records.append(record)
            self.by_window[data["window"]] = record
        elif event == "copied":
            row = data["rows"][self.rank]
            self.by_window[data["window"]]["copied_call"] = int(row[rt.ROW.index("copied_call")])
        elif event == "scheduled":
            self.records[-1].update(
                apply_call=data["apply_call"], plan=data["plan"], next_open=data["next_open"]
            )
        elif event == "applied":
            self.by_window[data["window"]]["effect"].update(
                dict.fromkeys(data["layers"], data["call"])
            )


def records(runtime) -> list[dict]:
    return runtime.debug_listener.records


def make_ranks(decisions, clock, copy_delays=(1, 1)):
    group = FakeGroup()
    ranks = []
    for rank, (decision, delay) in enumerate(zip(decisions, copy_delays)):
        buffers = statistics_buffers(L, E, "cpu")
        layers = [FakeLayer(clock, buffers, index) for index in range(L)]
        transport = FakeTransport(clock, delay)
        gather = group.gather(rank)
        tables = torch.zeros(L, TABLE, dtype=torch.int32)
        runtime = rt.AdaptiveRuntime(
            layers, transport, STREAM, decision, gather, "cpu", tables, buffers
        )
        runtime.debug_listener = Events(rank)

        def waiting(t=transport, r=rank) -> int:
            waiting = group.waiting.get(r)
            return t.waiting() + (1 if waiting and len(waiting[0]) < waiting[1] else 0)

        runtime.waiting = waiting
        ranks.append(runtime)
    return ranks


def settle(ranks) -> None:
    """Let every worker thread run until its jobs are done or waiting (for the clock or a peer)."""
    deadline = time.monotonic() + 30
    while any(r.worker.jobs.unfinished_tasks > r.waiting() for r in ranks):
        assert time.monotonic() < deadline, [
            (r.worker.jobs.unfinished_tasks, r.waiting()) for r in ranks
        ]
        time.sleep(0.0002)
    for runtime in ranks:
        assert runtime.worker.failure is None, runtime.worker.failure


def run(ranks, clock, calls: int) -> None:
    """Calls as the executor makes them, each seeing its batch first (no requests here)."""
    for _ in range(calls):
        for runtime in ranks:
            if runtime.wants_request_progress:
                runtime.add_request_progress([])
            runtime.boundary()
        settle(ranks)
        clock[0] += 1


def set_statistics(runtime, ema: np.ndarray, forwards: int) -> None:
    for layer, rows in zip(runtime.layers, ema):
        layer.statistics.ema.copy_(torch.from_numpy(rows))
        layer.statistics.forwards.fill_(forwards)


def even_split(rows, layout, geometry) -> np.ndarray:
    return even_shares(np.asarray(layout)[None])[0]


# The runtime's own functions: a test that spies twice must not spy on its first spy.
REAL_PLAN, REAL_CHOOSE = rt.plan_layout, rt.choose_layers_to_install


def spy(monkeypatch, seen: dict) -> None:
    """Record what the planner, the share solver and the layer choice are given."""

    def plan(load, *args):
        seen.setdefault("load", []).append(np.array(load))
        return REAL_PLAN(load, *args)

    def solve(rows, layout, geometry):
        seen.setdefault("rows", []).append(rows)
        return even_split(rows, layout, geometry)

    def choose(gains, steps_left, copy_cost):
        seen.setdefault("gains", []).append(dict(gains))
        seen.setdefault("steps_left", []).append(steps_left)
        return REAL_CHOOSE(gains, steps_left, copy_cost)

    monkeypatch.setattr(rt, "plan_layout", plan)
    monkeypatch.setattr(rt, "solve_shares", solve)
    monkeypatch.setattr(rt, "choose_layers_to_install", choose)


def spied_ranks(monkeypatch, timed: bool = False):
    """Both ranks with in-process deciders after their first call, and ``seen``: what the planner
    and the share solver were given; ``timed``: a copy group was timed, every layer may install."""
    seen = {}
    clock = [0]
    spy(monkeypatch, seen)
    decisions = [FakeDecision(rank) for rank in range(R)]
    for decision in decisions if timed else ():
        decision.decider.copy_timings[0] = (1, 0.5)
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, 1)
    return seen, clock, decisions, ranks


def manager() -> AdaptiveMoeLoadBalancer:
    config = MoeAdaptiveLoadBalancerConfig(num_slots=R * LOCAL)
    config.setup(ep_rank=0, ep_size=R)
    return AdaptiveMoeLoadBalancer(ep_rank=0, ep_size=R, config=config, mapping=None)


def request(max_new_tokens, decoding_iter, accepted_draft_tokens, dummy=False):
    return SimpleNamespace(
        is_attention_dp_dummy=dummy,
        is_cuda_graph_dummy=False,
        py_max_new_tokens=max_new_tokens,
        py_decoding_iter=decoding_iter,
        py_num_accepted_draft_tokens=accepted_draft_tokens,
    )


def window_inputs(monkeypatch, forwards: int):
    seen, clock, decisions, ranks = spied_ranks(monkeypatch)
    ema = np.random.default_rng(7).random((R, L, E), dtype=np.float32) * 64
    progress = ([(120, 10, 1), (70, 10, 1), (1, 1, 0, True)], [(75, 15, 2)])
    for runtime, values in zip(ranks, ema):
        set_statistics(runtime, values, forwards)
    run(ranks, clock, rt.WINDOW_FORWARDS)
    # The window closed: the gather waits for the requests' progress, read in slices.
    assert all(runtime.wants_request_progress for runtime in ranks)
    for runtime, items in zip(ranks, progress):
        runtime.add_request_progress([request(*item) for item in items])
    run(ranks, clock, 2)
    return seen, ema, decisions, ranks


def test_full_window_gathers_the_rank_sum_and_scales_it(host_cuda, monkeypatch):
    seen, ema, decisions, ranks = window_inputs(monkeypatch, rt.WINDOW_FORWARDS)
    assert decisions[0].sent[0] == ("window", 0) and decisions[1].sent[0] == ("window", 0)
    total = np.zeros((L, E))
    for rank in ema:
        total += rank
    scale = (1 - EMA_DECAY) / (1 - EMA_DECAY**rt.WINDOW_FORWARDS)
    assert rt.window_scale(rt.WINDOW_FORWARDS) == scale
    assert len(seen["load"]) == R * L
    for i, (load, rows) in enumerate(zip(seen["load"], seen["rows"])):
        assert np.array_equal(load, total[i % L])
        assert np.array_equal(rows, total[i % L] * scale)
    first = [records(runtime)[0] for runtime in ranks]
    assert first[0]["open_call"] == first[1]["open_call"] == 0
    # 100 + 50 + 30 tokens remain over 3 requests (the dummy does not count) at 7 / 3 per step.
    assert seen["steps_left"] == [180 / (7 / 3) / 3] * R


def test_missing_boundary_moves_only_the_device_forward_count(host_cuda, monkeypatch):
    base, _, _, base_ranks = window_inputs(monkeypatch, rt.WINDOW_FORWARDS)
    late, ema, _, late_ranks = window_inputs(monkeypatch, rt.WINDOW_FORWARDS + 1)
    for key in ("open_call", "solved_call"):
        assert records(base_ranks[0])[0][key] == records(late_ranks[0])[0][key]
    assert base["steps_left"] == late["steps_left"]
    scale = rt.window_scale(rt.WINDOW_FORWARDS + 1)
    assert np.array_equal(late["rows"][0], (ema[0][0].astype(np.float64) + ema[1][0]) * scale)


def test_ranks_counting_different_forwards_are_scaled_each_by_its_own(host_cuda, monkeypatch):
    """A rank that recorded one forward more (the others ran a step it did not record) counts
    with its own window scale; the decision is made, not skipped."""
    seen, clock, _, ranks = spied_ranks(monkeypatch)
    ema = np.random.default_rng(3).random((R, L, E), dtype=np.float32) * 64
    for runtime, values, forwards in zip(ranks, ema, (8, 9)):
        set_statistics(runtime, values, forwards)
    run(ranks, clock, rt.WINDOW_FORWARDS + 4)
    assert len(seen["load"]) == R * L  # every layer planned on both ranks: not skipped
    ratio = rt.window_scale(9) / rt.window_scale(8)
    expected = ema[0][0].astype(np.float64) + ema[1][0].astype(np.float64) * ratio
    assert np.array_equal(seen["load"][0], expected)


def test_a_rank_without_a_forward_skips_the_window(host_cuda, monkeypatch):
    seen, clock, _, ranks = spied_ranks(monkeypatch)
    for runtime, forwards in zip(ranks, (8, 0)):
        set_statistics(runtime, np.ones((L, E), dtype=np.float32), forwards)
    run(ranks, clock, rt.WINDOW_FORWARDS + 4)
    assert "load" not in seen
    assert all(records(runtime)[0]["tables"] == [] for runtime in ranks)
    assert records(ranks[0])[0]["next_open"] == records(ranks[1])[0]["next_open"] == 12


def test_layers_not_live_on_every_rank_are_not_planned(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    decision = FakeDecision(0)
    stats = decision.stats.numpy()
    stats[:, : L * E] = 1.0
    stats[:, L * E : L * (E + 1)] = 8
    stats[:, L * (E + 1) : L * (E + 2)] = [[0, -1], [-1, -1]]
    decision.decider.committed[:] = [0, -1]
    decision.decider.decide(0)
    assert len(seen["load"]) == 1 and np.array_equal(seen["load"][0], stats[:, E : 2 * E].sum(0))


def test_copy_items_take_free_slots_from_the_node_first():
    geometry = ExpertGeometry(8, 12, 4)
    old = geometry.home_layout()
    old[11] = 0
    new = old.copy()
    new[[5, 8]] = 0
    assert rt.copy_items(old, new, geometry, 1, 2) == [(2, 0, 0)]
    assert rt.copy_items(old, new, geometry, 2, 2) == [(2, 3, 2)]
    assert rt.copy_items(old, new, geometry, 0, 2) == []
    new[0] = 3
    with pytest.raises(ValueError, match="not free"):
        rt.copy_items(old, new, geometry, 0, 2)


def script(rank, table_of_layer_1=2):
    def update(window):
        tables = {0: np.full(TABLE, 1, np.int32), 1: np.full(TABLE, table_of_layer_1, np.int32)}
        copies = {0: [(4, 0, 0)] if rank else []}
        result = rt.LayoutUpdate(window, [(0, [0])], copies, tables, [0])
        result.digest = result.fingerprint()
        return result

    return update


def test_tables_apply_from_the_call_after_every_rank_copied(host_cuda):
    """Rank 1's copies finish four calls later than rank 0's: both apply the new tables at the
    same call, the one after the latest completion, before the next window is scheduled."""
    clock = [0]
    decisions = [FakeDecision(0, script=script(0)), FakeDecision(1, script=script(1))]
    ranks = make_ranks(decisions, clock, copy_delays=(1, 5))
    run(ranks, clock, 22)
    first = [records(runtime)[0] for runtime in ranks]
    solved = first[0]["solved_call"]
    copied = [r["copied_call"] for r in first]
    assert first[1]["solved_call"] == solved and copied == [solved + 1, solved + 5]
    apply = copied[1] + 1
    assert [r["apply_call"] for r in first] == [apply, apply]
    assert [r["effect"] for r in first] == [{0: apply, 1: apply}] * 2
    for runtime in ranks:
        assert runtime.tables.tolist() == [[1] * TABLE, [2] * TABLE]
        assert runtime.debug_listener.kinds == [
            "window",
            "decision",
            "copied",
            "applied",
            "scheduled",
        ]
        assert records(runtime)[0]["next_open"] == apply + 4
        assert runtime.applied_window.tolist() == [0, 0]
    assert ranks[0].transport.batches == [[]] and ranks[1].transport.batches == [[(0, 4, 0, 0)]]


def test_the_window_event_hands_over_the_gathered_statistics(host_cuda):
    clock = [0]
    decisions = [FakeDecision(0, script=script(0)), FakeDecision(1, script=script(1))]
    ranks = make_ranks(decisions, clock)
    run(ranks, clock, rt.WINDOW_FORWARDS + 2)
    for runtime, decision in zip(ranks, decisions):
        (stats,) = runtime.debug_listener.stats
        assert np.array_equal(stats, decision.stats.numpy())


def test_ranks_that_decided_differently_stop_before_applying(host_cuda):
    clock = [0]
    decisions = [FakeDecision(0, script=script(0)), FakeDecision(1, script=script(1, 3))]
    ranks = make_ranks(decisions, clock)
    with pytest.raises(RuntimeError, match="decided window 0 differently"):
        run(ranks, clock, 22)
    assert all(not runtime.tables.any() for runtime in ranks)


def test_windows_keep_their_numbers_across_installs(host_cuda, monkeypatch):
    """A window gathered while an earlier update's tables are still on record is sent to the
    decision process under its own number."""
    _, clock, decisions, ranks = spied_ranks(monkeypatch, timed=True)
    hot = np.tile([400.0, 400.0] + [1.0] * (E - 2), (L, 1)).astype(np.float32)
    for _ in range(3):  # three windows: the second and third gather after an install
        for runtime in ranks:
            set_statistics(runtime, hot, rt.WINDOW_FORWARDS)
        run(ranks, clock, 22)
    sent = [m[1] for m in decisions[0].sent if m[0] == "window"]
    assert sent == list(range(len(sent))) and len(sent) >= 3
    assert [r["window"] for r in records(ranks[0])] == sent


def test_the_fingerprint_covers_tables_and_moves_but_not_this_ranks_copies():
    update = script(1)(0)
    same_decision_other_rank = script(0)(0)
    other_table = script(1, 3)(0)
    other_moves = script(1)(0)
    other_moves.moved = []
    assert update.fingerprint() == same_decision_other_rank.fingerprint()
    assert update.fingerprint() not in (other_table.fingerprint(), other_moves.fingerprint())
    assert float(update.fingerprint()) == update.fingerprint()  # exact in the float64 row


def test_two_ranks_move_an_expert_and_apply(host_cuda, monkeypatch):
    """Real deciders on both ranks: a window that replicates the two hot experts gives every rank
    its own copy items, the same digest, and new tables on both."""
    _, clock, _, ranks = spied_ranks(monkeypatch, timed=True)
    hot = np.tile([400.0, 400.0] + [1.0] * (E - 2), (L, 1)).astype(np.float32)
    for runtime in ranks:
        set_statistics(runtime, hot, rt.WINDOW_FORWARDS)
    run(ranks, clock, rt.WINDOW_FORWARDS)
    for runtime in ranks:
        runtime.add_request_progress([request(1000, 10, 0) for _ in range(4)])
    run(ranks, clock, 12)
    first = [records(runtime)[0] for runtime in ranks]
    assert all(record["moved"] == [0, 1] for record in first)
    assert first[0]["apply_call"] == first[1]["apply_call"]
    # Experts 0 and 1 are rank 0's homes: rank 1 copies them in one batch, rank 0 copies nothing.
    assert ranks[0].transport.batches == [[]]
    (batch,) = ranks[1].transport.batches
    assert sorted(batch) == [
        (layer, slot, 0, expert) for layer in range(L) for slot, expert in ((4, 0), (5, 1))
    ]
    for runtime in ranks:
        assert sorted(records(runtime)[0]["effect"]) == [0, 1] and runtime.tables.any()
        assert runtime.applied_window.tolist() == [0, 0]


def test_no_boundary_runs_while_a_graph_is_captured(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    balancer = manager()
    calls = []
    balancer.runtime = SimpleNamespace(boundary=lambda: calls.append(1))
    balancer.enable_update_weights = True
    balancer.start_iter()
    balancer.end_iter()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    balancer.start_iter()
    assert calls == [1]


def window_stats(stats: np.ndarray, hot=(0, 1)) -> np.ndarray:
    """Every rank's window: its ``hot`` experts take 400 rows per layer, the others 1, over 8
    forwards; no layer is live yet; 1e5 tokens remain over 4 requests at 2 accepted per step."""
    stats[:, : L * E] = np.tile(np.where(np.isin(np.arange(E), hot), 400.0, 1.0), L)
    stats[:, L * E : L * (E + 1)] = 8
    stats[:, L * (E + 1) : L * (E + 2)] = -1
    totals = L * (E + 2) + rt.FIELDS.index("remaining")
    stats[:, totals : totals + 3] = [1e5, 4, 8]  # remaining, requests, accepted
    return stats


def test_an_update_moves_in_one_copy_group(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    decision = FakeDecision(0)
    window_stats(decision.stats.numpy())
    decision.decider.copy_timings[0] = (1, 0.5)  # one copy group timed: both layers may install
    update = decision.decider.decide(0)
    assert update.moved == [0, 1] and update.groups == [(0, [0, 1])]
    assert decision.decider.decide(1).groups == []  # nothing new to move: no group


def test_a_candidate_must_gain_on_the_previous_window_too(monkeypatch):
    seen = {}
    spy(monkeypatch, seen)
    candidate = np.where(np.arange(R * LOCAL) == LOCAL + 4, 2, HOME)  # expert 2 copied to GPU 1
    monkeypatch.setattr(rt, "plan_layout", lambda *args: SimpleNamespace(layout=candidate))
    decision = FakeDecision(0)
    decision.decider.copy_timings[0] = (1, 0.5)

    def window(number, hot):
        window_stats(decision.stats.numpy(), (hot,))
        return decision.decider.decide(number)

    assert window(0, 6).tables == {}  # GPU 1 is the busiest: the copy would load it further
    held = window(1, 2)  # gains on this window, loses on the previous one: held out
    assert held.tables == {} and max(seen["gains"][-1].values()) < 0
    assert sorted(window(2, 2).tables) == [0, 1]  # gains on both windows: installed


def test_decision_process_answers_and_reports_its_failure():
    process = rt.DecisionProcess(rt.DecisionConfig(GEOMETRY, MODEL, np.tile(HOME, (L, 1)), 1, 1))
    try:
        window_stats(process.stats.numpy())
        process.submit(("window", 0))
        kind, update = process.recv()
        assert kind == "update" and update.window == 0
        # No copy was timed yet: the best layer alone is installed, to time one.
        assert update.moved == [0] and set(update.tables) == {0}
        assert all(t.dtype == np.int32 and t.size == TABLE for t in update.tables.values())
        rows = np.array([[10, 2, 0.5, 1], [12, 2, 0.7, 1]])  # ROW of both ranks
        process.submit(("copied", 0, rows))
        kind, window, open_call, apply_call, plan = process.recv()
        assert (kind, window, apply_call, open_call) == ("schedule", 1, 13, 13 + plan.open_after)
        process.submit(("copied", 5, rows))
        with pytest.raises(RuntimeError, match="KeyError"):
            process.recv()
    finally:
        process.close()


def test_shutdown_agrees_over_the_ep_group_then_stops_the_process_and_the_copies():
    calls = []
    for peer_safe, done in ((False, False), (True, True)):
        balancer = manager()
        balancer.runtime = SimpleNamespace(close=lambda: calls.append("worker"))
        balancer.copy_stream = SimpleNamespace(synchronize=lambda: calls.append("stream"))
        balancer.transport = SimpleNamespace(allgather=lambda safe, peer=peer_safe: [safe, peer])
        assert balancer.shutdown() is done and balancer.is_shutdown is done
    assert calls == ["worker", "stream"]
