# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from test_adaptive_eplb_runtime import FORWARDS, HOME, FakeDecision, even_split, rt, window_stats


def test_the_observe_switch_decides_and_installs_nothing(monkeypatch):
    monkeypatch.setattr(rt, "solve_shares", even_split)
    monkeypatch.setenv(rt.OBSERVE_ENV, "1")
    decider = FakeDecision(0, copy_ms=0.5).decider
    decider.next_forwards = FORWARDS  # an earlier window measured the rows per forward
    window_stats(decider.stats)  # GPU 0's experts 0 and 1 are hot: replicas would pay
    assert decider.decide(0).tables == {}
    assert (decider.layout == HOME).all()
