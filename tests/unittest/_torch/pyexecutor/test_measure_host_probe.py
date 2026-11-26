# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the measurement-only main-thread probes."""

import json
import sys

import pytest

from tensorrt_llm._torch.pyexecutor.measure_host_probe import (
    INJECT_EVERY,
    INJECT_US,
    OUT_DIR,
    STAGE_STEPS,
    SWITCH_INTERVAL_S,
    HostProbe,
    StageSteps,
    dose_of,
    inject_boundary,
    spin,
)

pytestmark = pytest.mark.cpu_only


class Runtime:
    """The part of AdaptiveRuntime the dose patch touches: _boundary advances ``call``."""

    def __init__(self):
        self.call = 0

    def _boundary(self):
        self.call += 1


def test_unset_probe_does_nothing():
    probe = HostProbe.from_env({})
    assert probe.every == 0 and probe.steps is None and probe.switch_interval_s is None
    probe.install(Runtime)  # no doses: the class keeps its own _boundary
    assert not hasattr(Runtime._boundary, "measure_host_probe")
    assert probe.dump(0) is None


def test_doses_land_on_every_fourth_call_in_turn():
    doses = {c: dose_of(c, 4, (250, 250, 4000)) for c in range(25)}
    expected = {4: 250, 8: 250, 12: 4000, 16: 250, 20: 250, 24: 4000}
    assert {c: us for c, us in doses.items() if us} == expected


def test_patched_boundary_spins_its_dose_after_its_work():
    cls = type("R", (Runtime,), {})
    slept = []
    inject_boundary(cls, 4, (250, 4000), sleep=slept.append)
    rt = cls()
    for _ in range(16):
        rt._boundary()
    assert rt.call == 16  # the original work still runs every call
    assert slept == [250_000, 4_000_000, 250_000, 4_000_000]


def test_spin_holds_the_thread_for_its_time():
    now = [0]

    def clock():
        now[0] += 100
        return now[0]

    spin(1000, clock)
    assert now[0] >= 1100


def test_doses_need_both_settings():
    with pytest.raises(ValueError):
        HostProbe.from_env({INJECT_US: "250"})
    with pytest.raises(ValueError):
        HostProbe.from_env({INJECT_EVERY: "4"})
    with pytest.raises(ValueError):
        HostProbe.from_env({INJECT_US: "0", INJECT_EVERY: "4"})
    probe = HostProbe.from_env({INJECT_US: "250,250,4000", INJECT_EVERY: "4"})
    assert (probe.every, probe.doses_us) == (4, (250, 250, 4000))


def test_stage_steps_need_the_readout_directory():
    with pytest.raises(ValueError):
        HostProbe.from_env({STAGE_STEPS: "1"})


def test_stage_steps_keep_each_pass_and_stage(tmp_path):
    ticks = iter(range(0, 10_000, 10))
    steps = StageSteps(clock=lambda: next(ticks), cpu=lambda: 0)
    steps.mark("schedule")  # before any pass: nothing to attribute
    for _ in range(2):
        steps.begin()
        steps.mark("schedule")
        steps.mark("forward")
    doc = steps.doc()
    assert doc["stages"] == ["schedule", "forward"]
    assert doc["pass_index"] == [0, 0, 1, 1] and doc["stage"] == [0, 1, 0, 1]
    assert doc["wall_ns"] == [10, 10, 10, 10] and len(doc["pass_start_ns"]) == 2

    probe = HostProbe.from_env({STAGE_STEPS: "1", OUT_DIR: str(tmp_path)})
    probe.steps.begin()
    probe.steps.mark("forward")
    path = probe.dump(3)
    assert path == str(tmp_path / "rank3.stage-steps.json") and probe.dump(3) == path
    written = json.loads((tmp_path / "rank3.stage-steps.json").read_text())
    assert written["rank"] == 3 and written["stages"] == ["forward"]


def test_switch_interval_is_set_for_the_process():
    before = sys.getswitchinterval()
    try:
        HostProbe.from_env({SWITCH_INTERVAL_S: "0.0005"}).install(Runtime)
        assert sys.getswitchinterval() == pytest.approx(0.0005)
    finally:
        sys.setswitchinterval(before)


def test_unset_probe_leaves_the_executor_unchanged():
    # Every probe variable unset: no stage steps (the executor's stage timing stays as
    # TLLM_EXECUTOR_STAGE_TIMING sets it), the boundary unpatched, the switch interval untouched
    # and nothing written at the loop's end.
    probe = HostProbe.from_env({})
    before, original = sys.getswitchinterval(), Runtime._boundary
    probe.install(Runtime)
    assert probe.steps is None and Runtime._boundary is original
    assert sys.getswitchinterval() == before and probe.dump(0) is None
