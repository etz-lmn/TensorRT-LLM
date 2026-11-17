# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source contracts of the adaptive mode of ``DSv4RouteNvfp4QuantKernel`` (CPU, no GPU, no import of the kernel).

Memory order, one check per write -> read path:
  [MO-1] the launch's staged slots, weights and NVFP4 rows (generic stores) are read by later kernels (the router's
         and MegaMoE's TMA, async proxy): ordered by this grid's completion, which their griddepcontrol.wait or stream
         order waits for, as in the upstream fused launch; so the kernel must not let dependents launch early and
         must wait for its own predecessor before touching global memory.
  [MO-2] a row's picks and marks -> the block-completing CTA: every lane fences (acq_rel, gpu) after its stores, the
         warp synchronizes, lane 0 arrives with an acq_rel atomic.
  [MO-3] the completing CTA: lane 0's acq_rel arrival, then a CTA barrier and a gpu fence in every thread before any
         thread loads the other rows' picks.
The slot pick constants must equal the route op's (adaptiveEplbKernels.cu, adaptive_eplb_reference.py).
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / "tensorrt_llm/_torch/moe/kernels/blackwell/dsv4_route_quant.py"
ROUTE_OP = ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/adaptiveEplbKernels.cu"
REFERENCE = ROOT / "tests/unittest/_torch/moe/adaptive_eplb_reference.py"


def _method(name):
    source = KERNEL.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{name} not found in {KERNEL.name}")


def _calls(segment):
    """The ``cute.arch.*`` and ``self.*`` calls of a method in source order, with their keyword sem."""
    out = []
    for node in ast.walk(ast.parse(segment.replace("\n    ", "\n"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            sem = next(
                (
                    k.value.value
                    for k in node.keywords
                    if k.arg == "sem" and isinstance(k.value, ast.Constant)
                ),
                None,
            )
            out.append((node.lineno, node.col_offset, node.func.attr, sem))
    return [c[2:] for c in sorted(out)]


def test_mo1_dependents_wait_for_the_whole_grid():
    source = KERNEL.read_text()
    assert "launch_dependents" not in source, (
        "[MO-1] staging must be complete before a dependent may start"
    )
    kernel = _method("kernel")
    wait = kernel.index("griddepcontrol_wait()")
    for first_global in (
        "self.route(",
        "self.route_warp(",
        "self.adaptive_map(",
        "self.clear_tail(",
        "self.quantize(",
    ):
        assert kernel.index(first_global) > wait, (
            f"[MO-1] {first_global} before griddepcontrol_wait"
        )


def test_mo2_a_row_releases_its_picks_before_it_arrives():
    calls = _calls(_method("adaptive_arrive"))
    names = [c[0] for c in calls]
    fence, sync = names.index("fence_acq_rel_gpu"), names.index("sync_warp")
    arrivals = [i for i, c in enumerate(calls) if c[0] == "atomic_add"]
    assert arrivals and fence < sync < arrivals[0], "[MO-2] fence, warp sync, then the arrival"
    assert all(calls[i][1] == "acq_rel" for i in arrivals), (
        "[MO-2] the arrival is an acq_rel atomic"
    )
    pick = _calls(_method("adaptive_pick"))
    assert ("atomic_add", "relaxed") in pick, "the block mark precedes the row's release"
    store_picks = _method("store_picks")
    assert store_picks.index("self.adaptive_pick(") < store_picks.index("self.adaptive_arrive("), (
        "picks before arrival"
    )


def test_mo3_the_completing_cta_acquires_before_it_reads():
    segment = _method("adaptive_map")
    barrier, fence, mapping = (
        segment.index(x) for x in ("cute.arch.barrier()", "fence_acq_rel_gpu()", "self.map_block(")
    )
    assert barrier < fence < mapping, (
        "[MO-3] CTA barrier, gpu fence, then the loads of other rows' picks"
    )
    reset = segment.index("state[block] = Int32(0)")
    assert segment.rindex("cute.arch.barrier()", 0, reset) > mapping, (
        "the counts reset only after the mapping"
    )


def test_the_kernel_states_each_memory_order_point():
    source = KERNEL.read_text()
    for tag in ("[MO-2]", "[MO-3]"):
        assert tag in source, tag


@pytest.mark.parametrize(
    "name,cpp,ref",
    [
        ("ADAPTIVE_SPLIT_BLOCK", "kSplitBlock", "SPLIT_BLOCK"),
        ("ADAPTIVE_BLOCK_MULTIPLIER", "kBlockMultiplier", "BLOCK_MULTIPLIER"),
        ("ADAPTIVE_EXPERT_MULTIPLIER", "kExpertMultiplier", "EXPERT_MULTIPLIER"),
        ("ADAPTIVE_RANK_MULTIPLIER", "kRankMultiplier", "RANK_MULTIPLIER"),
    ],
)
def test_the_slot_pick_constants_are_the_route_ops(name, cpp, ref):
    kernel = int(re.search(rf"^{name} = (\S+)$", KERNEL.read_text(), re.M).group(1), 0)
    op = int(
        re.search(rf"constexpr \w+ {cpp} = (0x[0-9A-Fa-f]+|\d+)u?;", ROUTE_OP.read_text()).group(1),
        0,
    )
    reference = int(re.search(rf"^{ref} = (\S+)$", REFERENCE.read_text(), re.M).group(1), 0)
    assert kernel == op == reference
