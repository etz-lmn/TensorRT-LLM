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
"""CPU contract tests: the load balancer's local statistic counts only the forward's live rows.

The statistic of a rank's own routes reads the live-row word the engine writes before each
forward (tensorrt_llm/_torch/moe/live_rows.py) and counts the elements of the live rows
only; the statistic of routes gathered from every rank counts every row, since one rank's
word does not describe the others' rows.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_KERNELS = _ROOT / "cpp/tensorrt_llm/kernels/moe/loadBalance/moeLoadBalanceKernels.cu"
_OP = _ROOT / "cpp/tensorrt_llm/thop/moe/loadBalance/moeLoadBalanceOp.cpp"
_FAKES = _ROOT / "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py"
_BALANCER = _ROOT / "tensorrt_llm/_torch/moe/fused_moe/moe_load_balancer.py"
_HOST = _ROOT / "tensorrt_llm/_torch/moe/live_rows.py"
_MEGAMOE = _ROOT / "tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"


def _body(source: str, signature: str) -> str:
    """The brace-balanced body of the first C++ function whose text starts with ``signature``."""
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for position in range(open_brace, len(source)):
        depth += {"{": 1, "}": -1}.get(source[position], 0)
        if depth == 0:
            return source[open_brace : position + 1]
    raise AssertionError(f"unbalanced body of {signature}")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_statistic_kernel_counts_the_live_rows():
    source = _KERNELS.read_text()
    kernel = _flat(_body(source, "__global__ void statisticKernel("))
    signature = _flat(source[source.index("__global__ void statisticKernel(") :].split("{", 1)[0])
    assert "int const* liveRowWord" in signature
    assert "liveRowCount(*liveRowWord, totalEltCount / metaInfo.topK) * metaInfo.topK" in kernel
    assert "idx < liveEltCount" in kernel and "idx < totalEltCount" not in kernel
    # The word is read after the grid dependency wait, like every other input.
    assert kernel.index("cudaGridDependencySynchronize") < kernel.index("*liveRowWord")

    local = _flat(_body(source, "void moeHierarchicalStatisticLocalDevice("))
    assert re.search(r"statisticKernel,.*localRawExpertIds, liveRowWord\)", local)
    gathered = _flat(_body(source, "void moeStatisticDevice("))
    assert re.search(
        r"statisticKernel,.*gatheredRawExpertIds, static_cast<int const\*>\(nullptr\)\)", gathered
    )


def test_device_and_host_agree_on_the_word_layout():
    helper = _flat(_body(_KERNELS.read_text(), "__device__ __forceinline__ int liveRowCount("))
    (bits,) = [
        ast.literal_eval(n.value)
        for n in ast.parse(_HOST.read_text()).body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "REQUEST_BITS"
    ]
    mask = f"0x{(1 << bits) - 1:X}"
    assert f"int const real = word & {mask};" in helper
    assert f"int const requests = (word >> {bits}) & {mask};" in helper


def test_op_and_balancer_pass_the_word():
    op = _flat(_OP.read_text())
    assert "int is_last_stage, Tensor? live_row_word=None) -> ()" in op
    assert "localRawExpertIds.data_ptr<int>(), liveRowWordPtr, stream)" in op

    (fake,) = [
        f
        for f in ast.walk(ast.parse(_FAKES.read_text()))
        if isinstance(f, ast.FunctionDef)
        and any(
            "moe_hierarchical_statistic_local_device" in ast.unparse(d) for d in f.decorator_list
        )
    ]
    assert fake.args.args[-1].arg == "live_row_word"

    balancer = ast.parse(_BALANCER.read_text())
    (update,) = [
        f
        for f in ast.walk(balancer)
        if isinstance(f, ast.FunctionDef) and f.name == "update_local_statistic"
    ]
    calls = [
        n
        for n in ast.walk(update)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "torch.ops.trtllm.moe_hierarchical_statistic_local_device"
    ]
    assert len(calls) == 2
    assert all(ast.unparse(c.args[-1]) == "live_row_word" for c in calls)
    (word,) = [
        n
        for n in ast.walk(update)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "live_row_word"
    ]
    assert (
        ast.unparse(word.value)
        == "current_word(local_raw_expert_ids.device) if self.counts_live_rows else None"
    )


def test_megamoe_marks_its_balancer_by_what_its_kernels_read():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import megamoe_kernel_classes

    reads = {
        sm: [getattr(kernel, "reads_live_row_word", False) for kernel in megamoe_kernel_classes(sm)]
        for sm in (100, 103, 107)
    }
    assert reads == {100: [True], 103: [True], 107: [False, False]}
    marks = [
        node
        for node in ast.walk(ast.parse(_MEGAMOE.read_text()))
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["self.layer_load_balancer.counts_live_rows"]
    ]
    assert len(marks) == 1
    assert "megamoe_kernel_classes(get_sm_version())" in ast.unparse(marks[0].value)
