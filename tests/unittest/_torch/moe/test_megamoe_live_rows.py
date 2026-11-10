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
"""CPU contract tests: the MegaMoE launch reads the forward's live-row word end to end.

The engine writes the word before each forward (decoder runner); the MegaMoE backend
hands it to the op, the op to every kernel that reads it, and the kernel to its router,
which routes only the live rows, and to its top-k reduce, which writes zeros for the
others. Each link is checked in the source, so the chain cannot silently break.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_TORCH = _ROOT / "tensorrt_llm/_torch"
_MEGAMOE = _TORCH / "moe/kernels/cutedsl_megamoe"
_TOKEN_COMM = _MEGAMOE / "communication/nvlink_domain/token_comm.py"
_TOPK_REDUCE = _MEGAMOE / "helpers/topk_reduce.py"
_DEVICE_HELPER = _MEGAMOE / "helpers/live_rows.py"
_BLACKWELL = (
    _MEGAMOE / "kernel_src/blackwell/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py"
)
_OP = _TORCH / "moe/custom_ops/cute_dsl_megamoe_custom_op.py"
_BACKEND = _TORCH / "moe/fused_moe/mega_moe/mega_moe_cute_dsl.py"
_RUNNER = _TORCH / "pyexecutor/engine/runners/decoder/runner.py"
_HOST = _TORCH / "moe/live_rows.py"
_WORD = "live_row_word"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _functions(tree: ast.AST, name: str):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def _function(tree: ast.AST, name: str, owner: str = None) -> ast.FunctionDef:
    scope = tree
    if owner is not None:
        (scope,) = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == owner]
    found = _functions(scope, name)
    assert len(found) == 1, f"{name}: {len(found)} definitions"
    return found[0]


def _calls(node: ast.AST, callee: str):
    """Calls whose function is ``callee`` or ends with ``.callee``."""
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == callee)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == callee)
        )
    ]


def _passes(call: ast.Call, name: str) -> bool:
    """Whether ``call`` passes the variable ``name``, positionally or as a keyword."""
    values = list(call.args) + [keyword.value for keyword in call.keywords]
    return any(isinstance(v, ast.Name) and v.id == name for v in values) or any(
        keyword.arg == name for keyword in call.keywords
    )


def _params(function: ast.FunctionDef):
    return [arg.arg for arg in function.args.args + function.args.kwonlyargs]


def test_router_bounds_its_loads_and_routes_by_the_word():
    tree = _parse(_TOKEN_COMM)
    load = _function(tree, "_load_router_inputs", owner="_MetadataPushRouter")
    (bound,) = [
        node
        for node in ast.walk(load)
        if isinstance(node, ast.If) and _WORD in ast.unparse(node.test)
    ]
    (count,) = _calls(bound, "live_row_count")
    assert [ast.unparse(a) for a in count.args] == [
        "self._router_live_row_word",
        "self._router_token_rows",
    ]
    assigned = {t.id for n in bound.body if isinstance(n, ast.Assign) for t in n.targets}
    # The load guard and the trash-bucket guard of each element both use the bound.
    assert assigned == {"actual_token_count", "actual_elements"}

    kernel = _function(tree, "_router_kernel", owner="_MetadataPushRouter")
    assert {_WORD, "token_rows"} <= set(_params(kernel))
    stored = {
        (ast.unparse(n.targets[0]), ast.unparse(n.value))
        for n in ast.walk(kernel)
        if isinstance(n, ast.Assign)
    }
    assert ("self._router_live_row_word", _WORD) in stored
    assert ("self._router_token_rows", "token_rows") in stored
    launch = _function(tree, "launch_router", owner="_MetadataPushRouter")
    (call,) = _calls(launch, "_router_kernel")
    assert _passes(call, _WORD) and _passes(call, "token_rows")
    outer = _function(tree, "launch_router", owner="TokenCommNonDeterministic")
    (forward,) = _calls(outer, "launch_router")
    assert _passes(forward, _WORD) and _passes(forward, "token_rows")


@pytest.mark.parametrize("kernel_name", ["_reduce_bf16", "_reduce_mxfp8", "_reduce_fp4"])
def test_topk_reduce_zeroes_the_rows_past_the_live_ones(kernel_name):
    tree = _parse(_TOPK_REDUCE)
    kernel = _function(tree, kernel_name, owner="TopkReduce")
    assert _WORD in _params(kernel)
    (count,) = _calls(kernel, "live_row_count")
    assert ast.unparse(count.args[0]) == _WORD
    rows = [
        n
        for n in kernel.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "token_idx < live_rows"
    ]
    assert len(rows) == 1, "the reduce must run only for the live rows"
    (dropped,) = rows[0].orelse
    assert isinstance(dropped, ast.If) and _calls(dropped, "_store_zero_slice")

    launcher = _function(tree, "__call__", owner="TopkReduce")
    assert _WORD in _params(launcher)
    (launch,) = _calls(launcher, kernel_name)
    assert _passes(launch, _WORD)


def test_blackwell_kernel_hands_the_word_to_router_and_reduce():
    tree = _parse(_BLACKWELL)
    call = _function(tree, "__call__", owner="BlockScaledSwapAbMegaMoeKernel")
    assert _params(call)[-1] == _WORD, "trailing optional, after the READY pair"
    (router,) = _calls(call, "launch_router")
    assert _passes(router, _WORD)
    assert ast.unparse(router.args[-1]) == "Int32(output_activation.shape[0])"
    (reduce,) = _calls(call, "_topk_reduce")
    assert _passes(reduce, _WORD)
    reads = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "reads_live_row_word"
    ]
    assert len(reads) == 1 and ast.unparse(reads[0].value) == "True"


def test_op_and_backend_pass_the_word_of_the_running_forward():
    op = _parse(_OP)
    fakes = [
        f
        for f in _functions(op, "_")
        if any(
            "trtllm::cute_dsl_megamoe_nvfp4_blackwell" in ast.unparse(d) for d in f.decorator_list
        )
    ]
    entries = _functions(op, "run_cute_dsl_megamoe_nvfp4_blackwell") + fakes
    assert len(entries) == 2
    for entry in entries:
        assert _params(entry)[-1] == _WORD
    forward = _function(op, "forward", owner="Sm100MegaMoENvfp4Runner")
    guards = [
        n
        for n in ast.walk(forward)
        if isinstance(n, ast.If) and "reads_live_row_word" in ast.unparse(n.test)
    ]
    assert len(guards) == 1
    assert 'runtime_kwargs["live_row_word"]' in ast.unparse(guards[0]).replace("'", '"')

    backend = _parse(_BACKEND)
    launch = _function(backend, "_launch_megamoe_kernel")
    (megamoe,) = _calls(launch, "megamoe_op")
    (word,) = [k for k in megamoe.keywords if k.arg == _WORD]
    assert ast.unparse(word.value).startswith("current_word(")


def test_engine_writes_the_word_before_running_the_forward():
    runner = _parse(_RUNNER)
    forward = _function(runner, "_forward_decoder", owner="DecoderRunner")
    order = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("self._live_rows.write", "self._graphs.run")
    ]
    assert [ast.unparse(n.func) for n in sorted(order, key=lambda n: (n.lineno, n.col_offset))] == [
        "self._live_rows.write",
        "self._graphs.run",
    ]


def test_host_and_device_agree_on_the_word_layout():
    def constant(path, name):
        (value,) = [
            n.value
            for n in _parse(path).body
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name
        ]
        return ast.literal_eval(value)

    assert constant(_DEVICE_HELPER, "LIVE_ROW_REQUEST_BITS") == constant(_HOST, "REQUEST_BITS")
