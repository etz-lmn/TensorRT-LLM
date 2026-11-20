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
"""CPU checks of the MegaMoE token-tile return-ready contract on the Blackwell and Rubin kernels.

token_tile counts FC2 completion per (token block, tile) instead of per expert, so token-back may return a
token tile before the expert's other tiles finish. The checks execute the vendored kernels' own validation
statements, slot expressions and geometry helper from their sources, without importing CUDA: the legal
configurations, that the FC2 epilogue publishes into the slot the token-back warps wait on, and that the
built components agree on the completion target. Native compilation and memory ordering are GPU checks.
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_ROOT = Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe"
_MEGA = "kernel_src/{}/inference/mega/block_scaled_swap_ab_mega_moe_kernel.py"
_EPI = "kernel_src/{}/inference/mega/block_scaled_swap_ab_fc12_epilogue.py"
_COMM = "communication/nvlink_domain/token_comm.py"
# The EP kernels that accept token_tile; both share the token-comm consumer.
_ARCHES = ["blackwell", "rubin"]
_KEY = "token_back_ready_granularity"


def _method(path: str, owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((_ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _execute(nodes: list, **values) -> dict:
    nodes = copy.deepcopy(nodes)
    for node in nodes:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    namespace = {"Int32": int, "cutlass": NS(const_expr=bool), **values}
    module = ast.fix_missing_locations(ast.Module(body=future + nodes, type_ignores=[]))
    exec(compile(module, "<vendored MegaMoE source>", "exec"), namespace)  # nosec B102
    return namespace


def _value(node: ast.expr, **values):
    return eval(  # nosec B307
        compile(ast.Expression(node), "<vendored MegaMoE expression>", "eval"),
        {"Int32": int, **values},
    )


def _assignment(method: ast.FunctionDef, target: str) -> ast.Assign:
    return next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Assign) and any(ast.unparse(t) == target for t in n.targets)
    )


@pytest.fixture(scope="module")
def fc2_ready_geometry():
    tree = ast.parse((_ROOT / "helpers/fc2_ready_geometry.py").read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    return _execute(nodes)["derive_fc2_ready_geometry"]


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("warp", ["standalone_warps", "reuse_dispatch_warps"])
@pytest.mark.parametrize("schedule", ["grouped", "phase_interleave"])
@pytest.mark.parametrize(
    "damage", [None, "mma", "bulk", "counter", "quant", "warp", "reduce", "internode"]
)
def test_token_tile_legal_combinations(arch, warp, schedule, damage):
    if damage == "internode" and arch != "blackwell":
        pytest.skip("only the Blackwell kernel has the internode component")
    config = NS(
        token_back_ready_granularity="token_tile",
        schedule_mode=schedule,
        work_id_mode="atomic_counter",
        token_back_mode=warp,
        token_back_schedule_mode="atomic_counter",
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        fallback_cluster_shape_mn=None,
        fc2_use_bulk=True,
        quant_kind="nvfp4",
        reduce_topk_in_kernel=False,
        internode=False,
    )
    changes = {
        "mma": ("use_2cta_instrs", False),
        "bulk": ("fc2_use_bulk", False),
        "counter": ("token_back_schedule_mode", "static"),
        "quant": ("quant_kind", "mxfp8"),
        "warp": ("token_back_mode", "epi_warps"),
        "reduce": ("reduce_topk_in_kernel", True),
        "internode": ("internode", True),
    }
    if damage:
        setattr(config, *changes[damage])
    nodes = []
    for path, owner, method in (
        (_MEGA.format(arch), "BlockScaledSwapAbMegaMoeKernel", "_validate_geometry"),
        (_COMM, "TokenCommNonDeterministic", "_validate_configuration"),
    ):
        nodes += [
            n
            for n in _method(path, owner, method).body
            if isinstance(n, ast.If) and _KEY in ast.unparse(n.test)
        ]
    assert nodes
    if damage:
        with pytest.raises(ValueError):
            _execute(nodes, self=config, QuantKind=NS(nvfp4="nvfp4"))
    else:
        _execute(nodes, self=config, QuantKind=NS(nvfp4="nvfp4"))


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("tile", [64, 256])
def test_token_tile_producer_consumer_share_prefix_including_empty_experts(arch, tile):
    consumer = _method(_COMM, "TokenCommNonDeterministic", "token_back")
    ready = _assignment(consumer, "ready_slot").value
    outer = next(
        n
        for n in ast.walk(consumer)
        if isinstance(n, ast.While) and ast.unparse(n.test).startswith("local_expert <")
    )
    advance = next(
        n.body[0].value
        for n in outer.body
        if isinstance(n, ast.If)
        and _KEY in ast.unparse(n.test)
        and isinstance(n.body[0], ast.Assign)
    )
    producer = _assignment(
        _method(_EPI.format(arch), "SwapABFc2Epilogue", "signal_fc2_done"), "slot"
    ).value
    config, prefix, counts = NS(tokens_per_fc1_ready_slot=tile), 0, [0, 1, tile, 0, tile + 1, 3]
    for count in counts:
        expected = prefix
        for row in {0, count - 1, tile} & set(range(count)):
            index = expected + row // tile
            assert (
                _value(ready, self=config, expert_ready_slot_begin=prefix, token_in_expert=row)
                == index
            )
            tile_info = NS(cumulative_token_block_count=expected, tile_n_idx=row // tile)
            assert _value(producer, work_tile_info=tile_info) == index
        prefix = _value(
            advance, self=config, expert_ready_slot_begin=prefix, expert_token_count=count
        )
        assert prefix == expected + (count + tile - 1) // tile
    assert prefix == sum((n + tile - 1) // tile for n in counts)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("granularity", ["expert", "token_tile"])
@pytest.mark.parametrize("hidden,target", [(7168, 56), (7296, 60)])
def test_fc2_geometry_uses_token_tile_helper_only_when_enabled(
    fc2_ready_geometry, arch, granularity, hidden, target
):
    init = _method(_MEGA.format(arch), "BlockScaledSwapAbMegaMoeKernel", "__init__")
    geometry = next(
        node
        for node in init.body
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "derive_fc2_ready_geometry"
            for call in ast.walk(node)
        )
    )
    config = NS(
        token_back_ready_granularity=granularity,
        hidden_size=hidden,
        mma_tiler_mnk=(256, 128, 256),
        mma_instruction_mnk=(256, 128, 64),
        cluster_shape_mn=(4, 1),
        use_2cta_instrs=True,
        fallback_cluster_shape_mn=(2, 1),
    )
    calls = []

    def resolve(**kwargs):
        assert granularity == "token_tile", "the expert path must not call the token-tile helper"
        calls.append(kwargs)
        return fc2_ready_geometry(**kwargs)

    result = _execute(
        [geometry],
        self=config,
        mma_cta_count=2,
        ceil_div=lambda x, y: (x + y - 1) // y,
        derive_fc2_ready_geometry=resolve,
    )
    assert (
        result["tokens_per_fc1_ready_slot"],
        result["fc2_done_signals_per_token_tile"],
    ) == (128, target)
    assert len(calls) == (1 if granularity == "token_tile" else 0)
    if calls:
        assert calls[0]["fallback_cluster_shape_mn"] == (2, 1)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "damage", [None, "active_fallback_target", "slot_width", "mapper", "store"]
)
def test_token_tile_checks_constructed_producer_consumer_contract(arch, damage):
    validate = _execute(
        [
            _method(
                _MEGA.format(arch),
                "BlockScaledSwapAbMegaMoeKernel",
                "_validate_token_tile_ready_contract",
            )
        ],
        ceil_div=lambda x, y: (x + y - 1) // y,
    )["_validate_token_tile_ready_contract"]
    config = NS(
        hidden_size=7296,
        cluster_shape_mn=(4, 1),
        scheduler=NS(mapping_cta_tile_shape_mnk=(128, 128, 256), mapping_cluster_shape_mn=(1, 4)),
        epilogue=NS(fc2_use_tma=True, cta_tile_n=128, cta_tile_m=128),
        token_comm=NS(tokens_per_fc1_ready_slot=128, fc2_done_signals_per_token_tile=60),
    )
    if damage == "active_fallback_target":
        # The target of a preferred (4, 1) cluster counted on its (2, 1) fallback instead.
        config.token_comm.fc2_done_signals_per_token_tile = 58
    elif damage == "slot_width":
        config.token_comm.tokens_per_fc1_ready_slot = 256
    elif damage == "mapper":
        config.scheduler.mapping_cluster_shape_mn = (1, 2)
    elif damage == "store":
        config.epilogue.fc2_use_tma = False
    if damage:
        with pytest.raises(ValueError):
            validate(config)
    else:
        validate(config)


def test_blackwell_kernel_name_tags_token_tile_and_validates_after_the_shared_expert():
    """The token-tile build has a name of its own (its compiled-kernel cache entry), and the contract is
    checked on the finished components: after the fused shared expert's plan, which may claim FC2 tiles."""
    init = _method(_MEGA.format("blackwell"), "BlockScaledSwapAbMegaMoeKernel", "__init__")
    source = ast.unparse(init)
    assert source.index("self._init_shared_expert()") < source.index(
        "self._validate_token_tile_ready_contract()"
    )
    name = ast.unparse(_method(_MEGA.format("blackwell"), "BlockScaledSwapAbMegaMoeKernel", "name"))
    assert "_fc2readytile_v1" in name
