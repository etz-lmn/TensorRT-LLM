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
"""CPU contract tests for the proxy ordering of the MegaMoE NVLink token communication.

Bulk copies and TMA loads access memory through the async proxy; ordinary loads, stores
and the flags the CTAs and ranks synchronize on use the generic proxy. Data that one
proxy wrote and the other reads needs a proxy fence in between:

- Publication: each router CTA writes its per-expert counts to shared memory with
  ordinary stores and copies them to every peer with a bulk copy. Every writing thread
  issues ``fence.proxy.async.shared::cta`` and the CTA synchronizes before the elected
  thread issues the copy; without the fence a peer can receive counts that miss some of
  the writes.
- Consumption: a thread that acquires a flag and then reads the data it publishes with
  TMA issues ``fence.proxy.async.global`` between the two. The acquire orders the data
  for the generic proxy only. This holds for the token communication and for the GEMM
  kernels that read the FC1 input and the FC1 output after a counter wait.
- Completion: ``token_in`` publishes FC1 ready flags only after the bulk stores of the
  rows they cover have completed: a full ``cp.async.bulk.wait_group 0`` right before every
  publication (each row when every row publishes, else the row that fills a batch, and the
  tail flush). ``cp.async.bulk.wait_group.read`` waits only until a store has read its
  shared-memory stage, which is all a row that does not publish needs before the stage is
  reused.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_NVLINK_DOMAIN = (
    _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/communication/nvlink_domain"
)
_SOURCES = sorted(
    path.name for path in _NVLINK_DOMAIN.glob("*.py") if 'sem="acquire"' in path.read_text()
)
_PUBLISH = "_broadcast_sizes_to_peers"
_PREAMBLE = [
    "cute.arch.fence_proxy('async.shared', space='cta')",
    "cute.arch.sync_threads()",
]
_GLOBAL_FENCE = "cute.arch.fence_proxy('async.global')"
_WAIT_HELPERS = ("wait_for_sizes_ready", "wait_for_metadata_ready")

# Every acquire wait of the three files, as (method, flag): the waits after which the method reads the published
# data with TMA, and the waits after which it reads them with ordinary loads only (and the wait helpers themselves).
# A new wait has to be sorted into one of them.
_TMA_READ_WAITS = {
    ("_compute_push_tables", "sizes_ready"),  # the peers' rows of the size matrix
    ("token_in", "wait_for_metadata_ready"),  # the peers' token rows
    ("_token_back_multi_stage", "fc2_done"),  # the FC2 rows of a local expert
    ("token_back", "fc2_done"),
}
_GENERIC_READ_WAITS = {
    ("_router_multiple_ctas", "source_base_ready"),
    ("_router_helper_multiple_ctas", "histogram_done"),
    ("_router_push_metadata", "sorted_metadata_ready"),
    ("_router_push_metadata", "push_table_ready"),
    ("token_in", "wait_for_sizes_ready"),
    ("token_back", "wait_for_sizes_ready"),
    ("token_back", "wait_for_metadata_ready"),
    ("wait_for_sizes_ready", "sizes_ready"),
    ("wait_for_sizes_ready", "wait_for_sizes_ready"),
    ("wait_for_metadata_ready", "metadata_ready"),
}


def _publishes_counts(statement: ast.stmt) -> bool:
    value = getattr(statement, "value", None)
    return (
        isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == _PUBLISH
    )


def _bodies(tree: ast.AST):
    for node in ast.walk(tree):
        for field in ("body", "orelse"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                yield body


def _publication_preambles(source: Path) -> list[tuple[int, list[str]]]:
    """Each statement that publishes counts, with the two statements before it."""
    sites = []
    for body in _bodies(ast.parse(source.read_text())):
        for index, statement in enumerate(body):
            if _publishes_counts(statement):
                preamble = [ast.unparse(s) for s in body[max(0, index - 2) : index]]
                sites.append((statement.lineno, preamble))
    return sites


def _acquired_flag(loop: ast.While) -> str | None:
    """The flag a spin loop acquires (the root name of the loaded address), unless a nested loop does."""
    for node in ast.walk(loop):
        if node is not loop and isinstance(node, ast.While):
            return None
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and any(
                k.arg == "sem" and isinstance(k.value, ast.Constant) and k.value.value == "acquire"
                for k in node.keywords
            )
        ):
            address = node.args[0]
            while isinstance(address, ast.BinOp):
                address = address.left
            return ast.unparse(address)
    return None


def _helper_wait(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr in _WAIT_HELPERS
    ):
        return statement.value.func.attr
    return None


def _acquire_waits(source: Path) -> list[tuple[str, str, int, list[str]]]:
    """Each wait as (method, flag, line, the two statements after it)."""
    waits = []
    for function in ast.walk(ast.parse(source.read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        for body in _bodies(function):
            for index, statement in enumerate(body):
                flag = _helper_wait(statement)
                if flag is None and isinstance(statement, ast.While):
                    flag = _acquired_flag(statement)
                if flag is None:
                    # A wait written with spin_wait is a wait too, and has to be sorted.
                    flag = _spin_wait_flag(statement)
                if flag is not None:
                    after = [ast.unparse(s) for s in body[index + 1 : index + 3]]
                    waits.append((function.name, flag, statement.lineno, after))
    return waits


@pytest.mark.parametrize("source", _SOURCES)
def test_router_counts_reach_the_async_proxy_before_publication(source: str) -> None:
    sites = _publication_preambles(_NVLINK_DOMAIN / source)

    # The single-CTA router publishes from its data CTA, the multi-CTA one from its helper.
    assert len(sites) == 2
    for lineno, preamble in sites:
        assert preamble == _PREAMBLE, f"{source}:{lineno} publishes counts without the fence"


@pytest.mark.parametrize("source", _SOURCES)
def test_tma_reads_of_published_data_follow_a_proxy_fence(source: str) -> None:
    waits = _acquire_waits(_NVLINK_DOMAIN / source)

    unsorted = {(m, f) for m, f, _, _ in waits} - _TMA_READ_WAITS - _GENERIC_READ_WAITS
    assert not unsorted, f"{source}: sort the new acquire waits {sorted(unsorted)}"
    tma_waits = [w for w in waits if (w[0], w[1]) in _TMA_READ_WAITS]
    # Every file has the size matrix and the token rows; token_back reads the FC2 rows in each.
    assert {(m, f) for m, f, _, _ in tma_waits} >= {
        ("_compute_push_tables", "sizes_ready"),
        ("token_in", "wait_for_metadata_ready"),
        ("token_back", "fc2_done"),
    }
    for method, flag, lineno, after in tma_waits:
        # A warp barrier may first order the lanes that acquired; the fence then orders the TMA after them.
        if after[:1] == ["cute.arch.sync_warp()"]:
            after = after[1:]
        assert after[:1] == [_GLOBAL_FENCE], (
            f"{source}:{lineno} {method} reads {flag}'s data with TMA unfenced"
        )


# ----------------------------------------------------------------------------- bulk drain


_FULL_WAIT = "cute.arch.cp_async_bulk_wait_group(0)"


def _is_full_wait(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node) == _FULL_WAIT


def _fc1_ready_publications(source: Path) -> list[tuple[int, str, int, int]]:
    """Each point where token_in may publish FC1 ready flags, within its method: every
    accumulate() with the token-in batch and the tracker's fire() after the loop. Returned as
    (line, the drain that guards it, the line of that drain, the line of the last bulk store
    before it). The drain is the last bulk wait before the point, or a full wait under an
    ``if <tracker>.publishes_on_accumulate(<the accumulate's phase and batch>)`` that is the
    last such guard before the point."""
    sites = []
    for function in ast.walk(ast.parse(source.read_text())):
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        publications = [
            c
            for c in calls
            if isinstance(c.func, ast.Attribute)
            and c.func.attr == "accumulate"
            and "self.token_in_flag_batch" in ast.unparse(c)
        ]
        if not publications:
            continue
        receivers = {ast.unparse(c.func.value) for c in publications}
        publications += [
            c
            for c in calls
            if isinstance(c.func, ast.Attribute)
            and c.func.attr == "fire"
            and ast.unparse(c.func.value) in receivers
        ]
        waits = [c for c in calls if ast.unparse(c.func) == "cute.arch.cp_async_bulk_wait_group"]
        stores = [c.lineno for c in calls if ast.unparse(c.func) == "cp_async_bulk_s2g"]
        guards = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "publishes_on_accumulate"
        ]
        for call in publications:
            last_store = max((line for line in stores if line < call.lineno), default=0)
            earlier_guards = [g for g in guards if g.lineno < call.lineno]
            guard = max(earlier_guards, key=lambda g: g.lineno) if earlier_guards else None
            if (
                guard is not None
                and call.func.attr == "accumulate"
                and guard.lineno > last_store
                and [ast.unparse(a) for a in guard.test.args]
                == [ast.unparse(a) for a in call.args[:2]]
                and any(_is_full_wait(n) for stmt in guard.body for n in ast.walk(stmt))
            ):
                sites.append((call.lineno, _FULL_WAIT, guard.lineno, last_store))
                continue
            earlier_waits = [w for w in waits if w.lineno < call.lineno]
            last_wait = max(earlier_waits, key=lambda w: w.lineno) if earlier_waits else None
            sites.append(
                (
                    call.lineno,
                    "" if last_wait is None else ast.unparse(last_wait),
                    0 if last_wait is None else last_wait.lineno,
                    last_store,
                )
            )
    return sites


@pytest.mark.parametrize("source", _SOURCES)
def test_fc1_ready_is_published_after_the_row_store_completes(source: str) -> None:
    sites = _fc1_ready_publications(_NVLINK_DOMAIN / source)

    assert sites
    for lineno, drain, drain_line, last_store in sites:
        assert drain == _FULL_WAIT and drain_line > last_store, (
            f"{source}:{lineno} may publish FC1 ready before its bulk stores complete "
            f"(last bulk store at {last_store}, drain {drain!r} at {drain_line})"
        )


# ----------------------------------------------------------------------------- GEMM-side waits

_KERNEL_SRC = _ROOT / "tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/kernel_src"
_SKIPPED_AFTER_A_WAIT = {
    "iket.range_pop()",
    "cute.arch.sync_threads()",
    "cute.arch.sync_warp()",
}

# Every spin_wait of the kernel sources that read data another CTA or rank published, as
# (file, method, flag) -> whether TMA reads the data it guards. A new wait has to be sorted.
_KERNEL_WAITS = {
    (
        "blackwell/inference/mega/block_scaled_swap_ab_fc12_extension.py",
        "wait_for_input",
        "counter_pointer",
    ): True,
    (
        "blackwell/inference/mega/block_scaled_swap_ab_fc12_mainloop.py",
        "run_tma_b",
        "counter_pointer",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_fc12_extension.py",
        "wait_for_input",
        "counter_pointer",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop.py",
        "run_tma_b",
        "counter_pointer",
    ): True,
    (
        "rubin/inference/local_mega/block_scaled_swap_ab_fc12_mainloop.py",
        "run_tma_b",
        "counter_pointer",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py",
        "run_tma_b",
        "counter_pointer",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_fc12_mainloop_gen_specialized.py",
        "run_tma_b",
        "fc1_sf_ready_pointer",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py",
        "_kernel",
        "self.token_comm.token_data_ready_region",
    ): True,
    (
        "rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py",
        "_kernel",
        "self.token_comm.sizes_ready_region",
    ): False,
    (
        "rubin/inference/mega/block_scaled_swap_ab_mega_moe_kernel_gen_specialized.py",
        "_kernel",
        "self.token_comm.metadata_ready_region",
    ): None,  # the epilogue reads the records with loads; the gather warps read the rows with TMA
    (
        "rubin/inference/mega/token_comm_gen_specialized.py",
        "_run_refine",
        "self.metadata_ready_region",
    ): True,
    (
        "rubin/inference/mega/token_comm_gen_specialized.py",
        "_run_refine",
        "self.token_sf_plain_ready_region",
    ): True,
    (
        "rubin/inference/mega/token_comm_gen_specialized.py",
        "_run_sort",
        "self.topk_ready_region",
    ): False,
    (
        "rubin/inference/mega/token_comm_gen_specialized.py",
        "_run_sort",
        "self.prefix_ready_region",
    ): False,
    (
        "rubin/inference/mega/token_comm_gen_specialized.py",
        "_run_helper",
        "self.rank_count_ready_region",
    ): False,
}


def _spin_wait_flag(statement: ast.stmt) -> str | None:
    """The flag a ``spin_wait(...)`` statement waits on: the region a ``ptr(...)`` names,
    or the root name of the address."""
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == "spin_wait"
    ):
        return None
    address = statement.value.args[0]
    while isinstance(address, ast.BinOp):
        address = address.left
    if isinstance(address, ast.Call) and getattr(address.func, "attr", None) == "ptr":
        address = address.args[0]
    return ast.unparse(address)


def _wait_flags(statement: ast.stmt) -> list[str]:
    """The flags a statement waits on: a spin_wait, or a block of them one thread runs."""
    flag = _spin_wait_flag(statement)
    if flag is not None:
        return [flag]
    if isinstance(statement, ast.If) and not statement.orelse:
        flags = [_spin_wait_flag(s) for s in statement.body]
        if flags and all(flags):
            return flags
    return []


def _kernel_waits(path: Path) -> list[tuple[str, list[str], int, str]]:
    """Each wait as (method, flags, line, the first statement after it that is not a barrier,
    a profiler range or another wait)."""
    waits = []
    tree = ast.parse(path.read_text())
    # The waits inside a one-thread block are found as the block.
    blocks = {
        id(node.body) for node in ast.walk(tree) if isinstance(node, ast.If) and _wait_flags(node)
    }
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for body in _bodies(function):
            if id(body) in blocks:
                continue
            index = 0
            while index < len(body):
                flags = _wait_flags(body[index])
                if not flags:
                    index += 1
                    continue
                lineno = body[index].lineno
                index += 1
                while index < len(body) and (
                    _wait_flags(body[index]) or ast.unparse(body[index]) in _SKIPPED_AFTER_A_WAIT
                ):
                    flags += _wait_flags(body[index])
                    index += 1
                after = ast.unparse(body[index]) if index < len(body) else ""
                waits.append((function.name, flags, lineno, after))
    return waits


@pytest.mark.parametrize("source", sorted({file for file, _, _ in _KERNEL_WAITS}))
def test_gemm_tma_reads_after_a_counter_wait_follow_a_proxy_fence(source: str) -> None:
    waits = _kernel_waits(_KERNEL_SRC / source)

    found = {(source, method, flag) for method, flags, _, _ in waits for flag in flags}
    expected = {key for key in _KERNEL_WAITS if key[0] == source}
    assert found == expected, f"{source}: sort the acquire waits {sorted(found ^ expected)}"
    for method, flags, lineno, after in waits:
        reads_with_tma = [_KERNEL_WAITS[(source, method, flag)] for flag in flags]
        if any(reads_with_tma):
            assert after == _GLOBAL_FENCE, (
                f"{source}:{lineno} {method} reads {flags}'s data with TMA unfenced"
            )
