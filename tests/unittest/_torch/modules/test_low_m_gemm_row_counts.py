# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row counts of the low-M MMA kernels: serving never compiles a kernel for a new row count.

The kernel compile (``_compiled_mma``) and the CUDA stream are mocked, so these tests run on any
platform. A mocked kernel writes, into every output row, the first element of its input row, and
checks that every row-sized argument has the row count it was compiled for.
"""

import contextlib
import functools
import types

import pytest
import torch

import tensorrt_llm._torch.modules.low_m_gemm as _mod
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import warmup as _warmup

# Row counts a decode CUDA-graph warmup reaches with the batch sizes 1, 2, 4, ... 64.
_GRAPH_ROWS = (1, 2, 4, 8, 16, 32, 64)
_K = 128
_N = 16


class _Compiles:
    """Stands in for ``_compiled_mma`` under its ``lru_cache``: records the row count compiled."""

    def __init__(self):
        self.rows = []

    def __call__(self, m, *args, **kwargs):
        self.rows.append(m)

        def kernel(a, b, s, c, c2, pos, cos_sin, gamma, stream):
            assert a.shape[0] == c.shape[0] == c2.shape[0] == pos.shape[0] == m
            c.copy_(a[:, :1].to(c.dtype).expand_as(c))
            if c2 is not c:
                c2.copy_(a[:, :1].to(c2.dtype).expand_as(c2))

        return kernel


@pytest.fixture
def compiles(monkeypatch):
    fake = _Compiles()
    monkeypatch.setattr(_mod, "_compiled_mma", functools.lru_cache(maxsize=None)(fake))
    monkeypatch.setattr(_mod, "graph_safe_from_dlpack", lambda t: t)
    monkeypatch.setattr(_mod, "_COMPILED_ROWS", {}, raising=False)
    monkeypatch.setattr(_mod, "_ROW_COUNTS_FROZEN", False, raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: types.SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(_warmup, "warmup_named_allreduce_topologies", lambda model: None)
    return fake


class _Timer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def phase(self, name, **kwargs):
        return contextlib.nullcontext()


def _rows(m):
    """[m, K] BF16 rows whose first element is the row number + 1."""
    x = torch.zeros((m, _K), dtype=torch.bfloat16)
    x[:, 0] = torch.arange(1, m + 1, dtype=torch.bfloat16)
    return x


def _warm_up(rows=_GRAPH_ROWS):
    """``DecoderWarmup.run`` whose forwards reach ``rows`` low-M rows, as a decode graph warmup does."""
    weight = torch.zeros((_N, _K), dtype=torch.bfloat16)
    runner = object.__new__(_warmup.DecoderWarmup)
    runner.model = None
    runner._warmup_timer = _Timer()

    def forwards(resource_manager, kv_cache_manager):
        for m in rows:
            _mod.bf16_lowm_linear(_rows(m), weight)

    runner._warmup_scheduled = forwards
    runner.run(types.SimpleNamespace(get_resource_manager=lambda kind: None))
    return weight


def test_serving_compiles_no_row_count_after_warmup(compiles):
    """An eager step after warmup meets row counts the graph warmup never ran (3, 5, 6, 7, ...).
    Without the fix each one compiled a kernel mid-serving and stalled every rank."""
    weight = _warm_up()
    compiled_by_warmup = len(compiles.rows)
    for m in range(1, getattr(_mod, "_MMA_MAX_ROWS", 64) + 1):
        out = _mod.bf16_lowm_linear(_rows(m), weight)
        assert out.shape == (m, _N)
        assert torch.equal(out[:, 0], _rows(m)[:, 0].float())
    assert compiles.rows[compiled_by_warmup:] == []


def test_warmup_compiles_every_power_of_two(compiles):
    """A kernel warmup reached at one row count gets every power of two, under the same cache key
    the serving calls use, so no later row count compiles."""
    weight = _warm_up(rows=(8,))
    assert sorted(compiles.rows) == list(_mod._MMA_POWER_OF_TWO_ROWS)
    compiled_by_warmup = len(compiles.rows)
    for m in range(1, _mod._MMA_MAX_ROWS + 1):
        assert torch.equal(_mod.bf16_lowm_linear(_rows(m), weight)[:, 0], _rows(m)[:, 0].float())
    assert compiles.rows[compiled_by_warmup:] == []


def test_capture_keeps_exact_row_counts(compiles, monkeypatch):
    weight = _warm_up()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _mod.bf16_lowm_linear(_rows(7), weight)
    assert compiles.rows[-1] == 7


def test_reopened_row_counts_compile_exactly(compiles):
    weight = _warm_up()
    _mod.freeze_lowm_row_counts(False)
    out = _mod.bf16_lowm_linear(_rows(7), weight)
    assert compiles.rows[-1] == 7 and out.shape == (7, _N)


def test_padded_rows_of_split_and_rope_outputs(compiles):
    """The MXFP8 split output and the RoPE positions are padded and cut like the rows."""
    weight = torch.zeros((_N, _K), dtype=torch.uint8)
    scale = torch.zeros((_N, _K // 32), dtype=torch.uint8)
    cos_sin = torch.zeros((4, 2, 32), dtype=torch.float32)

    def run(m):
        positions = torch.zeros((m,), dtype=torch.int32)
        return _mod.mxfp8_lowm_linear(
            _rows(m), weight, scale, split_n=8, rope=(positions, cos_sin, 64)
        )

    run(8)
    _mod.freeze_lowm_row_counts(True)
    compiled_by_warmup = len(compiles.rows)
    for m in range(1, _mod._MMA_MAX_ROWS + 1):
        out, out2 = run(m)
        assert out.shape == (m, 8) and out2.shape == (m, _N - 8)
        assert torch.equal(out[:, 0], _rows(m)[:, 0]) and torch.equal(out2[:, 0], _rows(m)[:, 0])
    assert compiles.rows[compiled_by_warmup:] == []


def test_padded_rows_of_grouped_inverse_rope(compiles):
    weight = torch.zeros((_N, _K), dtype=torch.uint8)
    scale = torch.zeros((_N // 128 + 1, _K // 128), dtype=torch.float32)
    cos_sin = torch.zeros((4, 2, 32), dtype=torch.float32)

    def run(m):
        positions = torch.zeros((m,), dtype=torch.int32)
        return _mod.fp8_grouped_lowm(
            _rows(m), weight, scale, 0, inverse_rope=(positions, cos_sin, 64)
        )

    run(16)
    _mod.freeze_lowm_row_counts(True)
    compiled_by_warmup = len(compiles.rows)
    for m in range(1, _mod._MMA_MAX_ROWS + 1):
        out = run(m)
        assert out.shape == (m, _N) and torch.equal(out[:, 0], _rows(m)[:, 0])
    assert compiles.rows[compiled_by_warmup:] == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _mod.lowm_fp8_supported(), reason="needs SM100/SM103"
)
@pytest.mark.parametrize("kind", ["bf16_router", "fp8_block128"])
@pytest.mark.parametrize("m", [3, 5, 7, 13, 33])
def test_padded_rows_are_bitwise_their_own_kernels(kind, m, monkeypatch):
    """On the GPU: the real rows of a kernel compiled for 64 rows equal the m-row kernel's."""
    monkeypatch.setattr(_mod, "_COMPILED_ROWS", {})
    monkeypatch.setattr(_mod, "_ROW_COUNTS_FROZEN", False)
    torch.manual_seed(m)
    if kind == "bf16_router":
        k, n = 7168, 384
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        run = lambda x: _mod.bf16_lowm_linear(x, weight)  # noqa: E731
    else:
        k, n = 7168, 1536
        weight = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.ones((n // 128, k // 128), dtype=torch.float32, device="cuda")
        run = lambda x: _mod.fp8_grouped_lowm(x, weight, scale, 0)  # noqa: E731
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    exact = run(x)
    padded = run(_mod._pad_rows(x, 64))[:m]
    assert torch.equal(exact, padded)
