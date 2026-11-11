# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The measurement switch TRTLLM_MOE_ROW_COUNT_DIR: per forward, every MoE layer's live rows per
expert, nothing at all with the switch off."""

import ast
import base64
import importlib
import inspect
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.moe import live_rows, row_count

TOP_K = 6
EXPERTS = 16
PADDING_EXPERT = 15  # padding rows route here only, so a count on it is a counted padding row


def fake_record(ids, rows, forwards, enabled, first=True, live_row_word=None):
    """trtllm::adaptive_eplb_record on the host: the ids of the word's live rows, by the op's rule."""
    if int(enabled[0]) == 0:
        return
    flat = ids.reshape(-1)
    count = flat.numel()
    keep = count
    if live_row_word is not None:
        word = int(live_row_word[0])
        real, requests = word & 0xFFFF, (word >> 16) & 0xFFFF
        tokens = count // TOP_K
        if requests == 0:
            keep = 0
        elif real < requests and tokens % requests == 0:
            keep = tokens // requests * real * TOP_K
    kept = flat[:keep].long()
    kept = kept[(kept >= 0) & (kept < rows.numel())]
    rows += torch.bincount(kept, minlength=rows.numel()).to(rows.dtype)
    if first:
        forwards += 1


@pytest.fixture
def host_op(monkeypatch):
    monkeypatch.setattr(torch.ops.trtllm, "adaptive_eplb_record", fake_record, raising=False)
    word = torch.full((1,), live_rows.ALL_ROWS, dtype=torch.int32)
    monkeypatch.setattr(live_rows, "current_word", lambda device: word)
    return word


def routes(real_requests, requests, rows_per_request):
    """[requests * rows_per_request, TOP_K] ids: real rows on experts 0..5, padding rows on PADDING_EXPERT."""
    real = torch.arange(TOP_K, dtype=torch.int32).repeat(real_requests * rows_per_request, 1)
    padding = torch.full(
        ((requests - real_requests) * rows_per_request, TOP_K), PADDING_EXPERT, dtype=torch.int32
    )
    return torch.cat([real, padding])


def read_lines(path):
    lines = [json.loads(line) for line in Path(path).read_text().splitlines()]
    for line in lines:
        if line["kind"] == "forward":
            raw = zlib.decompress(base64.b64decode(line["rows"]))
            shape = (len(line["launches"]), EXPERTS)
            line["rows"] = np.frombuffer(raw, dtype=line["dtype"]).reshape(shape)
    return lines


def counter(tmp_path, layers=(0, 2)):
    return row_count.RowCounter(str(tmp_path), 3, list(layers), EXPERTS, TOP_K, torch.device("cpu"))


@pytest.mark.cpu_only
def test_the_switch_is_off_without_the_directory(monkeypatch):
    monkeypatch.delenv(row_count.ENV, raising=False)
    assert not importlib.reload(row_count).ENABLED
    monkeypatch.setenv(row_count.ENV, "/tmp/rows")
    assert importlib.reload(row_count).ENABLED
    monkeypatch.delenv(row_count.ENV)
    importlib.reload(row_count)


@pytest.mark.cpu_only
def test_every_caller_is_behind_the_switch():
    from tensorrt_llm._torch.moe.fused_moe import moe_scheduler
    from tensorrt_llm._torch.pyexecutor import py_executor
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner

    for module, names in (
        (moe_scheduler, {"record"}),
        (py_executor, {"set_iter"}),
        (runner, {"attach"}),
    ):
        tree = ast.parse(inspect.getsource(module))
        guarded = set()
        for node in ast.walk(tree):
            switch = isinstance(node, (ast.If, ast.IfExp)) and ast.unparse(node.test)
            if switch and "row_count.ENABLED" in switch:
                body = node.body if isinstance(node, ast.If) else [node.body]
                guarded |= {id(n) for b in body for n in ast.walk(b)}
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and ast.unparse(n.func) in {f"row_count.{name}" for name in names}
        ]
        assert calls and all(id(c) in guarded for c in calls), module.__name__


@pytest.mark.cpu_only
def test_only_live_rows_count(tmp_path, host_op):
    rows = counter(tmp_path)
    rows.begin()
    host_op[0] = live_rows.encode(3, 4)
    rows.record(0, routes(3, 4, 6), True)
    host_op[0] = live_rows.NO_ROWS
    rows.record(2, routes(0, 1, 6), True)
    rows.end(dict(iter=7))
    rows.begin()
    host_op[0] = live_rows.ALL_ROWS  # a launch over part of the forward keeps every row
    rows.record(0, routes(3, 4, 6), True)
    rows.end(dict(iter=8))
    rows.close()
    header, first, second, end = read_lines(rows.path)
    assert header["kind"] == "header" and header["layers"] == [0, 2] and header["top_k"] == TOP_K
    assert first["iter"] == 7 and first["launches"] == [1, 0, 1]
    assert first["totals"] == [3 * 6 * TOP_K, 0, 0]
    assert first["rows"][0, PADDING_EXPERT] == 0 and first["rows"][0, :TOP_K].tolist() == [18] * 6
    assert second["iter"] == 8 and second["rows"][0, PADDING_EXPERT] == 6 * TOP_K
    assert second["totals"] == [4 * 6 * TOP_K, 0, 0]
    assert end == dict(kind="end", forwards=2, stalls=0, unrecorded_forwards=0, error=None)


@pytest.mark.cpu_only
def test_chunks_count_once_per_forward(tmp_path, host_op):
    rows = counter(tmp_path)
    rows.begin()
    for first in (True, False):
        rows.record(2, routes(2, 2, 6), first)
    rows.end(dict(iter=1))
    rows.close()
    forward = read_lines(rows.path)[1]
    assert forward["launches"] == [0, 0, 1] and forward["totals"] == [0, 0, 2 * 2 * 6 * TOP_K]


@pytest.mark.cpu_only
def test_unrecorded_forwards_leave_nothing(tmp_path, host_op):
    rows = counter(tmp_path)
    rows.record(0, routes(4, 4, 6), True)  # warmup or capture before the first recorded forward
    rows.skipped()
    rows.begin()
    rows.record(0, routes(1, 1, 6), True)
    rows.end(dict(iter=5))
    rows.record(0, routes(4, 4, 6), True)  # a forward of the pass that is not recorded
    rows.skipped()
    rows.begin()
    rows.end(dict(iter=6))
    rows.close()
    _, first, second, end = read_lines(rows.path)
    assert first["totals"][0] == 6 * TOP_K and first["launches"][0] == 1
    assert second["totals"][0] == 0 and second["launches"][0] == 0
    assert end["unrecorded_forwards"] == 2


@pytest.mark.cpu_only
def test_an_unknown_layer_and_a_second_writer_are_refused(tmp_path, host_op):
    rows = counter(tmp_path)
    with pytest.raises(RuntimeError, match="layer 1"):
        rows.record(1, routes(1, 1, 6), True)
    with pytest.raises(FileExistsError):
        counter(tmp_path)
    rows.close()


@pytest.mark.cpu_only
def test_each_pass_hands_its_iteration_to_one_forward():
    row_count.set_iter(11)
    assert row_count.take_iter() == 11
    assert row_count.take_iter() is None


@pytest.mark.cpu_only
def test_attach_finds_the_layers_and_refuses_kept_padding_rows(tmp_path, host_op, monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import MoEScheduler

    class Scheduler(MoEScheduler):
        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class Layer(torch.nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            self.scheduler = Scheduler(self)
            self.layer_idx, self.num_experts = layer_idx, EXPERTS
            self.routing_method = SimpleNamespace(experts_per_token=TOP_K)

    model = torch.nn.ModuleList([Layer(i) for i in (4, 0, 63)])
    model.extra_attrs = {}
    monkeypatch.setenv(row_count.ENV, str(tmp_path))
    monkeypatch.setenv(live_rows.KEEP_ENV, "1")
    with pytest.raises(RuntimeError, match=live_rows.KEEP_ENV):
        row_count.attach(model, 0, torch.device("cpu"))
    monkeypatch.delenv(live_rows.KEEP_ENV)
    attached = row_count.attach(model, 0, torch.device("cpu"))
    assert model.extra_attrs[row_count.MODEL_EXTRA_ATTR] is attached
    attached.close()
    header = read_lines(attached.path)[0]
    assert header["layers"] == [0, 4, 63] and header["layer_slots"] == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_op_counts_live_rows_inside_a_cuda_graph(tmp_path):
    try:
        torch.ops.trtllm.adaptive_eplb_record
    except (AttributeError, RuntimeError):
        pytest.skip("needs trtllm::adaptive_eplb_record")
    device = torch.device("cuda", torch.cuda.current_device())
    word = live_rows.LiveRowWord(live_rows.encode(4, 4))
    word_tensor = word.tensor(device)
    rows = row_count.RowCounter(str(tmp_path), 0, [0], EXPERTS, TOP_K, device)
    ids = routes(4, 4, 6).to(device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            rows._layers[0].record(ids, True, word_tensor)
        ids.copy_(routes(3, 4, 6).to(device))
        for real in (3, 0):
            word.write(live_rows.encode(real, 4) if real else live_rows.NO_ROWS)
            rows.begin()
            graph.replay()
            rows.end(dict(iter=real))
    torch.cuda.synchronize()
    rows.close()
    _, three, none, _ = read_lines(rows.path)
    assert three["totals"] == [3 * 6 * TOP_K] and three["rows"][0, PADDING_EXPERT] == 0
    assert none["totals"] == [0] and none["launches"] == [1]
