# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The measurement switch TRTLLM_KEEP_PADDING_ROWS=1 keeps every row: the word is never written."""

import pytest
import torch

from tensorrt_llm._torch.moe import live_rows


@pytest.mark.cpu_only
def test_the_switch_stops_the_writes(monkeypatch):
    assert live_rows.LiveRowWord()._writes
    monkeypatch.setenv(live_rows.KEEP_ENV, "1")
    assert not live_rows.LiveRowWord()._writes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_megamoe_cutedsl_keep_switch_holds_every_row(monkeypatch):
    monkeypatch.setenv(live_rows.KEEP_ENV, "1")
    device = torch.device("cuda", torch.cuda.current_device())
    word = live_rows.LiveRowWord(live_rows.encode(3, 8))
    tensor = word.tensor(device)
    for value in (live_rows.encode(5, 8), live_rows.NO_ROWS):
        word.write(value)
    torch.cuda.synchronize()
    assert int(tensor.item()) == live_rows.ALL_ROWS
