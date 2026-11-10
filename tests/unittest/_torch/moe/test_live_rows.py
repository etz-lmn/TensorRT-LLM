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
"""Host reference of the live-row word (``live_rows``, which the kernels compute too) and the engine's words."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.moe import live_rows as lr
from tensorrt_llm._torch.utils import model_extra_attrs

pytestmark = pytest.mark.cpu_only


def live_rows(word: int, rows: int) -> int:
    """The leading rows of ``rows`` rows that ``word`` keeps."""
    real, requests = word & lr.MAX_REQUESTS, word >> lr.REQUEST_BITS
    if requests == 0:
        return 0
    return rows if real >= requests or rows % requests else rows // requests * real


def _requests(*kinds: str):
    """Requests in row order: ``r`` real, ``d`` padding."""
    return [SimpleNamespace(is_dummy=kind == "d") for kind in kinds]


def _row_owner(requests, rows_per_request):
    """The request of every row: each request owns ``rows_per_request`` consecutive rows."""
    return [request for request in requests for _ in range(rows_per_request)]


def _kept_rows(requests, rows_per_request, *, uniform=True):
    word = lr.forward_word(requests, uniform=uniform)
    return live_rows(word, len(requests) * rows_per_request)


@pytest.mark.parametrize("rows_per_request", [1, 4, 3])
@pytest.mark.parametrize(
    "kinds",
    [
        "rrrrrrrr",  # a full batch
        "rrrrrddd",  # graph padding behind five real requests
        "rddddddd",  # one real request left
        "dddddddd",  # an attention-DP placeholder rank padded to the graph size
        "d",  # an attention-DP placeholder rank running eagerly
    ],
)
def test_padding_rows_dropped_and_real_rows_kept(kinds, rows_per_request):
    # 1 row per request: no speculation; 4: the target forward of one-model MTP with
    # three draft tokens; 3: an MTP draft layer over the same requests.
    requests = _requests(*kinds)
    kept = _kept_rows(requests, rows_per_request)
    for row, owner in enumerate(_row_owner(requests, rows_per_request)):
        assert (row < kept) == (not owner.is_dummy), (kinds, rows_per_request, row)


def test_every_launch_of_one_forward_keeps_its_own_rows():
    # One word serves launches of different row counts in one forward.
    word = lr.forward_word(_requests(*"rrrdd"), uniform=True)
    assert word == lr.encode(3, 5)
    assert [live_rows(word, rows) for rows in (5, 10, 15, 20)] == [3, 6, 9, 12]


def test_rows_not_split_evenly_into_requests_are_all_kept():
    word = lr.encode(3, 5)
    assert live_rows(word, 7) == 7
    assert live_rows(word, 0) == 0


def test_padding_before_a_real_request_is_kept():
    # Only padding behind the last real request is dropped.
    requests = _requests(*"rdrdd")
    assert lr.forward_word(requests, uniform=True) == lr.encode(3, 5)
    assert live_rows(lr.forward_word(requests, uniform=True), 10) == 6
    assert lr.forward_word(_requests(*"rrdr"), uniform=True) == lr.ALL_ROWS


def test_rows_of_unequal_requests_are_kept_unless_all_are_padding():
    assert lr.forward_word(_requests(*"rrdd"), uniform=False) == lr.ALL_ROWS
    assert lr.forward_word(_requests(*"dd"), uniform=False) == lr.NO_ROWS
    assert lr.forward_word([], uniform=True) == lr.ALL_ROWS


def test_word_layout():
    assert lr.ALL_ROWS == lr.encode(1, 1)
    assert lr.NO_ROWS == 0
    assert live_rows(lr.ALL_ROWS, 96) == 96
    assert live_rows(lr.NO_ROWS, 96) == 0
    assert live_rows(lr.encode(0, 4), 96) == 0
    assert live_rows(lr.encode(4, 4), 96) == 96
    top = lr.MAX_REQUESTS
    assert live_rows(lr.encode(top - 1, top), 2 * top) == 2 * (top - 1)
    assert lr.encode(top, top) >> lr.REQUEST_BITS == top
    for real, requests in ((1, 0), (2, 1), (-1, 1), (0, top + 1)):
        with pytest.raises(ValueError):
            lr.encode(real, requests)


def test_kernels_read_the_word_of_the_running_forward(monkeypatch):
    # Writes before a kernel allocates the word only record the value: no device.
    engine = lr.LiveRowWord()
    engine.write(lr.encode(2, 4))
    assert engine._value == lr.encode(2, 4)
    monkeypatch.setattr(lr.LiveRowWord, "tensor", lambda word, device: word)
    with model_extra_attrs({lr.MODEL_EXTRA_ATTR: engine}):
        assert lr.current_word("cuda:0") is engine
    for attrs in ({}, None):
        with model_extra_attrs(attrs):
            assert lr.current_word("cuda:0") is lr._ALL_ROWS_WORDS[0]
    assert lr._ALL_ROWS_WORDS[0]._value == lr.ALL_ROWS
