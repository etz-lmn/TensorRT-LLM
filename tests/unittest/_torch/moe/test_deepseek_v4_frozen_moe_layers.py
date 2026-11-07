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
"""CPU tests of the experiment hook that builds DeepSeek-V4 MoE layers without the balancer."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models import modeling_deepseekv4
from tensorrt_llm._torch.moe.fused_moe import moe_load_balancer

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "raw,expected", [("", frozenset()), (" ", frozenset()), ("2,7", frozenset({2, 7}))]
)
def test_frozen_layers_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: frozenset
) -> None:
    monkeypatch.setenv(modeling_deepseekv4.FROZEN_MOE_LAYERS_ENV, raw)
    assert modeling_deepseekv4.frozen_moe_layers() == expected


@pytest.mark.parametrize("raw", ["7,2", "2,2", "-1", "two"])
def test_frozen_layers_must_be_ascending_distinct_indices(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(modeling_deepseekv4.FROZEN_MOE_LAYERS_ENV, raw)
    with pytest.raises(ValueError):
        modeling_deepseekv4.frozen_moe_layers()


@pytest.mark.parametrize("fails", [False, True])
def test_frozen_layer_sees_no_balancer_and_the_balancer_is_restored(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    seen = {}

    def create_moe(**kwargs):
        seen["thread_local"] = moe_load_balancer.get_moe_load_balancer()
        seen["config"] = kwargs["model_config"].moe_load_balancer
        if fails:
            raise RuntimeError("create_moe failed")
        return "moe"

    monkeypatch.setattr(modeling_deepseekv4, "create_moe", create_moe)
    balancer = object()
    model_config = SimpleNamespace(moe_load_balancer="balancer config")
    moe_load_balancer._current_moe_load_balancer.instance = balancer
    try:
        if fails:
            with pytest.raises(RuntimeError, match="create_moe failed"):
                modeling_deepseekv4.create_moe_without_load_balancer(
                    model_config=model_config, layer_idx=2
                )
        else:
            moe = modeling_deepseekv4.create_moe_without_load_balancer(
                model_config=model_config, layer_idx=2
            )
            assert moe == "moe"
        assert seen == {"thread_local": None, "config": None}
        assert model_config.moe_load_balancer == "balancer config"
        assert moe_load_balancer.get_moe_load_balancer() is balancer
    finally:
        del moe_load_balancer._current_moe_load_balancer.instance
