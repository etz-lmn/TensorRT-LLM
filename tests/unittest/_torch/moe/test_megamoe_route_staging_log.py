# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.mark.cpu_only
def test_route_staging_path_logged_once_per_bucket(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mod

    lines = []
    monkeypatch.setattr(mod, "_ROUTE_STAGING_SEEN", set())
    monkeypatch.setattr(mod.logger, "info", lines.append)
    for path, bucket in [
        ("megamoe_stage_routes", 16),
        ("megamoe_stage_routes", 16),
        ("megamoe_stage_routes", 32),
        ("the balancer's route op wrote them", 16),
    ]:
        mod._log_route_staging(path, bucket)
    assert lines == [
        "MegaMoE route staging: megamoe_stage_routes (bucket 16)",
        "MegaMoE route staging: megamoe_stage_routes (bucket 32)",
        "MegaMoE route staging: the balancer's route op wrote them (bucket 16)",
    ]
