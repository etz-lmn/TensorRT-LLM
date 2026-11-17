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
"""Bitwise A/B of MegaMoE outputs across two builds of the token communication (a measurement driver).

Runs the four-rank MegaMoE layer of test_megamoe_live_rows_gpu.py with the NVFP4 combine, the
configuration whose token-back waits for fc2_done and reads the FC2 rows with TMA. Each arm
saves every rank's outputs, with the digest of the token_comm.py module it ran, into a
directory of its own, named by the arm (the stem of its JUnit file). When the other arm's
directory is already there (the arm that ran first), this arm only reads it: the two digests
must differ (else the A/B compared one build with itself) and the outputs must be equal bit
for bit. A proxy fence orders memory and changes no arithmetic, so any difference is a failure.

The run also records, from the kernel descriptor the forward used, that token-back pushes the
FC2 rows (token_back_push_data) and that the FC2 epilogue stores them without bulk copies:
the path the fc2_done fences sit on. The dump root is MEGAMOE_BITWISE_DUMP_ROOT, else the
directory of the JUnit file the run writes (the A/B harness writes one per arm into its run
directory); without either the test skips.
"""

import hashlib
import importlib
import json
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from tensorrt_llm._utils import get_sm_version, mpi_rank

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_COMBINE_FORMAT = "16e2m1xbf16"
_SEEDS = (1, 2, 3)
_TOKEN_COMM = (
    "tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.communication.nvlink_domain.token_comm"
)
_JUNIT_SUFFIX = ".junit.xml"


def _dump_root_and_arm(config: pytest.Config) -> tuple[Path, str] | None:
    """The directory both arms write under and this arm's name (its JUnit file's stem), or None."""
    junit = config.getoption("xmlpath", None)
    arm = os.environ.get("MEGAMOE_BITWISE_ARM") or (
        Path(junit).name.removesuffix(_JUNIT_SUFFIX) if junit else None
    )
    root = os.environ.get("MEGAMOE_BITWISE_DUMP_ROOT")
    if root is None and junit:
        root = str(Path(junit).parent / "megamoe-fence-bitwise" / _COMBINE_FORMAT)
    return (Path(root), arm) if root and arm else None


def _token_comm_digest() -> str:
    """The digest of the token_comm.py this process imported, read at the module's own path: the A/B
    harness overlays a changed file as a copy and an unchanged one as a symlink, so the path must not
    be resolved to the image's package directory."""
    module = importlib.import_module(_TOKEN_COMM)
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()[:16]


def _record_and_compare(root: Path, arm: str, digest: str, results: list) -> int:
    """Write this arm's outputs into root/<arm>, then compare them with every other arm already there.

    Returns how many other arms were compared. The arm's own directory must not exist yet: a rerun of
    the same arm into one run directory is refused rather than mixed with an earlier dump."""
    mine = root / arm
    assert not mine.exists(), f"{mine} already holds a dump of arm {arm}: refused, not overwritten"
    root.mkdir(parents=True, exist_ok=True)  # shared by both arms
    staging = Path(tempfile.mkdtemp(prefix=f".{arm}.", dir=root))  # always a new directory
    for result in results:
        torch.save(result["outputs"], staging / f"rank{result['rank']}.pt")
    (staging / "path.json").write_text(
        json.dumps(dict(arm=arm, digest=digest, paths=[r["path"] for r in results]), indent=1)
        + "\n"
    )
    staging.rename(mine)

    others = sorted(
        d for d in root.iterdir() if d.is_dir() and d.name != arm and not d.name.startswith(".")
    )
    for other in others:
        theirs_meta = json.loads((other / "path.json").read_text())
        assert theirs_meta["digest"] != digest, (
            f"arms {other.name} and {arm} imported the same token_comm.py ({digest}): nothing was compared"
        )
        for result in results:
            theirs = torch.load(other / f"rank{result['rank']}.pt")
            assert len(theirs) == len(result["outputs"])
            for index, (a, b) in enumerate(zip(theirs, result["outputs"])):
                assert torch.equal(a, b), (
                    f"rank {result['rank']} output {index} differs between arm {other.name} "
                    f"(token_comm {theirs_meta['digest']}) and arm {arm} (token_comm {digest})"
                )
    return len(others)


def _worker(seeds: tuple) -> dict:
    try:
        return _worker_impl(seeds)
    except Exception:
        traceback.print_exc()
        raise


def _worker_impl(seeds: tuple) -> dict:
    from _torch.moe.moe_test_utils import MoeBackendType
    from _torch.moe.quantize_utils import get_test_quant_params
    from _torch.moe.test_megamoe_live_rows_gpu import (
        _REAL_REQUESTS,
        _REQUESTS,
        _ROWS_PER_REQUEST,
        _SHAPE,
        _WORLD,
        _forward_kernels,
    )
    from _torch.moe.test_moe_module import (
        _create_mapping_for_parallel_mode,
        _create_model_config,
        _ensure_dist_for_megamoe,
    )

    from tensorrt_llm._torch.moe import live_rows
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op
    from tensorrt_llm._torch.moe.fused_moe import (
        DEFAULT_MOE_ACTIVATION,
        RenormalizeMoeRoutingMethod,
        create_moe,
    )
    from tensorrt_llm._torch.moe.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scope
    from tensorrt_llm._torch.utils import ActivationType, model_extra_attrs
    from tensorrt_llm.models.modeling_utils import QuantAlgo

    os.environ.pop("TRTLLM_FORCE_COMM_METHOD", None)
    os.environ["MEGAMOE_COMBINE_FORMAT"] = _COMBINE_FORMAT
    rank = mpi_rank()
    mapping = _create_mapping_for_parallel_mode(_WORLD, "DEP")
    mapping.rank = rank
    torch.cuda.set_device(rank)
    backend_name = MoeBackendType.MEGAMOE_CUTEDSL.value
    _ensure_dist_for_megamoe(backend_name, rank, _WORLD)
    num_experts, top_k, hidden_size, intermediate_size = _SHAPE
    rows = _REQUESTS * _ROWS_PER_REQUEST
    dtype = torch.bfloat16

    with torch.device(f"cuda:{rank}"):
        torch.manual_seed(0)
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k, force_enable_pytorch_op=True)
        x = torch.randn((rows, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((rows, num_experts), dtype=dtype, device="cuda")
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            QuantAlgo.NVFP4, x, MoeBackendType.MEGAMOE_CUTEDSL
        )
        quant_kwargs.pop("ref_cls", None)
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=False,
            swiglu_gptoss_style=False,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            num_local_experts=num_experts // mapping.moe_ep_size,
            activation_type=ActivationType.Swiglu,
        )
        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=backend_name,
            max_num_tokens=max(256, rows),
        )
        with (
            nccl_window_tensor_scope(x),
            create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=False,
                activation=DEFAULT_MOE_ACTIVATION,
                weight_loading_mode=getattr(
                    quantize_util, "weight_loading_mode", MoEWeightLoadingMode.VANILLA
                ),
            ) as fused_moe,
        ):
            fused_moe.create_weights()
            fused_moe.load_weights([quantize_util.create_weights(**quant_kwargs)])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{rank}")
            word = live_rows.LiveRowWord()
            attrs = {live_rows.LIVE_ROW_WORD_ATTR: word}

            def forward(inputs: torch.Tensor, logits: torch.Tensor, value: int) -> tuple:
                word.write(value)
                with (
                    torch.inference_mode(),
                    model_extra_attrs(attrs),
                    _forward_kernels(op.Sm100MegaMoENvfp4Runner) as kernels,
                ):
                    output = fused_moe.forward(
                        inputs, logits, all_rank_num_tokens=[rows] * _WORLD
                    ).clone()
                torch.cuda.synchronize()
                return output.cpu(), kernels

            requests = [
                SimpleNamespace(is_dummy=index >= _REAL_REQUESTS[rank])
                for index in range(_REQUESTS)
            ]
            dropped = live_rows.forward_word(
                requests,
                num_contexts=0,
                seq_lens=torch.full((_REQUESTS,), _ROWS_PER_REQUEST, dtype=torch.int32),
            )
            outputs, kernels = [], []
            for seed in seeds:
                # New values in the same tensors, which the window scope above registered.
                generator = torch.Generator(device="cuda").manual_seed(seed)
                x.copy_(torch.randn(x.shape, dtype=dtype, device="cuda", generator=generator))
                router_logits.copy_(
                    torch.randn(
                        router_logits.shape, dtype=dtype, device="cuda", generator=generator
                    )
                )
                for value in (live_rows.ALL_ROWS, live_rows.ALL_ROWS, dropped):
                    output, used = forward(x, router_logits, value)
                    outputs.append(output)
                    kernels.extend(used)
            kernel = kernels[0]
            path = dict(
                token_back_mode=kernel.token_back_mode,
                token_back_push_data=bool(kernel.token_comm.token_back_push_data),
                fc2_use_bulk=bool(kernel.fc2_use_bulk),
                combine_stage_depth=int(kernel.token_comm.combine_stage_depth),
            )
            return dict(rank=rank, outputs=outputs, path=path)


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the MegaMoE Blackwell kernel",
)
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.threadleak(enabled=False)  # the pool's manager thread persists by design
def test_megamoe_nvfp4_outputs_are_bitwise_equal_across_builds(
    request: pytest.FixtureRequest,
) -> None:
    from _torch.moe.test_megamoe_live_rows_gpu import _WORLD, _init_worker

    located = _dump_root_and_arm(request.config)
    if located is None:
        pytest.skip(
            "needs a JUnit file (the one-node A/B harness) or MEGAMOE_BITWISE_DUMP_ROOT and _ARM"
        )
    root, arm = located
    with MPIPoolExecutor(
        initializer=_init_worker, initargs=(sys.path,), max_workers=_WORLD
    ) as executor:
        results = sorted(executor.map(_worker, [_SEEDS] * _WORLD), key=lambda r: r["rank"])

    for result in results:
        path = result["path"]
        assert path["token_back_push_data"] and not path["fc2_use_bulk"], (
            f"rank {result['rank']} did not run the STG FC2 / token-back path: {path}"
        )
        outs = result["outputs"]
        for first, again in zip(outs[0::3], outs[1::3]):
            assert torch.equal(first, again), f"rank {result['rank']}: a repeat differs"

    _record_and_compare(root, arm, _token_comm_digest(), results)
