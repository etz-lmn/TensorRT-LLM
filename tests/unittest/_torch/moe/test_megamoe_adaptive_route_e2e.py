# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MegaMoE layers on four GPUs (attention DP, EP4, 384 experts) with and without an adaptive load
balancer (512 slots, home route tables, no runtime), at decode (8 / 64 / 128 rows per rank, the one-round router)
and prefill (1024 / 4096 / 8192, the metadata-push router) sizes.

Bit for bit: an adaptive layer whose dispatch router maps and counts its routes returns what the same layer returns
on the staged route, and what a layer without the balancer returns; its statistics are the routing's picks of the
live rows, nothing while the recording flag is clear.

Timed (eager, per layer, reported as JUnit properties; judged by the reader, not here): the fused route without a
balancer, the adaptive layer on the staged route, and on the router's route. Wall per layer is a chain of the
layers between two CUDA events; host per layer is the launch loop up to its synchronization.
"""

import json
import pickle
import statistics
import sys
import time
from typing import Optional

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm._utils import mpi_allgather, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_WORLD_SIZE = 4
_NUM_EXPERTS = 384
_NUM_SLOTS = 512
_TOP_K = 6
_HIDDEN_SIZE = 7168
_INTERMEDIATE_SIZE = 3072
_CLAMP = 10.0
_SCALING = 2.5
_DECODE_ROWS = (8, 64, 128)
_PREFILL_ROWS = (1024, 4096, 8192)
_MAX_NUM_TOKENS = 8192
_LAYERS = 8  # distinct layers per arm; a timed chain calls them in turn
_CHAIN = 61  # layer calls per timed chain, the model's MoE layers
_REPS = 20
_ROUNDS = 3

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)


def _init_worker(paths: list[str]) -> None:
    sys.path[:] = paths


def _assert_on_all_ranks(problem: Optional[str]) -> None:
    problems = mpi_allgather(problem)
    assert not any(problems), problems


def _replica_row(packed: list) -> torch.Tensor:
    """A route table row with replicas, as an install publishes one: the 128 spare slots each take a copy of one
    of 32 hot experts, with shares that include 0 and 1 copies (policy.build_route_tables)."""
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        ExpertGeometry,
        build_route_tables,
    )

    geometry = ExpertGeometry(_NUM_EXPERTS, _NUM_SLOTS, _WORLD_SIZE)
    layout = geometry.home_layout().tolist()
    free = [slot for slot, expert in enumerate(layout) if expert < 0]
    for k, slot in enumerate(free):
        layout[slot] = (k * 12) % _NUM_EXPERTS
    shares = [1.0] * _NUM_SLOTS
    for k, slot in enumerate(free):
        shares[slot] = (0.0, 1.0, 0.5, 2.0)[k % 4]
    return torch.tensor(
        build_route_tables(layout, shares, _NUM_EXPERTS, _NUM_SLOTS).packed(), dtype=torch.int32
    )


def _run() -> list:
    # Here, not at module level: quantize_utils needs parameterized, which only the GPU image has.
    from _torch.moe.quantize_utils import NVFP4QuantizeUtil

    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
        find_backends,
        prepare_layers,
    )
    from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import maybe_create_moe_load_balancer
    from tensorrt_llm._torch.moe.kernels.cutedsl_megamoe.kernel_src.blackwell.inference.mega import (
        block_scaled_swap_ab_mega_moe_kernel as kernel_module,
    )
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig

    # The one-round router engages from EP16; lower its gate so EP4 takes it at decode sizes, as EP32 does.
    kernel_module._ONE_ROUND_MIN_WORLD_SIZE = 1
    rank, world_size = mpi_rank(), mpi_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
        moe_ep_size=world_size,
        moe_tp_size=1,
        enable_attention_dp=True,
    )
    pretrained = PretrainedConfig(
        architectures=["DeepseekV41ForCausalLM"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_experts=_NUM_EXPERTS,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=_SCALING,
        n_hash_layers=0,
        n_shared_experts=1,
        vocab_size=129280,
        swiglu_limit=_CLAMP,
        torch_dtype=torch.bfloat16,
    )
    routed_quant = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)

    def model_config(adaptive: bool) -> ModelConfig:
        config = ModelConfig(
            pretrained_config=pretrained,
            mapping=mapping,
            quant_config=routed_quant,
            moe_backend="MEGAMOE_CUTEDSL",
            max_num_tokens=_MAX_NUM_TOKENS,
            moe_max_num_tokens=_MAX_NUM_TOKENS,
        )
        if adaptive:
            config.moe_load_balancer = MoeAdaptiveLoadBalancerConfig(num_slots=_NUM_SLOTS)
        return config

    results = []
    with device, torch.inference_mode():
        torch.manual_seed(1701)
        quantizer = NVFP4QuantizeUtil(
            num_experts=_NUM_EXPERTS,
            dtype=torch.bfloat16,
            intermediate_size=_INTERMEDIATE_SIZE,
            hidden_size=_HIDDEN_SIZE,
            quant_config=routed_quant,
        )
        weights = quantizer.create_weights(x_sf_global=torch.tensor(448.0 * 6.0))
        gate_weight = torch.randn(_NUM_EXPERTS, _HIDDEN_SIZE) / _HIDDEN_SIZE**0.5
        bias = torch.randn(_NUM_EXPERTS, dtype=torch.float32) * 0.05

        def build(config: ModelConfig, layer_idx: int) -> DeepseekV4MoE:
            moe = DeepseekV4MoE(
                num_experts=_NUM_EXPERTS,
                top_k=_TOP_K,
                hidden_size=_HIDDEN_SIZE,
                intermediate_size=_INTERMEDIATE_SIZE,
                shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
                aux_stream_dict={
                    AuxStreamType.MoeShared: torch.cuda.Stream(),
                    AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
                },
                dtype=torch.bfloat16,
                model_config=config,
                override_quant_config=routed_quant,
                layer_idx=layer_idx,
            )
            moe.experts.load_weights([weights])
            moe.experts.post_load_weights()
            moe.gate.load_weights([{"weight": gate_weight, "e_score_correction_bias": bias}])
            # The routed experts are what differs; the shared expert stays out of every arm.
            moe.shared_experts = None
            return moe

        plain = torch.nn.ModuleList(build(model_config(False), i) for i in range(_LAYERS))
        adaptive_config = model_config(True)
        with maybe_create_moe_load_balancer(adaptive_config, mapping) as balancer:
            adaptive = torch.nn.ModuleList(build(adaptive_config, i) for i in range(_LAYERS))
        balancer.register_weight_slots_after_to_cuda()
        balancer.finalize_model()
        prepare_layers(find_backends(adaptive, balancer.layers), balancer.layers, device)
        backends = [layer.experts.backend for layer in adaptive]
        router_route = type(backends[0]).routes_adaptive_in_router

        def use_router(on: bool) -> None:
            for backend in backends:
                backend.routes_adaptive_in_router = (
                    router_route.__get__(backend) if on else (lambda num_tokens: False)
                )

        def record(on: bool) -> None:
            balancer.observe_scheduled_batch(
                1 if on else 0, True, False, lambda start, stop: (0, 0, 0)
            )

        def forward(layers, x, counts):
            return [layer(x, all_rank_num_tokens=counts).clone() for layer in layers[:2]]

        # Bit for bit, decode and prefill sizes, with the statistics of the router's route.
        for rows in (*_DECODE_ROWS, *_PREFILL_ROWS):
            counts = [rows] * world_size
            torch.manual_seed(2000 + rows + rank)
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            want = forward(plain, x, counts)
            use_router(False)
            staged = forward(adaptive, x, counts)
            stats = balancer.statistics(device)
            stats["storage"].zero_()
            use_router(True)
            record(True)
            # The picks the first layer's fused route returns, as the router reads them from the bucket.
            fused, picked = backends[0].try_fused_route_quant, []

            def capture(*args, _fused=fused, **kwargs):
                result = _fused(*args, **kwargs)
                if result is not None:
                    picked.append(result.token_selected_experts.clone())
                return result

            backends[0].try_fused_route_quant = capture
            routed = forward(adaptive, x, counts)
            backends[0].try_fused_route_quant = fused
            record(False)
            torch.cuda.synchronize()
            if not picked:
                _assert_on_all_ranks(
                    f"{rows} rows: the adaptive layer did not take the fused route"
                )
            picks = torch.bincount(picked[0].reshape(-1).long(), minlength=_NUM_EXPERTS).float()
            problem = None
            for name, got in (("staged", staged), ("router", routed)):
                if not all(torch.equal(a, b) for a, b in zip(got, want)):
                    problem = f"{rows} rows: the {name} route's output differs from the layer without a balancer"
            if problem is None and not torch.equal(stats["rows"][0].cpu(), picks.cpu()):
                problem = f"{rows} rows: the router's statistics differ from the routing's picks"
            if problem is None and int(stats["forwards"][0].item()) != 1:
                problem = (
                    f"{rows} rows: the router counted {int(stats['forwards'][0].item())} forwards"
                )
            _assert_on_all_ranks(problem)

        # A runtime install rewrites a layer's route table row in place: a graph captured on the home
        # tables replays the replicated ones as an eager forward routes them (replica slots hold no
        # copied weights here, so the check is replay against eager, not against the plain layer).
        tables = balancer.route_tables(device)
        home_tables = tables.clone()
        replica_tables = torch.stack(
            [_replica_row(home_tables[i].tolist()) for i in range(_LAYERS)]
        ).to(device)
        use_router(True)
        for rows in (64, 1024):
            counts = [rows] * world_size
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            for _ in range(2):
                adaptive[0](x, all_rank_num_tokens=counts)
            torch.cuda.synchronize()
            graph, slot = torch.cuda.CUDAGraph(), None
            with torch.cuda.graph(graph):
                slot = adaptive[0](x, all_rank_num_tokens=counts).clone()
            tables.copy_(replica_tables)
            torch.cuda.synchronize()
            slot.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            eager = adaptive[0](x, all_rank_num_tokens=counts).clone()
            torch.cuda.synchronize()
            same = torch.equal(slot, eager)
            tables.copy_(home_tables)
            del graph
            _assert_on_all_ranks(
                None if same else f"{rows} rows: the replay after an install differs from eager"
            )

        # Kernels per layer call (torch profiler): the router's route launches no route op or staging.
        names = {}
        for arm, router in (("adaptive_staged", False), ("adaptive_router", True)):
            use_router(router)
            x = torch.randn((64, _HIDDEN_SIZE), dtype=torch.bfloat16)
            adaptive[0](x, all_rank_num_tokens=[64] * world_size)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                adaptive[0](x, all_rank_num_tokens=[64] * world_size)
                torch.cuda.synchronize()
            names[arm] = sorted(
                e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
            )
        staged_only = [
            n for n in names["adaptive_router"] if "StageInputs" in n or "adaptiveEplbRoute" in n
        ]
        _assert_on_all_ranks(
            None if not staged_only else f"the router's route still launches {staged_only}"
        )
        results.append(dict(rank=rank, kernels_per_layer_call=names))

        # Timed arms, interleaved per round; the router's route also on replicated tables (its route_us).
        arms = {
            "fused_no_balancer": (plain, None, None),
            "adaptive_staged": (adaptive, False, None),
            "adaptive_router": (adaptive, True, None),
            "adaptive_router_replicas": (adaptive, True, replica_tables),
        }
        for rows in (*_DECODE_ROWS, *_PREFILL_ROWS):
            counts = [rows] * world_size
            x = torch.randn((rows, _HIDDEN_SIZE), dtype=torch.bfloat16)
            for round_ in range(_ROUNDS):
                for arm, (layers, router, table) in arms.items():
                    if router is not None:
                        use_router(router)
                    tables.copy_(home_tables if table is None else table)
                    for _ in range(2):  # warm: compile and settle the bucket
                        layers[0](x, all_rank_num_tokens=counts)
                    torch.cuda.synchronize()
                    walls, hosts = [], []
                    for _ in range(_REPS):
                        MPI.COMM_WORLD.Barrier()
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        t0 = time.perf_counter()
                        for call in range(_CHAIN):
                            layers[call % _LAYERS](x, all_rank_num_tokens=counts)
                        host = time.perf_counter() - t0
                        end.record()
                        torch.cuda.synchronize()
                        walls.append(start.elapsed_time(end) * 1000.0 / _CHAIN)
                        hosts.append(host * 1e6 / _CHAIN)
                    results.append(
                        dict(
                            rank=rank,
                            rows=rows,
                            arm=arm,
                            round=round_,
                            wall_us_per_layer=statistics.median(walls),
                            host_us_per_layer=statistics.median(hosts),
                            experts=_NUM_EXPERTS,
                            slots=_NUM_SLOTS if layers is adaptive else _NUM_EXPERTS,
                            tables="home" if table is None else "replicas",
                        )
                    )
        tables.copy_(home_tables)
        use_router(True)
    return results


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason=f"Requires {_WORLD_SIZE} GPUs",
)
def test_megamoe_adaptive_route_matches_and_is_timed(record_property) -> None:
    if mpi_world_size() > 1:
        pytest.skip("starts its own ranks")
    with MPIPoolExecutor(
        max_workers=_WORLD_SIZE, initializer=_init_worker, initargs=(sys.path,)
    ) as executor:
        futures = [executor.submit(_run) for _ in range(_WORLD_SIZE)]
        rows = [row for future in futures for row in future.result()]
    record_property("adaptive_route_timings", json.dumps(rows))
    print("ADAPTIVE_ROUTE_TIMINGS " + json.dumps(rows))
