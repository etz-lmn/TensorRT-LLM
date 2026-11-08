# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive EPLB load balancer and its per-layer handle.

Statistics and slot routing run inside the forward, so CUDA graphs capture them and no host work
is needed per step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.distributed as dist

from ..moe_load_balancer import BaseMoeLoadBalancer
from .policy import EMA_DECAY, ExpertGeometry, TimeModel
from .runtime import AdaptiveRuntime, DecisionConfig, DecisionProcess, build_route_tables
from .transport import PeerSlotTransport, migrate_model_weights

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping


def statistics_buffers(layers: int, num_experts: int, device: torch.device | str) -> dict:
    """Every layer's statistics in contiguous buffers: ``ema`` ``[L, E]`` and ``forwards`` ``[L, 1]``."""
    return dict(
        ema=torch.zeros(layers, num_experts, dtype=torch.float32, device=device),
        forwards=torch.zeros(layers, 1, dtype=torch.int64, device=device),
    )


class ExpertStatistics(NamedTuple):
    """One layer's statistics in the ops' argument order, and the ``[1]`` flag that the ops read at
    every launch, so that a captured CUDA graph records statistics only while the flag is set."""

    ema: torch.Tensor
    forwards: torch.Tensor
    enabled: torch.Tensor

    @classmethod
    def of(cls, buffers: dict, layer: int, enabled: torch.Tensor) -> ExpertStatistics:
        """The statistics of layer ``layer`` in buffers made by :func:`statistics_buffers`."""
        return cls(buffers["ema"][layer], buffers["forwards"][layer], enabled)

    def record(self, token_selected_experts: torch.Tensor, first_call: bool = True) -> None:
        """Add ``[tokens, top_k]`` expert ids to the statistics; the later calls of a forward add to
        its first call's counts."""
        torch.ops.trtllm.adaptive_eplb_record(token_selected_experts, *self, EMA_DECAY, first_call)


class AdaptiveMoeLayerLoadBalancer:
    """Handle of one MoE layer (a :class:`~..moe_load_balancer.MoeLayerLoadBalancer`): records its
    routes and maps experts to slots on the device, inside CUDA graphs. Its slot weights move whole
    after loading, so none of the host weight hooks of standard EPLB apply."""

    produces_replica_plan = False
    stages_routes = True

    def __init__(
        self,
        manager: AdaptiveMoeLoadBalancer,
        *,
        layer_idx: int,
        expert_count: int,
        slot_count_per_rank: int,
        repeated_count: int = 1,
    ) -> None:
        self.manager = manager
        self.layer_idx = int(layer_idx)
        self.expert_count = int(expert_count)
        self.num_slots = int(slot_count_per_rank) * manager.ep_size
        self.repeated_count = int(repeated_count)
        self.statistics: ExpertStatistics | None = None
        self._tables: list[torch.Tensor] = []
        self._slot_dtype = torch.int32
        self._route_staging: Callable[[int, int], object | None] | None = None
        self._pending: tuple[torch.Tensor, bool] | None = None
        self._route_into = None

    def get_repeat_count(self) -> int:
        return self.repeated_count

    def is_dynamic_routing(self) -> bool:
        """False: no host statistics, host weight migration or stage hooks."""
        return False

    def records_statistics(self) -> bool:
        """True: the MoE modules pass every forward's routed experts to the statistics hooks."""
        return True

    def need_load_shared_weights(self) -> bool:
        return False

    def get_load_expert_ids(self) -> list[int]:
        """The experts this rank loads from the checkpoint: its home experts."""
        local = self.expert_count // self.manager.ep_size
        start = self.manager.ep_rank * local
        return list(range(start, start + local))

    def home_layout(self) -> list[int]:
        """Every expert on its home slot (:meth:`~.policy.ExpertGeometry.home_layout`)."""
        geometry = ExpertGeometry(self.expert_count, self.num_slots, self.manager.ep_size)
        return geometry.home_layout().tolist()

    def set_initial_weight_assignments(self, initial_weight_assignments: Sequence[int]) -> None:
        """Raise ``ValueError`` unless the assignments cover the slots and keep the home experts."""
        assignments = [int(expert) for expert in initial_weight_assignments]
        if len(assignments) != self.num_slots:
            raise ValueError(f"expected {self.num_slots} slot assignments, got {len(assignments)}")
        if set(assignments) - set(range(self.expert_count)):
            raise ValueError("slot assignments name an expert the layer does not have")
        homes = self.home_layout()
        if any(home >= 0 and home != expert for home, expert in zip(homes, assignments)):
            raise ValueError("slot assignments do not hold every expert on its home slot")

    def prepare_device_state(
        self,
        slot_dtype: torch.dtype,
        device: torch.device | str,
        route_staging: Callable[[int, int], object | None] | None = None,
    ) -> None:
        """Bind the layer's statistics and route tables on ``device`` before the first forward.

        ``slot_dtype`` is the dtype the backend reads slot ids in and ``route_staging`` its
        ``route_staging`` method, if it has one.
        """
        manager = self.manager
        flag = manager.recording_flag(device)
        self.statistics = ExpertStatistics.of(manager.statistics(device), self.layer_idx, flag)
        tables = manager.route_tables(device)[self.layer_idx]
        experts = self.expert_count
        slots = self.num_slots
        self._tables = tables.split([experts, experts, slots, slots])
        self._slot_dtype = slot_dtype
        self._route_staging = route_staging
        self._route_into = torch.ops.trtllm.adaptive_eplb_route_into

    def update_local_statistic(
        self,
        local_raw_expert_ids: torch.Tensor,
        is_first_stage: bool,
        is_last_stage: bool,
        allreduce: object = None,
    ) -> None:
        """Keep the routes of a chunk or repeat; the next :meth:`route` of the same ids adds them to
        the statistics.

        Forwards that are not recorded (context steps, warmup) record nothing, except under CUDA
        graph capture, where the device flag gates every replay.
        """
        statistics = self.statistics
        pending = self._pending
        self._pending = None
        if pending is not None:
            statistics.record(*pending)
        if self.manager.recording or (
            statistics.ema.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            self._pending = (local_raw_expert_ids, is_first_stage)

    update_statistic_with_local_ids = update_local_statistic

    def route(
        self,
        token_selected_experts: torch.Tensor,
        offset_by_ep_rank: bool = False,
        *,
        overlap_fn: Callable[[], None] | None = None,
        token_final_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map every routed expert to a slot, in one launch that also adds the kept routes to the
        statistics.

        Ids outside the experts map to ``num_slots``. With ``token_final_scales`` (``[tokens,
        top_k]`` float32) and a backend that stages routes, the launch writes the slots and weights
        into the backend's staging and returns a view of it, which the backend does not stage again.
        """
        ids = token_selected_experts
        pending = self._pending
        self._pending = None
        if pending is not None and pending[0] is not ids:
            self.statistics.record(*pending)
            pending = None
        staging = None
        if (
            self._route_staging is not None
            and token_final_scales is not None
            and token_final_scales.dtype == torch.float32
            and token_final_scales.is_contiguous()
            and token_final_scales.shape == ids.shape
        ):
            staging = self._route_staging(*ids.shape)
        if staging is None:
            out = torch.empty_like(ids, dtype=self._slot_dtype)
            weights = (None, None)
        else:
            out = staging.topk_idx
            weights = (token_final_scales, staging.topk_weights)
        if pending is None:
            statistics_args = (None, None, None, 0.0, True)
        else:
            statistics_args = (*self.statistics, EMA_DECAY, pending[1])
        self._route_into(ids, *self._tables, self.manager.ep_rank, out, *weights, *statistics_args)
        if overlap_fn is not None:
            overlap_fn()
        return out[: ids.shape[0]]


class AdaptiveMoeLoadBalancer(BaseMoeLoadBalancer):
    """Model-level adaptive EPLB balancer of rank ``ep_rank`` of ``ep_size``.

    Statistics and routes live on the device, so a forward outside ``MoeLoadBalancerIterContext``
    (the first replay after a graph capture) is recorded like any other.
    """

    def __init__(
        self, *, ep_rank: int, ep_size: int, config: MoeAdaptiveLoadBalancerConfig, mapping: Mapping
    ) -> None:
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.config = config
        self.mapping = mapping
        self.num_slots = int(config.num_slots)
        self.num_local_slots = self.num_slots // self.ep_size
        self.layers: list[AdaptiveMoeLayerLoadBalancer] = []
        self.next_layer_repeated_count: int | None = None
        self.in_iter = False
        self.is_shutdown = False
        self._previous_balancer: BaseMoeLoadBalancer | None = None
        self.enable_statistic = False
        self.enable_update_weights = False
        self.transport: PeerSlotTransport | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self.runtime: AdaptiveRuntime | None = None
        # Whether the forwards now running record their routes.
        self.recording = False
        self._recording_flag: torch.Tensor | None = None
        self._route_tables: torch.Tensor | None = None
        self._statistics: dict[str, torch.Tensor] | None = None

    def is_static_routing(self) -> bool:
        return False

    def set_repeated_for_next_layer(self, repeated_count: int) -> None:
        if repeated_count <= 0:
            raise ValueError("repeat count must be positive")
        self.next_layer_repeated_count = int(repeated_count)

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> AdaptiveMoeLayerLoadBalancer:
        """Register one MoE layer and return its handle; ``ValueError`` if it does not fit the config."""
        self.config.validate_expert_capacity(expert_count)
        if int(slot_count_per_rank) != self.num_local_slots:
            raise ValueError(
                f"layer slot count {slot_count_per_rank} differs from the configured "
                f"{self.num_local_slots} local slots"
            )
        layer = AdaptiveMoeLayerLoadBalancer(
            self,
            layer_idx=len(self.layers),
            expert_count=expert_count,
            slot_count_per_rank=slot_count_per_rank,
            repeated_count=self.next_layer_repeated_count or 1,
        )
        self.next_layer_repeated_count = None
        self.layers.append(layer)
        return layer

    def _nothing_before_runtime(self, *args: object, **kwargs: object) -> None:
        """Nothing to do: the slot weights move in :meth:`prepare_runtime`."""

    register_weight_slots_after_to_cuda = _nothing_before_runtime
    finalize_model = _nothing_before_runtime
    set_warm_up_iter_count = _nothing_before_runtime

    def prepare_runtime(self, model: torch.nn.Module) -> None:
        """Move the slot weights into MNNVL memory and start the runtime under the backend's time
        model. Collective over EP; runs before the KV cache is sized."""
        self.transport, self.copy_stream, backends = migrate_model_weights(
            model, self.layers, self.mapping
        )
        first = self.layers[0].layer_idx
        time_model = backends[first].decode_time_model(
            self.transport.expert_bytes(first), torch.cuda.get_device_name()
        )
        self.start_runtime(TimeModel(**time_model), self.mapping.gpus_per_node)

    def start_runtime(self, model: TimeModel, ranks_per_node: int) -> None:
        """Start the windows, the decision process and the installs. Collective over EP."""
        if not dist.is_initialized():
            raise RuntimeError(
                "adaptive EPLB needs torch.distributed: its statistics gathers run on a gloo group "
                "of the EP ranks, and a MoE backend that supports it initializes the default group"
            )
        geometry = ExpertGeometry(self.layers[0].expert_count, self.num_slots, self.ep_size)
        homes = np.array([layer.home_layout() for layer in self.layers], dtype=np.int64)
        decision = DecisionProcess(
            DecisionConfig(geometry, model, homes, self.ep_rank, ranks_per_node)
        )
        # The runtime's worker thread gathers on a group of its own: a group the main thread also
        # uses would interleave their collectives differently on different ranks. Its window and
        # copy gathers alternate on every rank, so one group carries both.
        ranks = sorted(self.transport.allgather(dist.get_rank()))
        group = dist.new_group(ranks, backend="gloo", use_local_synchronization=True)

        def gather(send: torch.Tensor, rows: list[torch.Tensor]) -> None:
            dist.all_gather(rows, send, group=group)

        device = torch.device("cuda", torch.cuda.current_device())
        self.runtime = AdaptiveRuntime(
            self.layers,
            self.transport,
            self.copy_stream,
            decision,
            gather,
            device,
            self.route_tables(device),
            self.statistics(device),
        )

    def statistics(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Every layer's statistics in contiguous buffers (:func:`statistics_buffers`)."""
        if self._statistics is None:
            self._statistics = statistics_buffers(
                len(self.layers), self.layers[0].expert_count, device
            )
        return self._statistics

    def route_tables(self, device: torch.device | str) -> torch.Tensor:
        """The route tables of all layers, one int32 row of ``2 * (experts + slots)`` per layer.

        Every layer starts with the table of its home layout.
        """
        if self._route_tables is None:
            layer = self.layers[0]
            home = build_route_tables(layer.home_layout(), None, layer.expert_count, self.num_slots)
            self._route_tables = torch.tensor(
                [home.packed()] * len(self.layers), dtype=torch.int32, device=device
            )
        return self._route_tables

    def recording_flag(self, device: torch.device | str) -> torch.Tensor:
        """The ``[1]`` int32 flag the statistics read: set for generation steps outside warmup."""
        if self._recording_flag is None:
            self._recording_flag = torch.zeros(1, dtype=torch.int32, device=device)
        return self._recording_flag

    def observe_scheduled_batch(self, scheduled_requests: object, is_warmup: bool) -> None:
        """Record the statistics of generation steps outside warmup only, and pass the generation
        requests to the runtime when it asks for their progress."""
        generation = scheduled_requests.generation_requests
        recording = bool(generation) and not scheduled_requests.context_requests and not is_warmup
        if recording != self.recording:
            self.recording = recording
            if self._recording_flag is not None:
                self._recording_flag.fill_(int(recording))
        if self.runtime is not None and self.runtime.wants_request_progress:
            self.runtime.add_request_progress(generation)

    def start_iter(self) -> None:
        if self.in_iter:
            raise RuntimeError("already in a MoE load-balancer iteration")
        self.in_iter = True
        if self.runtime is not None and self.enable_update_weights:
            if not torch.cuda.is_current_stream_capturing():
                self.runtime.boundary()

    def end_iter(self) -> None:
        if not self.in_iter:
            raise RuntimeError("not in a MoE load-balancer iteration")
        self.in_iter = False

    def shutdown(self, local_safe: bool = True) -> bool:
        """Stop the runtime once every EP rank is safe, else return ``False``. Collective over EP."""
        if self.is_shutdown:
            return True
        local_safe = bool(local_safe and not self.in_iter)
        if self.transport is not None:
            local_safe = all(self.transport.allgather(local_safe))
        if not local_safe:
            return False
        if self.runtime is not None:
            self.runtime.close()
        if self.transport is not None:
            self.copy_stream.synchronize()
        self.is_shutdown = True
        return True
