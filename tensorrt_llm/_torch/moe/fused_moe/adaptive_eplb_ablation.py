# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-only ablation of adaptive EPLB's decisions; not part of the feature.

The decision process (``adaptive_eplb/decision_main.py``) loads this file next to the package and
serves its rank through :func:`serve`. ``TLLM_ADAPTIVE_EPLB_ABLATION`` selects what changes:

* unset or empty: nothing; the runtime module is served as it is.
* ``even_shares``: every expert's rows split evenly over its copies instead of the solved shares.
  The installed shares are then even too, so solving them again never gains and no layer publishes
  new shares without a copy. Which layers install stays the product's rule: an install must repay
  its copy over the steps left, against the home layout too.
* ``install_all``: even shares, and every layer whose planned layout gains on this window installs
  it at once. There is no payback over the steps left, no comparison with the home layout net of
  the copy, no check on the previous window, no hold on a time model bucket that failed its
  calibration, and no single first install to time a copy.
* ``install_all_solved``: ``install_all``'s installs with the product's solved shares: a layer
  whose planned layout and solved shares gain installs them, and installed layouts whose solved
  shares gain again publish them without a copy, as the product refreshes its shares.
* ``freeze``: the product decides until its first install made with the copy price known (the
  product installs only its best layer while no copy has been timed), then nothing more: no
  further install and no new shares; the shares solved at that install stay.
* ``xfit``: the layout and shares in the ``.npz`` file ``TLLM_ADAPTIVE_EPLB_XFIT_LAYOUT`` names
  (``layout`` and ``shares``, ``[layers, num_slots]``, fitted on another workload) install at the
  first window, which also sizes the windows as the product's first window does, and never change.

Any other value stops the decision process with an error.
"""

import os
import sys

import numpy as np

VARIABLE = "TLLM_ADAPTIVE_EPLB_ABLATION"
MODES = ("", "even_shares", "install_all", "install_all_solved", "freeze", "xfit")
XFIT_LAYOUT = "TLLM_ADAPTIVE_EPLB_XFIT_LAYOUT"


def mode() -> str:
    """The ablation the environment selects; raises ``ValueError`` on an unknown one."""
    value = os.environ.get(VARIABLE, "")
    if value not in MODES:
        raise ValueError(f"{VARIABLE}={value!r} is not one of {MODES}")
    return value


def even_shares(load, layout, geometry, model=None) -> np.ndarray:
    """``[num_slots]`` shares that split every expert's rows evenly over its copies: the split
    the route tables make without shares, normalized per expert as solved shares are."""
    layout = np.asarray(layout)
    copies = np.bincount(layout[layout >= 0], minlength=geometry.num_experts)
    return np.where(layout >= 0, 1.0 / np.maximum(copies[np.maximum(layout, 0)], 1), 0.0)


def apply(runtime, value: str, patch=setattr) -> None:
    """Change the loaded ``runtime`` module for the ablation ``value`` (``patch`` sets one
    attribute; tests pass their own to undo it)."""
    if value not in MODES:
        raise ValueError(f"{VARIABLE}={value!r} is not one of {MODES}")
    if value in ("even_shares", "install_all"):
        patch(runtime, "solve_shares", even_shares)
    if value in ("install_all", "install_all_solved"):
        patch(runtime._Decider, "decide", decide_install_all)
    if value in ("freeze", "xfit"):
        patch(runtime._Decider, "decide", held(runtime._Decider.decide, value))


def held(decide, value):
    """``decide`` until the layouts are fixed, then decisions that change nothing: ``freeze``
    fixes them at the product's first install that copies, ``xfit`` at the file's layouts."""

    def decide_held(self, window: int):
        rt = sys.modules[type(self).__module__]
        if getattr(self, "ablation_fixed", False):
            update = rt.LayoutUpdate(window)
            self.awaiting_copy[window] = (self.open_call + self.forwards, None, None, 0)
            update.digest = update.fingerprint()
            return update
        if value == "xfit":
            return install_file(self, rt, window)
        priced = bool(self.copy_samples)  # the product: free to install every paying layer
        update = decide(self, window)
        self.ablation_fixed = value == "freeze" and priced and bool(update.moved)
        return update

    return decide_held


def install_file(self, rt, window: int):
    """Install the layouts and shares of the ``XFIT_LAYOUT`` file in every layer."""
    fitted = np.load(os.environ[XFIT_LAYOUT])
    layouts, shares = fitted["layout"].astype(np.int64), fitted["shares"].astype(np.float64)
    if layouts.shape != self.layout.shape or shares.shape != self.layout.shape:
        raise ValueError(f"{XFIT_LAYOUT}: shape {layouts.shape}, the runtime's {self.layout.shape}")
    cfg, geometry = self.config, self.config.geometry
    experts, cut = geometry.num_experts, len(layouts) * geometry.num_experts
    forwards = self.stats[:, cut : cut + len(layouts)].astype(np.float64)
    if forwards.min() >= 1:  # the product's window sizing: its rows per forward
        routed = self.stats[:, :cut].astype(np.float64).reshape(len(self.stats), len(layouts), -1)
        per_forward = float((routed / forwards[:, :, None]).sum(0).sum(1).mean())
        if per_forward > 0.0:
            self.next_forwards = rt.window_forwards(per_forward, experts, cfg.rows_per_expert)
    update, received = rt.LayoutUpdate(window), np.zeros(geometry.ep_size)
    for layer in range(len(layouts)):
        tables = rt.build_route_tables(layouts[layer], shares[layer], experts, layouts.shape[1])
        update.tables[layer] = np.array(tables.packed(), dtype=np.int32)
        fresh = (layouts[layer] >= 0) & (layouts[layer] != self.layout[layer])
        if fresh.any():
            update.moved.append(layer)
            received += fresh.reshape(geometry.ep_size, -1).sum(1)
            update.copy_items[layer] = rt.copy_items(
                self.layout[layer], layouts[layer], geometry, cfg.ep_rank, cfg.ranks_per_node
            )
    self.layout[:], self.shares[:], self.committed[:] = layouts, shares, window
    self.awaiting_copy[window] = (self.open_call + self.forwards, None, None, int(received.max()))
    self.ablation_fixed = True
    update.digest = update.fingerprint()
    return update


def serve(runtime, conn) -> None:
    """Serve one rank as ``runtime.serve`` does, under the ablation the environment selects."""
    value = mode()
    apply(runtime, value)
    if value:
        print(f"adaptive EPLB ablation: {VARIABLE}={value}", file=sys.stderr, flush=True)
    runtime.serve(conn)


def decide_install_all(self, window: int):
    """``_Decider.decide`` with every hold on an install removed (see the module docstring):
    the statistics, the rows priced ahead, the planned layouts and the bookkeeping of the copies
    are the product's; a layer installs exactly when its planned layout gains on this window."""
    rt = sys.modules[type(self).__module__]
    cfg, stats, geometry = self.config, self.stats, self.config.geometry
    layers, slots = self.layout.shape
    experts = geometry.num_experts
    close = self.open_call + self.forwards
    tail = {name: stats[:, layers * (experts + 2) + i] for i, name in enumerate(rt.FIELDS)}
    applied = [
        [(int(g), int(n), float(ms))] if g >= 0 else []
        for g, n, ms in zip(tail["applied"], tail["applied_layers"], tail["applied_ms"])
    ]
    self.install_timings.update((g, (n, ms)) for g, n, ms in rt.agreed_timings(applied))
    if (tail["stats_copy_ms"] > 0).all():
        self.stats_copies.append(float(tail["stats_copy_ms"].max()))
    update = rt.LayoutUpdate(window)
    self.awaiting_copy[window] = (close, None, None, 0)
    forwards = stats[:, layers * experts : layers * (experts + 1)].astype(np.float64)
    if forwards.min() < 1:
        return update
    routed = stats[:, : layers * experts].astype(np.float64).reshape(len(stats), layers, -1)
    rows = (routed / forwards[:, :, None]).sum(0)
    per_forward = float(rows.sum(1).mean())
    counted = float(routed.sum((0, 2)).mean())
    sizing = self.next_forwards is None and counted < cfg.rows_per_expert * experts
    if per_forward > 0.0:
        self.next_forwards = rt.window_forwards(per_forward, experts, cfg.rows_per_expert)
    if sizing:
        return update
    total = {name: float(values.astype(np.float64).sum()) for name, values in tail.items()}
    steps_left = None
    if min(total["accepted"], total["requests"]) > 0:
        per_step = total["accepted"] / total["requests"]
        steps_left = total["remaining"] / per_step / total["requests"]
    requests, before = total["requests"], self.close_requests
    change = abs(requests / before - 1.0) if min(requests, before) > 0.0 else 0.0
    self.close_requests, ahead = requests, None
    if change > self.largest_change:
        ahead = requests * min(requests / before, 1.0)
        rows = rows * (ahead / ((before + requests) / 2.0))
    self.largest_change = max(self.largest_change, change)
    requests = ahead or ((before + requests) / 2.0 if before > 0.0 else requests)
    applied_window = stats[:, layers * (experts + 1) : layers * (experts + 2)].min(0)
    planned = np.flatnonzero(self.committed <= applied_window)
    model = cfg.model.at(float((routed.sum(2) / forwards).mean(1).max()))
    self.evaluations.model = model
    home, installed = cfg.home_layouts[planned], self.layout[planned]
    new_layout = np.empty((len(planned), slots), dtype=np.int64)
    new_shares = np.empty((len(planned), slots))
    for i, layer in enumerate(planned.tolist()):
        plan = rt.plan_layout(rows[layer], geometry, home[i], installed[i], self.active)
        new_layout[i] = plan.layout
        new_shares[i] = rt.solve_shares(rows[layer], new_layout[i], geometry, model)
    current = (rows[planned], installed, self.shares[planned])
    gains = rt.evaluate_gains(*current, new_layout, new_shares, model, geometry)
    growth = self.evaluations.read_cost(
        range(layers), rows, self.layout, self.shares, self.open_call
    )
    self.kept.append((window, rows, requests))
    moved, received = [], np.zeros(geometry.ep_size)
    for i, layer in enumerate(planned.tolist()):
        if not gains[i] > 0.0:
            continue
        tables = rt.build_route_tables(new_layout[i], new_shares[i], experts, slots).packed()
        update.tables[layer] = np.array(tables, dtype=np.int32)
        fresh = (new_layout[i] >= 0) & (new_layout[i] != self.layout[layer])
        if fresh.any():
            moved.append(layer)
            received += fresh.reshape(geometry.ep_size, -1).sum(1)
            update.copy_items[layer] = rt.copy_items(
                self.layout[layer], new_layout[i], geometry, cfg.ep_rank, cfg.ranks_per_node
            )
        self.layout[layer] = new_layout[i]
        self.shares[layer] = new_shares[i]
        self.committed[layer] = window
    update.moved = sorted(moved)
    self.awaiting_copy[window] = (close, growth, steps_left, int(received.max()))
    update.digest = update.fingerprint()
    return update
