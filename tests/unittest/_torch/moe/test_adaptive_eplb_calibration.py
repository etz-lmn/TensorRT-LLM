# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of adaptive EPLB's warm-up calibration: designs, fit, acceptance, fallback, model."""

import numpy as np
import pytest

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import calibration as cal
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
    ExpertGeometry,
    TimeModel,
    solve_shares,
)

pytestmark = pytest.mark.cpu_only

G = cal.Geometry(ep=4, local_slots=8, homes=6, tokens=64, top_k=4, tile_rows=32)
TRUE = dict(
    per_active_slot_us=5.0, per_extra_tile_us=1.5, per_row_us=0.02, hot_slot_extra_tile_us=0.8
)
BUILT_IN = TimeModel(
    per_active_slot_us=4.0,
    per_extra_tile_us=1.0,
    per_row_us=0.01,
    hot_slot_extra_tile_us=0.5,
    scale=0.9,
    tile_rows=G.tile_rows,
)
CFG = cal.CalibrationConfig()


def truth(rows, remote_us=0.0, intercept=40.0):
    f = cal.features(rows, G)
    c = np.r_[[TRUE[k] for k in cal.FEATURES], remote_us]
    return intercept + (f @ c).max()


def times(designs, rng, remote_us=0.0, noise=0.1):
    return np.array([truth(d.rows, remote_us) for d in designs]) + rng.normal(0, noise, len(designs))


def test_designs_realise_their_rows():
    designs = cal.make_designs(G, CFG, (G.tokens,))
    assert {d.family for d in designs} == {"kernel", "locality"}
    rng = np.random.default_rng(0)
    for d in designs[:: len(designs) // 10]:
        for src in range(G.ep):
            slots = cal.realise(d.rows[src], G.tokens, G.top_k, rng)
            assert (np.sort(slots, 1)[:, 1:] != np.sort(slots, 1)[:, :-1]).all()
            assert (np.bincount(slots.ravel(), minlength=d.rows.shape[1]) == d.rows[src]).all()
    # locality designs: the same slot rows at rising rows from the receiving GPU itself
    loc = [d for d in designs if d.family == "locality"][: CFG.locality_levels]
    assert all((d.rows.sum(0) == loc[0].rows.sum(0)).all() for d in loc)
    remote = [cal.features(d.rows, G)[:, 4].sum() for d in loc]
    assert remote == sorted(remote, reverse=True) and remote[0] > remote[-1]


def test_fit_recovers_known_terms():
    designs = cal.make_designs(G, CFG, (G.tokens,))
    y = times(designs, np.random.default_rng(1))
    fit = cal.fit_bucket(G, designs, y, BUILT_IN, CFG)
    assert fit.accepted, fit.why
    for k, v in TRUE.items():
        assert fit.terms[k] == pytest.approx(v, rel=0.15, abs=0.05), k
    assert fit.terms["time_shares"] and fit.terms["scale"] == 1.0
    assert not fit.remote["resolved"]
    assert fit.holdout_rmse_us < fit.fallback_rmse_us


def test_remote_term_kept_only_when_resolved():
    designs = cal.make_designs(G, CFG, (G.tokens,))
    y = times(designs, np.random.default_rng(2), remote_us=0.05)
    fit = cal.fit_bucket(G, designs, y, BUILT_IN, CFG)
    assert fit.accepted and fit.remote["resolved"]
    assert fit.remote["per_remote_row_us"] == pytest.approx(0.05, rel=0.3)
    # folded into the per-row cost at the route op's even spread: (ep - 1) / ep of the rows remote
    assert fit.terms["per_row_us"] == pytest.approx(0.02 + 0.05 * 3 / 4, rel=0.15)


def test_a_bad_fit_falls_back(monkeypatch):
    designs = cal.make_designs(G, CFG, (G.tokens,))
    y = times(designs, np.random.default_rng(3))
    monkeypatch.setattr(cal, "fit_max", lambda f, y, start, it: np.r_[0.0, 50.0 * np.ones(f.shape[-1])])
    fit = cal.fit_bucket(G, designs, y, BUILT_IN, CFG)
    assert not fit.accepted and "worse than the built-in" in fit.why
    y[0] = np.nan
    monkeypatch.undo()
    assert not cal.fit_bucket(G, designs, y, BUILT_IN, CFG).accepted
    table = cal.make_table([cal.fit_bucket(G, designs, y, BUILT_IN, CFG)], {})
    assert cal.time_model_of(table, BUILT_IN) is BUILT_IN


def _fake_launcher(route_us_true=0.0, fail_vote=False):
    """The launches of rank 0 of an EP group whose every rank sends what rank 0 sends."""
    designs = {}
    for t in (G.tokens,):
        rng = np.random.default_rng([CFG.seed, 2, 0, t])
        for d in cal.make_designs(G, CFG, (t,)):
            designs[cal.realise(d.rows[0], G.tokens, G.top_k, rng).tobytes()] = d

    def gather(obj):
        if fail_vote and isinstance(obj, tuple) and len(obj) == 2 and obj[1] is not None:
            return [obj, ("ValueError: rank 1", None)]
        return [obj] * G.ep

    def kernel(t, slots):
        return truth(designs[np.asarray(slots).tobytes()].rows)

    def route(t, experts, table):
        e, s = G.ep * G.homes, G.ep * G.local_slots
        counts, starts, slots = table[:e], table[e : 2 * e], table[2 * e : 2 * e + s]
        out = np.empty_like(experts)
        seen = np.zeros(e, int)
        for i, x in np.ndenumerate(experts):
            out[i] = slots[starts[x] + seen[x] % counts[x]]
            seen[x] += 1
        rows = np.stack([np.bincount(out.ravel(), minlength=s)] * G.ep)
        replicas = (counts > 1).any()
        return truth(rows) + 3.0 + route_us_true * replicas, out

    return cal.Launcher(kernel, route, gather)


@pytest.mark.parametrize("route_us", [0.0, 2.0])
def test_calibrate_fits_route_us_and_seals_a_model(route_us):
    model, table, why = cal.calibrate(
        lambda t: G._replace(tokens=t), [G.tokens], 0, _fake_launcher(route_us), BUILT_IN, {}, CFG
    )
    assert table is not None and not why
    (bucket,) = table["buckets"]
    assert bucket["accepted"]
    assert bucket["route"]["route_us"] == pytest.approx(route_us, abs=0.3)
    assert table == cal.seal(table)
    at = model.at(G.tokens * G.top_k)
    assert at.time_shares and at.scale == 1.0 and at.buckets == ()
    assert at.per_active_slot_us == pytest.approx(TRUE["per_active_slot_us"], rel=0.15)


def test_calibrate_falls_back_on_any_rank_failure():
    model, table, why = cal.calibrate(
        lambda t: G._replace(tokens=t), [G.tokens], 0, _fake_launcher(fail_vote=True), BUILT_IN, {}, CFG
    )
    assert model is BUILT_IN and table is None and "rank 1" in why


def test_policy_uses_time_shares_with_the_fitted_model():
    designs = cal.make_designs(G, CFG, (G.tokens,))
    fit = cal.fit_bucket(G, designs, times(designs, np.random.default_rng(4)), BUILT_IN, CFG)
    model = cal.time_model_of(cal.make_table([fit], {}), BUILT_IN)
    assert not BUILT_IN.at(G.tokens * G.top_k).time_shares
    small, large = model.at(1.0), model.at(1e9)  # one bucket: every size pads to it or beyond
    assert small.time_shares and large.time_shares and small.per_row_us == fit.terms["per_row_us"]
    torch = pytest.importorskip("torch")
    if not hasattr(torch.ops.trtllm, "adaptive_eplb_solve_shares"):
        pytest.skip("no adaptive EPLB op library")
    geometry = ExpertGeometry(num_experts=G.ep * G.homes, num_slots=G.ep * G.local_slots, ep_size=G.ep)
    layout = geometry.home_layout()
    layout[G.local_slots + G.homes] = 0  # expert 0's replicas on GPU 1's and GPU 2's first spare slot
    layout[2 * G.local_slots + G.homes] = 0
    load = np.full(geometry.num_experts, 8.0)
    load[0] = 600.0
    by_rows = solve_shares(load, layout, geometry)
    by_time = solve_shares(load, layout, geometry, small)
    assert by_rows.shape == by_time.shape == (geometry.num_slots,)


def test_a_gain_inside_its_standard_error_installs_nothing():
    from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import (
        gain_standard_error,
        resolved_gains,
    )

    designs = cal.make_designs(G, CFG, (G.tokens,))
    fit = cal.fit_bucket(G, designs, times(designs, np.random.default_rng(5), noise=2.0), BUILT_IN, CFG)
    model = cal.time_model_of(cal.make_table([fit], {}), BUILT_IN).at(1.0)
    assert len(model.covariance) == 25 and model.gain_z == CFG.gain_z
    geometry = ExpertGeometry(num_experts=G.ep * G.homes, num_slots=G.ep * G.local_slots, ep_size=G.ep)
    home = geometry.home_layout()
    new = home.copy()
    new[G.local_slots + G.homes] = 0
    load = np.full((2, geometry.num_experts), 8.0)
    load[1, 0] = 600.0  # layer 1: expert 0 is hot
    shares = (home >= 0).astype(float)
    new_shares = (new >= 0).astype(float)
    layouts = np.stack([home, home]), np.stack([new, new])
    se = gain_standard_error(load, layouts[0], np.stack([shares] * 2), layouts[1], np.stack([new_shares] * 2), model, geometry)
    assert se.shape == (2,) and (se >= 0).all()
    gains = np.array([0.5 * model.gain_z * se[0], 2.0 * model.gain_z * se[1] + 1e-9])
    assert (resolved_gains(gains, se, model) == np.array([0.0, gains[1]])).all()
    assert gain_standard_error(load, *layouts[:1], np.stack([shares] * 2), layouts[1], np.stack([new_shares] * 2), BUILT_IN, geometry) is None
