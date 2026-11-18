# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-run calibration of adaptive EPLB's decode time model (:class:`~.policy.TimeModel`).

Once per process, before the first decode forward, synthetic slot loads are launched through the
real MoE backend of the first balanced layer at every decode bucket of the deployment, and the
model's terms are fitted to the times:

* kernel designs: random slot loads ``x [ep, local slots]`` (rows each GPU's slots receive), each
  realised as physical slot ids of ``tokens`` tokens per source rank and ``top_k`` distinct slots per
  token; every source sends every slot an equal part of its rows. A GPU's features are its active
  slots, tiles beyond a slot's first, rows, tiles beyond the first of its busiest slot, and rows
  from other GPUs (remote rows); a design takes ``a + max over GPUs of c . features``.
* locality designs: the same loads at several fractions of rows sent from the receiving GPU itself,
  so the remote rows vary at fixed loads. The remote-row term is kept only when it predicts the
  held-out designs better than the fit without it, by more than ``z`` standard errors.
* route designs: rows on the home slots alone, routed through the layer's route op on its home table
  and on a table with replicas; ``route_us`` is the paired extra over the kernel terms' prediction,
  kept when above ``z`` standard errors.

A bucket's fit is accepted when, on every ``holdout_every``-th design held out of the fit, it is not
worse than the built-in model (refitted to a scale and an intercept on the same designs) by more
than ``z`` standard errors; else that bucket keeps the built-in model. Every rank computes the
same sealed table; when any check fails on any rank before the first launch, or the ranks' tables
differ, every rank keeps the built-in model.

numpy only at import: the fit and the designs run without a GPU (the tests drive them with a model
of the times); :func:`calibrate` reaches torch and the backend through the arguments it is given.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import NamedTuple

import numpy as np

from .policy import TimeModel

FEATURES = ("per_active_slot_us", "per_extra_tile_us", "per_row_us", "hot_slot_extra_tile_us")
REMOTE = "per_remote_row_us"
TABLE_SCHEMA = "adaptive_eplb_warmup_calibration"
ATTEMPTS = 100  # draws per design before a geometry counts as unable to hold the designs


@dataclass(frozen=True)
class CalibrationConfig:
    """Sizes and acceptance rules of the warm-up calibration. None depends on a model or workload.

    * ``kernel_designs``, ``route_designs``, ``locality_designs`` x ``locality_levels``: designs per
      bucket. 96 kernel designs hold out 24, as many as the offline calibration
      (``measure_adaptive_eplb_time_model.py``: 128 designs, every fourth held out) at three fourths
      of its launches.
    * ``holdout_every``: every n-th design is held out of the fit (the offline calibration's 4).
    * ``z``: standard errors a hold-out difference must exceed (the offline calibration refuses a
      bucket whose hold-out scale differs by more than 3 standard errors).
    * ``warm``, ``timed``, ``reps``: launches before timing, launches per timing, passes.
    * ``max_iterations``: alternations of busiest-GPU assignment and least squares in one fit.
    * ``gain_z``: standard errors of a predicted gain (new layout minus current, from the bucket's
      coefficient covariance) the gain must exceed for the layer to install (:attr:`TimeModel.gain_z`);
      the same 3 standard errors as ``z``.
    """

    enabled: bool = True
    kernel_designs: int = 96
    locality_designs: int = 8
    locality_levels: int = 4
    route_designs: int = 8
    holdout_every: int = 4
    z: float = 3.0
    warm: int = 2
    timed: int = 8
    reps: int = 2
    max_iterations: int = 50
    seed: int = 0
    gain_z: float = 3.0


class Geometry(NamedTuple):
    ep: int
    local_slots: int
    homes: int  # experts per rank (its home slots come first)
    tokens: int  # per source rank and forward
    top_k: int
    tile_rows: int


class Design(NamedTuple):
    family: str  # 'kernel' | 'locality' | 'route'
    rows: np.ndarray  # [ep sources, ep * local slots] int64 rows each source sends each slot


# --- designs ---


def _deal(x: np.ndarray, ep: int, offset: int = 0) -> np.ndarray:
    """``[ep, slots]`` rows per source of slot rows ``x`` whose total is a multiple of ``ep``: the
    rows in slot order dealt to the sources in turn, so every source sends the same number."""
    units = np.repeat(np.arange(x.size), x)
    out = np.zeros((ep, x.size), dtype=np.int64)
    np.add.at(out, ((np.arange(units.size) + offset) % ep, units), 1)
    return out


def _random_loads(g: Geometry, rng: np.random.Generator, balanced: bool) -> np.ndarray | None:
    """``[ep, local slots]`` slot rows summing to every source's ``tokens x top_k`` routes, no slot
    above ``ep x tokens`` (a token routes to a slot once); ``balanced``: every GPU receives what it
    sends (each slot at most ``tokens``), else random rows per GPU, concentration and hot slots."""
    ep, m, t, k, tile = g.ep, g.local_slots, g.tokens, g.top_k, g.tile_rows
    cap = t if balanced else ep * t
    gpus = np.full(ep, t * k) if balanced else None
    hot = np.zeros((ep, m))
    if not balanced:
        for r in rng.choice(ep, int(rng.integers(0, ep // 4 + 2)), replace=False):
            top = min(cap, tile * int(rng.integers(1, cap // tile + 2)))
            hot[r, 0] = max(top - int(rng.integers(tile // 4 + 1)), 1)
    w = rng.lognormal(0.0, rng.uniform(0.0, 1.5), (ep, m)) * (hot == 0)
    w *= np.arange(m) < rng.integers(1, m + 1, (ep, 1))
    if not balanced:
        w *= rng.lognormal(0.0, 0.7, (ep, 1))
    x = hot.copy()
    if gpus is None:
        rest = int(ep * t * k - hot.sum())
        if rest < 0 or w.sum() <= 0:
            return None
        x += rng.multinomial(rest, (w / w.sum()).ravel()).reshape(ep, m)
    else:  # at least top_k active slots per GPU; rows above the cap move to its other active slots
        for r in range(ep):
            on = np.zeros(m, bool)
            on[rng.choice(m, int(rng.integers(min(k, m), m + 1)), replace=False)] = True
            p = np.where(on, np.maximum(w[r], 1e-3), 0.0)
            row = rng.multinomial(int(gpus[r]), p / p.sum())
            while row.max() > cap:
                over = row > cap
                spill = int((row[over] - cap).sum())
                row[over] = cap
                room = np.where(on & (row < cap), cap - row, 0)
                row += rng.multinomial(spill, room / room.sum())
            x[r] = row
    if x.max() > cap:
        return None
    return x.astype(np.int64)


def _locality(x: np.ndarray, g: Geometry, fraction: float) -> np.ndarray:
    """Rows per source of balanced loads ``x`` with ``fraction`` of every GPU's routes sent to its own
    slots (largest remainder over its slots), the rest dealt in turn to every source."""
    ep, m = x.shape
    local = int(round(fraction * g.tokens * g.top_k))
    out = np.zeros((ep, ep * m), dtype=np.int64)
    for r in range(ep):
        want = x[r] * (local / max(int(x[r].sum()), 1))
        own = np.floor(want).astype(np.int64)
        short = local - int(own.sum())
        if short > 0:
            own[np.argsort(own - want, kind="stable")[:short]] += 1
        out[r, r * m : (r + 1) * m] = np.minimum(own, x[r])
    rest = x.reshape(-1) - out.sum(0)
    out += _deal(rest, ep)  # equal parts: every source kept the same local rows
    return out


def _draws(n: int, draw: Callable[[], np.ndarray | None]) -> list[np.ndarray]:
    """``n`` accepted draws; raises when the geometry rejects nearly every draw."""
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > ATTEMPTS * max(n, 1):
            raise ValueError(f"only {len(out)} of {n} designs in {tries - 1} draws")
        x = draw()
        if x is not None:
            out.append(x)
    return out


def make_designs(g: Geometry, cfg: CalibrationConfig, seed: Sequence[int]) -> list[Design]:
    """The kernel and locality designs of one bucket, the same on every rank for the same seed.
    Locality designs need every GPU's routes to fit its own slots (``top_k <= local slots``)."""
    rng = np.random.default_rng([cfg.seed, *seed, *g])
    xs = _draws(cfg.kernel_designs, lambda: _random_loads(g, rng, balanced=False))
    out = [Design("kernel", _deal(x.reshape(-1), g.ep)) for x in xs]
    if g.top_k <= g.local_slots:
        fractions = np.linspace(0.0, 1.0, cfg.locality_levels)
        for x in _draws(cfg.locality_designs, lambda: _random_loads(g, rng, balanced=True)):
            out.extend(Design("locality", _locality(x, g, f)) for f in fractions)
    return out


def route_loads(g: Geometry, cfg: CalibrationConfig, seed: Sequence[int]) -> list[np.ndarray]:
    """``[ep, homes]`` expert rows of the route designs: rows on the home slots alone."""
    rng = np.random.default_rng([cfg.seed, 1, *seed, *g])
    home = g._replace(local_slots=g.homes)
    return _draws(cfg.route_designs, lambda: _random_loads(home, rng, balanced=False))


def realise(rows: np.ndarray, tokens: int, top_k: int, rng: np.random.Generator) -> np.ndarray:
    """``[tokens, top_k]`` slot ids of one source sending ``rows[s]`` rows to slot ``s``, every token
    to ``top_k`` distinct slots: largest slot first, each to the tokens with the most routes left
    (always possible when every slot takes at most ``tokens`` rows and they sum to tokens x top_k)."""
    if rows.sum() != tokens * top_k or rows.max(initial=0) > tokens:
        raise ValueError("the rows are not routes of distinct slots")
    left = np.full(tokens, top_k)
    out = np.empty((tokens, top_k), dtype=np.int64)
    fill = np.zeros(tokens, dtype=np.int64)
    for s in np.argsort(-rows, kind="stable"):
        n = int(rows[s])
        if n == 0:
            break
        pick = np.argpartition(-(left + 0.5 * rng.random(tokens)), n - 1)[:n]
        out[pick, fill[pick]] = s
        fill[pick] += 1
        left[pick] -= 1
    return out


def features(rows: np.ndarray, g: Geometry) -> np.ndarray:
    """``[ep, 5]`` per GPU: active slots, tiles beyond each slot's first, rows, tiles beyond the
    first of its busiest slot, rows from other GPUs (``rows`` per source and slot)."""
    ep = rows.shape[0]
    x = rows.sum(0).reshape(ep, -1).astype(np.float64)
    tiles = np.ceil(x / g.tile_rows)
    own = np.array([rows[r].reshape(ep, -1)[r].sum() for r in range(ep)], dtype=np.float64)
    return np.stack(
        [
            (x > 0).sum(1),
            np.maximum(tiles - 1.0, 0.0).sum(1),
            x.sum(1),
            np.maximum(tiles.max(1) - 1.0, 0.0),
            x.sum(1) - own,
        ],
        1,
    )


# --- fit ---


def nnls(a: np.ndarray, b: np.ndarray, free: int = 0) -> np.ndarray:
    """Least squares with every coefficient but the first ``free`` non-negative (Lawson-Hanson)."""
    n = a.shape[1]
    sign = np.r_[np.ones(free, bool), np.zeros(n - free, bool)]
    active = sign.copy()
    w = np.zeros(n)
    for _ in range(3 * n):
        if active.any():
            z = np.zeros(n)
            z[active] = np.linalg.lstsq(a[:, active], b, rcond=None)[0]
            bad = active & ~sign & (z < 0)
            while bad.any():
                step = np.min(w[bad] / np.maximum(w[bad] - z[bad], 1e-300))
                w = w + step * (z - w)
                active &= sign | (w > 1e-12)
                z = np.zeros(n)
                if active.any():
                    z[active] = np.linalg.lstsq(a[:, active], b, rcond=None)[0]
                bad = active & ~sign & (z < 0)
            w = z
        grad = a.T @ (b - a @ w)
        grad[active] = 0.0
        if grad.max(initial=0.0) <= 1e-9 * max(1.0, float(np.abs(b).max(initial=0.0))):
            break
        active[int(np.argmax(grad))] = True
    return w


def fit_max(f: np.ndarray, y: np.ndarray, start: np.ndarray, iterations: int) -> np.ndarray:
    """``[1 + d]`` intercept and non-negative costs of ``y ~ a + max_g f[:, g] . c`` (``f [D, ep, d]``):
    the busiest GPU of every design under the current costs, least squares on those, until the
    busiest GPUs repeat."""
    c, seen, idx = np.asarray(start, dtype=np.float64), set(), np.arange(len(y))
    w = np.r_[0.0, c]
    for _ in range(iterations):
        busiest = np.argmax(f @ c, 1)
        if busiest.tobytes() in seen:
            break
        seen.add(busiest.tobytes())
        a = np.concatenate([np.ones((len(y), 1)), f[idx, busiest]], 1)
        w = nnls(a, y, free=1)
        c = w[1:]
    return w


def covariance(f: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``[d, d]`` covariance of the costs ``w[1:]`` of :func:`fit_max`'s fit: the least squares on the
    busiest GPUs at ``w``, the residual variance times the inverse normal matrix of the coefficients
    the fit left free (a cost the non-negativity pinned at 0 is held there: no variance)."""
    busiest = np.argmax(f @ w[1:], 1)
    a = np.concatenate([np.ones((len(y), 1)), f[np.arange(len(y)), busiest]], 1)
    free = np.r_[True, w[1:] > 0.0]
    dof = len(y) - int(free.sum())
    out = np.zeros((a.shape[1], a.shape[1]))
    if dof > 0:
        resid = y - a @ w
        sub = a[:, free]
        out[np.ix_(free, free)] = float(resid @ resid) / dof * np.linalg.pinv(sub.T @ sub)
    return out[1:, 1:]


def predict(w: np.ndarray, f: np.ndarray) -> np.ndarray:
    return w[0] + (f @ w[1:]).max(1)


def _paired(worse: np.ndarray, better: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of the paired squared-error differences ``worse^2 - better^2``."""
    d = worse**2 - better**2
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.inf


@dataclass(frozen=True)
class BucketFit:
    rows: float  # rows a GPU's tokens route per forward in the bucket
    accepted: bool
    why: str
    terms: dict  # TimeModel fields at scale 1 (absolute us), when accepted
    holdout_rmse_us: float
    fallback_rmse_us: float
    remote: dict  # the remote-row term's test
    route: dict  # route_us's test
    designs: int


def fit_bucket(
    g: Geometry,
    designs: Sequence[Design],
    y: np.ndarray,
    fallback: TimeModel,
    cfg: CalibrationConfig,
    route_y: tuple[np.ndarray, np.ndarray] | None = None,
    route_f: tuple[np.ndarray, np.ndarray] | None = None,
) -> BucketFit:
    """The fit of one bucket from the designs' times ``y`` (us), with the acceptance tests.

    ``route_y`` are the route designs' times on the home table and on the table with replicas,
    ``route_f`` their features (from the slots the route op wrote)."""
    f = np.stack([features(d.rows, g) for d in designs])
    held = np.arange(len(y)) % cfg.holdout_every == cfg.holdout_every - 1
    fit = ~held
    own = np.array([getattr(fallback, k) for k in FEATURES]) * fallback.scale
    base = fit_max(f[..., :4], y, own, cfg.max_iterations)
    both = fit_max(f, y, np.r_[base[1:], 0.0], cfg.max_iterations)
    w4 = fit_max(f[fit][..., :4], y[fit], own, cfg.max_iterations)
    w5 = fit_max(f[fit], y[fit], np.r_[w4[1:], 0.0], cfg.max_iterations)
    e4 = y[held] - predict(w4, f[held][..., :4])
    e5 = y[held] - predict(w5, f[held])
    gain, se = _paired(e4, e5)
    remote = dict(per_remote_row_us=float(both[5]), mse_gain=gain, se=se, resolved=gain > cfg.z * se)
    # the built-in model refitted to a scale and an intercept on the same training designs
    p = (f[..., :4] @ own).max(1)
    k, a = np.polyfit(p[fit], y[fit], 1)
    e_own = y[held] - (a + k * p[held])
    w, e = (both, e5) if remote["resolved"] else (base, e4)
    worse, worse_se = _paired(e, e_own)
    rows = float(g.tokens * g.top_k)
    rmse, own_rmse = float(np.sqrt(np.mean(e**2))), float(np.sqrt(np.mean(e_own**2)))
    route = dict(route_us=0.0, mean=None, se=None, resolved=False)
    if route_y is not None and route_f is not None:
        r = [route_y[i] - predict(base, route_f[i][..., :4]) for i in range(2)]
        diff = r[1] - r[0]
        mean, rse = float(diff.mean()), float(diff.std(ddof=1) / math.sqrt(len(diff)))
        route.update(mean=mean, se=rse, resolved=mean > cfg.z * rse)
        route["route_us"] = mean if route["resolved"] else 0.0
    why = ""
    if not np.isfinite(w).all() or not np.isfinite(y).all():
        why = "a time or a cost is not finite"
    elif worse > cfg.z * worse_se:
        why = f"held-out error {rmse:.2f} us is worse than the built-in model's {own_rmse:.2f} us"
    terms = {}
    if not why:
        c = dict(zip(FEATURES, w[1:5].tolist()))
        cov = covariance(f if remote["resolved"] else f[..., :4], y, w)
        if remote["resolved"]:  # the route op spreads every source's rows alike: (ep - 1) / ep remote
            c["per_row_us"] += w[5] * (g.ep - 1) / g.ep
            fold = np.eye(4, 5)
            fold[2, 4] = (g.ep - 1) / g.ep
            cov = fold @ cov @ fold.T
        full = np.zeros((5, 5))  # the four kernel costs and route_us, in TimeModel.COVARIED order
        full[:4, :4] = cov
        full[4, 4] = route["se"] ** 2 if route["resolved"] else 0.0
        terms = dict(c, scale=1.0, route_us=route["route_us"], time_shares=True)
        terms.update(covariance=tuple(full.ravel().tolist()), gain_z=cfg.gain_z)
    return BucketFit(rows, not why, why, terms, rmse, own_rmse, remote, route, len(y))


# --- table ---


def canonical(value):
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.generic):
        return canonical(value.item())
    return value


def seal(table: dict) -> dict:
    body = {k: v for k, v in table.items() if k != "sha256"}
    text = json.dumps(canonical(body), sort_keys=True, separators=(",", ":"))
    return dict(body, sha256=hashlib.sha256(text.encode()).hexdigest())


def make_table(fits: Sequence[BucketFit], identity: dict, seconds: float | None = None) -> dict:
    return seal(
        dict(
            schema=TABLE_SCHEMA,
            identity=identity,
            buckets=[asdict(fit) for fit in sorted(fits, key=lambda fit: fit.rows)],
            seconds=seconds,
        )
    )


def time_model_of(table: dict | None, fallback: TimeModel) -> TimeModel:
    """The fallback model with every accepted bucket's terms (:attr:`TimeModel.buckets`); a bucket
    not accepted keeps the fallback's fields, as does every size without a table."""
    if not table or not any(b["accepted"] for b in table["buckets"]):
        return fallback
    keep = {k: getattr(fallback, k) for k in (*FEATURES, "scale", "route_us", "time_shares", "covariance", "gain_z")}
    rows = [
        (float(b["rows"]), tuple(sorted((b["terms"] if b["accepted"] else keep).items())))
        for b in table["buckets"]
    ]
    return replace(fallback, buckets=tuple(rows))


# --- GPU ---


class Launcher(NamedTuple):
    """What :func:`calibrate` launches through, so the tests can stand in for the GPU.

    ``kernel(T, slots)`` times the MoE launch of ``slots [tokens, top_k]`` physical slot ids in the
    bucket of ``T`` tokens (min over ranks of the median us per launch); ``route(T, experts, table)``
    times the route op on the packed ``table`` followed by that MoE launch and returns ``(us, slot ids
    the route op wrote)``; ``gather(obj)`` all-gathers over the EP ranks."""

    kernel: Callable[[int, np.ndarray], float]
    route: Callable[[int, np.ndarray, np.ndarray], tuple[float, np.ndarray]]
    gather: Callable[[object], list]


def calibrate(
    geometry_of: Callable[[int], Geometry],
    tokens: Sequence[int],
    rank: int,
    launcher: Launcher,
    fallback: TimeModel,
    identity: dict,
    cfg: CalibrationConfig = CalibrationConfig(),
) -> tuple[TimeModel, dict | None, str]:
    """``(model, sealed table or None, why the fallback)``: collective over the EP ranks.

    Every rank builds every bucket's designs and realises its own tokens, then the ranks vote on
    the designs' digest before the first launch; a failure anywhere keeps ``fallback`` on every
    rank. The launches run in the same order on every rank."""
    from .policy import ExpertGeometry, active_slots, build_route_tables, plan_layout

    start = time.monotonic()
    plans, why = [], None
    try:
        for t in sorted(set(int(t) for t in tokens)):
            g = geometry_of(t)
            designs = make_designs(g, cfg, (t,))
            rng = np.random.default_rng([cfg.seed, 2, rank, t])
            slots = [realise(d.rows[rank], g.tokens, g.top_k, rng) for d in designs]
            geo = ExpertGeometry(g.ep * g.homes, g.ep * g.local_slots, g.ep)
            home = geo.home_layout()
            tables, experts = [], []
            for x in route_loads(g, cfg, (t,)):
                load = x.reshape(-1).astype(np.float64)
                plan = plan_layout(load, geo, home, home, active_slots(geo))
                packed = [
                    np.array(build_route_tables(lay, None, geo.num_experts, geo.num_slots).packed(), np.int32)
                    for lay in (home, plan.layout)
                ]
                rows = _deal(x.reshape(-1), g.ep)[rank]
                experts.append(realise(rows, g.tokens, g.top_k, rng))
                tables.append(packed)
            plans.append((t, g, designs, slots, experts, tables))
        digest = hashlib.sha256(
            json.dumps([(t, list(g), [d.rows.tobytes().hex()[:64] for d in ds]) for t, g, ds, *_ in plans]).encode()
        ).hexdigest()
    except Exception as error:  # noqa: BLE001 - the vote carries it to every rank
        why, digest = f"{type(error).__name__}: {error}", None
    votes = launcher.gather((why, digest))
    failed = [f"{w} (EP rank {r})" for r, (w, _) in enumerate(votes) if w]
    if failed or len({d for _, d in votes}) != 1:
        return fallback, None, failed[0] if failed else "the EP ranks built different designs"
    fits = []
    for t, g, designs, slots, experts, tables in plans:  # no rank-dependent branch from here
        y = np.full((cfg.reps, len(designs)), np.nan)
        order = np.random.default_rng([cfg.seed, 3, t])
        for rep in range(cfg.reps):
            for d in order.permutation(len(designs)):
                y[rep, d] = launcher.kernel(t, slots[d])
        route_y = np.full((2, len(experts)), np.nan)
        route_slots: list[list[np.ndarray]] = [[], []]
        for i in range(len(experts)):
            for which in range(2):
                us, out = launcher.route(t, experts[i], tables[i][which])
                route_y[which, i] = us
                route_slots[which].append(out)
        route_f = []
        for which in range(2):  # the slots every rank's route op wrote, as rows per source and slot
            per_rank = launcher.gather([np.asarray(s).reshape(-1) for s in route_slots[which]])
            route_f.append(
                np.stack(
                    [
                        features(
                            np.stack(
                                [np.bincount(per_rank[r][i], minlength=g.ep * g.local_slots) for r in range(g.ep)]
                            ),
                            g,
                        )
                        for i in range(len(experts))
                    ]
                )
            )
        fit = fit_bucket(g, designs, np.median(y, 0), fallback.at(g.tokens * g.top_k), cfg, route_y, route_f)
        fits.append(fit)
    seconds = float(max(launcher.gather(time.monotonic() - start)))
    table = make_table(fits, identity, seconds)
    shas = launcher.gather(table["sha256"])
    if len(set(shas)) != 1:
        return fallback, None, "the EP ranks computed different tables"
    return time_model_of(table, fallback), table, ""


def torch_launcher(backend: object, layer: object, top_k: int, hidden: int, group_gather, cfg: CalibrationConfig):
    """A :class:`Launcher` on ``backend`` (the MoE backend of ``layer``, an adaptive EPLB layer).

    Every launch is the backend's decode sequence: the bucket hint, ``quantize_input`` and
    ``run_moe``; the route launches go through ``layer.route`` with the given table in place of the
    layer's (restored after). Times are CUDA events over ``cfg.timed`` launches after ``cfg.warm``,
    the min over ranks (the kernel waits for the slowest rank, so the min drops a rank's own launch
    skew) of the median over ``cfg.reps`` is taken by the caller."""
    import torch

    from ..impl_contract import MoERunContext

    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    inputs: dict[int, tuple] = {}

    def inputs_of(t: int):
        if t not in inputs:
            x = torch.randn(t, hidden, dtype=torch.bfloat16, device=device, generator=gen)
            inputs[t] = (x, torch.full((t, top_k), 1.0 / top_k, dtype=torch.float32, device=device))
        return inputs[t]

    def moe(t: int, ids: torch.Tensor, scales: torch.Tensor, routed: bool) -> torch.Tensor:
        x, _ = inputs_of(t)
        backend.set_adaptive_launch_tokens(t)
        if hasattr(backend, "set_adaptive_launch_output_dtype"):
            backend.set_adaptive_launch_output_dtype(torch.bfloat16)
        slots = layer.route(ids, token_final_scales=scales) if routed else ids
        x_q, x_sf = backend.quantize_input(x)
        backend.run_moe(
            MoERunContext(token_selected_experts=slots, token_final_scales=scales, x=x_q, x_sf=x_sf,
                          output_dtype=torch.bfloat16)
        )
        return slots

    def timed(fn) -> float:
        for _ in range(cfg.warm):
            fn()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(cfg.timed):
            fn()
        end.record()
        end.synchronize()
        return 1e3 * begin.elapsed_time(end) / cfg.timed

    def kernel(t: int, slots: np.ndarray) -> float:
        ids = torch.from_numpy(slots).to(device=device, dtype=backend.slot_id_dtype)
        _, scales = inputs_of(t)
        return min(group_gather(timed(lambda: moe(t, ids, scales, False))))

    def route(t: int, experts: np.ndarray, table: np.ndarray) -> tuple[float, np.ndarray]:
        tables = layer.manager.route_tables(device)[layer.layer_idx]
        saved = tables.clone()
        tables.copy_(torch.from_numpy(table).to(device))
        try:
            ids = torch.from_numpy(experts).to(device=device, dtype=torch.int32)
            _, scales = inputs_of(t)
            us = min(group_gather(timed(lambda: moe(t, ids, scales, True))))
            out = moe(t, ids, scales, True).cpu().numpy()
        finally:
            tables.copy_(saved)
        return us, out

    return Launcher(kernel, route, group_gather)
