# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU model of the adaptive mode of ``DSv4RouteNvfp4QuantKernel`` against ``adaptive_eplb_route_into``.

The fused launch routes one token row per CTA and maps its picks in the same launch:
  * a pick whose expert has one copy (every pick of a home layer) is written by its row: slot ``slots[starts[e]]``;
  * a pick whose expert has copies marks its 1024-element split block; the row that completes the block's
    arrival count maps the block's marked picks by the route op's rule (ordinal among the block's picks of that
    expert, Q32 shares), in chunks of the CTA's threads as the kernel does;
  * rows past the live-row word get slot -1 and weight 0, the bucket past the rows too (tail CTAs);
  * statistics: each live valid pick adds 1.0 to its expert's row while ``enabled``; one thread adds the
    forward when ``enabled`` and ``first``.
``_fused_model`` mirrors that structure step by step; the test compares it with the reference op over random
tables, ids, live words, flags and arrival orders. Each staged element must be written exactly once.
"""

import random

import numpy as np
import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference

from tensorrt_llm._torch.moe.live_rows import live_row_count

SPLIT_BLOCK = reference.SPLIT_BLOCK
THREADS = 128  # the kernel's CTA, which maps a block in chunks of one pick per thread
TOP_K, EXPERTS, SLOTS = 6, 384, 512
U32 = (1 << 32) - 1


def _point_step(n):
    """``pointStep``: floor(2^32 / n) for n > 1."""
    return U32 if n <= 1 else (1 << 32) // n


def _slot(expert, ordinal, n, block, ep_rank, counts, starts, slots, shares):
    """The route op's ``weightedSlot`` (reference.route for one element)."""
    copies, start = counts[expert], starts[expert]
    point = (expert * reference.EXPERT_MULTIPLIER + block * reference.BLOCK_MULTIPLIER) & U32
    point = (point + ep_rank * reference.RANK_MULTIPLIER + ordinal * _point_step(n)) & U32
    k = 0
    while k < copies - 1 and point >= shares[start + k]:
        k += 1
    return slots[start + k]


def _first_copy(expert, counts, starts, slots):
    """``firstCopySlot`` / weightedSlot with one copy; num_slots for an unroutable expert."""
    if not 0 <= expert < len(counts):
        return len(slots)
    copies, start = counts[expert], starts[expert]
    if copies <= 0 or start < 0 or start + copies > len(slots):
        return len(slots)
    return slots[start]


def _replicated(expert, counts, starts, slots):
    """Whether the pick needs the block's ordinal: a routable expert with several copies."""
    if not 0 <= expert < len(counts):
        return False
    copies, start = counts[expert], starts[expert]
    return copies > 1 and start >= 0 and start + copies <= len(slots)


def _map_block(block, flat_ids, count, live, tables, ep_rank, staged, writes):
    """The completing row's CTA: map the block's replicated picks in chunks of THREADS, as the kernel."""
    counts, starts, slots, shares = tables
    lo, hi = block * SPLIT_BLOCK, min(count, (block + 1) * SPLIT_BLOCK)
    spread = [e if _replicated(e, counts, starts, slots) else EXPERTS for e in flat_ids[lo:hi]]
    totals = {}
    for s in spread:  # pass 1: shared histogram of the block
        if s < EXPERTS:
            totals[s] = totals.get(s, 0) + 1
    seen = {}
    for chunk in range(lo, hi, THREADS):  # pass 2: one pick per thread
        width = min(THREADS, hi - chunk)
        part = spread[chunk - lo : chunk - lo + width]
        for t in range(width):
            s = part[t]
            if s == EXPERTS:
                continue
            rank = sum(1 for u in range(t) if part[u] == s)  # earlier threads of the chunk
            ordinal = seen.get(s, 0) + rank
            f = chunk + t
            if f < live:
                staged[f] = _slot(s, ordinal, totals[s], block, ep_rank, *tables)
                writes[f] += 1
        for t in range(width):  # the chunk's last pick of each expert advances its count
            s = part[t]
            if s < EXPERTS and all(part[u] != s for u in range(t + 1, width)):
                seen[s] = seen.get(s, 0) + sum(1 for u in range(width) if part[u] == s)


def _fused_model(ids, weights, tables, ep_rank, bucket, word, enabled, first, rows0, fwd0, rng, straddle_bug=False):
    counts, starts, slots, shares = tables
    tokens = ids.shape[0]
    count = tokens * TOP_K
    live = count if word is None else live_row_count(word, tokens) * TOP_K
    flat = ids.reshape(-1).tolist()
    scratch = [-999] * count  # the returned ids, written by each row before its arrival
    staged = np.full(bucket * TOP_K, 0x5EED, dtype=np.int64)
    staged_w = np.full(bucket * TOP_K, 1234.5, dtype=np.float32)
    writes = np.zeros(bucket * TOP_K, dtype=np.int64)
    rows = rows0.copy()
    fwd = fwd0
    blocks = (count + SPLIT_BLOCK - 1) // SPLIT_BLOCK
    expected = [
        min(tokens - 1, ((b + 1) * SPLIT_BLOCK - 1) // TOP_K) - (b * SPLIT_BLOCK) // TOP_K + 1
        for b in range(blocks)
    ]
    if straddle_bug:  # negative control: a row is counted only in the block of its first pick
        expected = [sum(1 for r in range(tokens) if (r * TOP_K) // SPLIT_BLOCK == b) for b in range(blocks)]
    arrivals, marked = [0] * blocks, [0] * blocks
    order = list(range(tokens))
    rng.shuffle(order)
    for row in order:  # routing CTAs in any order
        for k in range(TOP_K):
            f = row * TOP_K + k
            e = flat[f]
            scratch[f] = e
            kept = f < live
            staged_w[f] = weights[row, k] if kept else 0.0
            if _replicated(e, counts, starts, slots):
                marked[f // SPLIT_BLOCK] = 1  # atomicOr before the release
                if not kept:
                    staged[f] = -1
                    writes[f] += 1
            else:
                staged[f] = _first_copy(e, counts, starts, slots) if kept else -1
                writes[f] += 1
            if enabled and kept and 0 <= e < EXPERTS:
                rows[e] = np.float32(rows[e] + np.float32(1.0))  # atomicAdd, exact on integers
        if row == 0 and enabled and first:
            fwd += 1
        for b in range((row * TOP_K) // SPLIT_BLOCK, ((row + 1) * TOP_K - 1) // SPLIT_BLOCK + 1):
            arrivals[b] += 1  # atom.add.acq_rel.gpu
            if arrivals[b] == expected[b] and marked[b]:
                _map_block(b, scratch, count, live, tables, ep_rank, staged, writes)
    for f in range(count, bucket * TOP_K):  # tail CTAs
        staged[f], staged_w[f] = -1, 0.0
        writes[f] += 1
    assert (writes == 1).all(), "every staged element is written exactly once"
    return staged, staged_w.reshape(bucket, TOP_K), rows, fwd


def _tables(rng, replicas):
    counts = [1] * EXPERTS
    for _ in range(replicas):
        counts[rng.randrange(EXPERTS)] += 1
    used = sum(counts)
    slots = list(range(SLOTS))
    rng.shuffle(slots)
    starts, shares, pos = [], [], 0
    for e in range(EXPERTS):
        starts.append(pos)
        cuts = sorted(rng.randrange(1 << 32) for _ in range(counts[e] - 1)) + [U32]
        shares += cuts
        pos += counts[e]
    shares += [U32] * (SLOTS - used)
    return counts, starts, slots, shares


def _case(rng, tokens, replicas):
    tables = _tables(rng, replicas)
    ids = np.array([[rng.randrange(EXPERTS) for _ in range(TOP_K)] for _ in range(tokens)], dtype=np.int64)
    if tokens:
        hot = [rng.randrange(EXPERTS) for _ in range(3)]
        for row in range(tokens):
            for k in range(TOP_K):
                if rng.random() < 0.3:
                    ids[row, k] = rng.choice(hot)
    weights = np.array([[rng.random() for _ in range(TOP_K)] for _ in range(tokens)], dtype=np.float32)
    bucket = next(b for b in (256, 1024, 4096) if b >= tokens)
    word = None
    if rng.random() < 0.6:
        requests = rng.randrange(1, max(2, tokens // 4) + 1)
        word = (requests << 16) | rng.randrange(0, requests + 1)
    return tables, ids, weights, bucket, word


def _reference(tables, ids, weights, bucket, word, enabled, first, rows0, fwd0, ep_rank):
    counts, starts, slots, shares = tables
    out = torch.full((bucket * TOP_K,), 0x5EED, dtype=torch.int64)
    weights_out = torch.full((bucket, TOP_K), 1234.5, dtype=torch.float32)
    rows = torch.from_numpy(rows0.copy())
    forwards = torch.tensor([fwd0], dtype=torch.int64)
    reference.route_into(
        torch.from_numpy(ids),
        torch.tensor(counts, dtype=torch.int32),
        torch.tensor(starts, dtype=torch.int32),
        torch.tensor(slots, dtype=torch.int32),
        torch.tensor(shares, dtype=torch.int64),
        ep_rank,
        out,
        torch.from_numpy(weights),
        weights_out,
        rows,
        forwards,
        torch.tensor([int(enabled)], dtype=torch.int32),
        first,
        None if word is None else torch.tensor([word], dtype=torch.int32),
    )
    return out.numpy(), weights_out.numpy(), rows.numpy(), int(forwards[0])


@pytest.mark.parametrize("replicas", [0, 1, 40, 128], ids=["home", "one", "some", "many"])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64, 170, 171, 256, 512, 683, 1024])
def test_fused_route_model_matches_the_route_op(tokens, replicas):
    rng = random.Random(tokens * 131 + replicas)
    for trial in range(3):
        tables, ids, weights, bucket, word = _case(rng, tokens, replicas)
        enabled, first = trial != 1, trial != 2
        rows0 = np.array([rng.randrange(1 << 12) for _ in range(EXPERTS)], dtype=np.float32)
        ep_rank = rng.randrange(32)
        want = _reference(tables, ids, weights, bucket, word, enabled, first, rows0, 7, ep_rank)
        for order_seed in range(2):
            got = _fused_model(
                ids, weights, tables, ep_rank, bucket, word, enabled, first, rows0, 7, random.Random(order_seed)
            )
            assert np.array_equal(got[0], want[0]), "staged slots"
            assert got[1].tobytes() == want[1].tobytes(), "staged weights"
            assert got[2].tobytes() == want[2].tobytes(), "statistics rows, bit for bit"
            assert got[3] == want[3], "forwards"


def test_the_model_catches_a_block_mapped_early():
    """Negative control: counting a straddling row only in its first block lets the next block be mapped before
    that row wrote its picks there; the comparison (or the write-once check) must fail."""
    caught = 0
    for seed, tokens in enumerate((171, 342, 683, 1024)):
        rng = random.Random(seed)
        tables, ids, weights, bucket, _ = _case(rng, tokens, 128)
        rows0 = np.zeros(EXPERTS, np.float32)
        want = _reference(tables, ids, weights, bucket, None, True, True, rows0, 0, 3)
        try:
            got = _fused_model(ids, weights, tables, 3, bucket, None, True, True, rows0, 0, random.Random(seed), True)
            caught += not np.array_equal(got[0], want[0])
        except (AssertionError, KeyError, IndexError):
            caught += 1
    assert caught == 4
