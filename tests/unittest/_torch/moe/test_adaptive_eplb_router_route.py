# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract of the adaptive route the MegaMoE dispatch router applies (adaptive_eplb/router_route.py).

The module is loaded by path: it is plain Python, and these checks need neither torch nor a GPU.
"""

import importlib.util
import pathlib
import random
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "router_route", _ROOT / "tensorrt_llm/_torch/moe/fused_moe/adaptive_eplb/router_route.py"
)
rr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rr)

Q32 = 1 << 32
MAX = Q32 - 1


def build_tables(expert_slots, shares, num_experts, num_slots):
    """policy.build_route_tables without numpy: copies, first entries, slots and Q32 bounds."""
    counts = [len(c) for c in expert_slots]
    starts, acc = [], 0
    for c in counts:
        starts.append(acc)
        acc += c
    slots = [s for copies in expert_slots for s in copies] + [-1] * (num_slots - acc)
    cumulative = [MAX] * num_slots
    for copies, start in zip(expert_slots, starts):
        weights = [max(0.0, shares[s]) for s in copies]
        total = sum(weights) or float(len(copies))
        weights = weights if sum(weights) else [1.0] * len(copies)
        running, bound = 0.0, 0
        for k in range(len(copies) - 1):
            running += weights[k]
            bound = min(max(int(running / total * 4294967296.0 + 0.5), bound), MAX)
            cumulative[start + k] = bound
    packed = counts + starts + slots + [v - Q32 if v >> 31 else v for v in cumulative]
    return packed, (counts, starts, slots, cumulative)


def cxx_copy_of(p, copies, cumulative):
    """adaptiveEplbKernels.cu weightedSlot's copy for point p (the staged route_into path)."""
    k = 0
    while k < copies - 1 and p >= cumulative[k]:
        k += 1
    return k


def random_layout(rng, num_experts=24, ep=4, local=8, zero_one=0.3):
    """A layout with every expert at home and replicas on free slots; shares with 0/1 copies."""
    per = num_experts // ep
    num_slots = ep * local
    expert_slots = [[(e // per) * local + e % per] for e in range(num_experts)]
    free = [s for s in range(num_slots) if s % local >= per]
    rng.shuffle(free)
    for s in free[: len(free) // 2]:
        expert_slots[rng.randrange(num_experts)].append(s)
    shares = [0.0] * num_slots
    for copies in expert_slots:
        if len(copies) > 1 and rng.random() < zero_one:
            hot = rng.randrange(len(copies))
            for k, s in enumerate(copies):
                shares[s] = 1.0 if k == hot else 0.0
        else:
            for s in copies:
                shares[s] = rng.random()
    return expert_slots, shares, num_experts, num_slots


class RouterRouteTest(unittest.TestCase):
    def test_copy_shares_are_the_q32_shares_and_honor_zero_and_one(self):
        rng = random.Random(7)
        for _ in range(200):
            expert_slots, shares, e, s = random_layout(rng)
            _, (counts, starts, slots, cumulative) = build_tables(expert_slots, shares, e, s)
            for x in range(e):
                c, st = counts[x], starts[x]
                cum = cumulative[st : st + c]
                widths = rr.copy_shares(c, cum)
                self.assertEqual(sum(widths), Q32)
                weights = [shares[slots[st + k]] for k in range(c)]
                for k in range(c):
                    # Every point of a copy's range goes to it, the neighbours' to them.
                    lo = sum(widths[:k])
                    if widths[k]:
                        self.assertEqual(rr.copy_of(lo, c, cum), k)
                        self.assertEqual(rr.copy_of(lo + widths[k] - 1, c, cum), k)
                    if weights[k] == 0.0 and sum(weights) > 0:
                        self.assertEqual(widths[k], 0, "a share of 0 takes no point")
                    if weights[k] == 1.0 and sum(weights) == 1.0:
                        self.assertEqual(widths[k], Q32, "a share of 1 takes every point")
                        for p in (0, MAX, rng.randrange(Q32)):
                            self.assertEqual(rr.copy_of(p, c, cum), k)

    def test_staged_route_shares_zero_exactly_and_one_but_its_last_point(self):
        """The staged route (route_into) keeps its semantics: a share of 0 takes no point and a share
        of 1 every point but 2^32 - 1, when that copy is not its expert's last (bound 2^32 - 1)."""
        cum = [0, MAX, MAX]  # copies 0, 1, 2 with shares 0, 1, 0
        self.assertEqual(cxx_copy_of(0, 3, cum), 1)
        self.assertEqual(cxx_copy_of(MAX - 1, 3, cum), 1)
        self.assertEqual(cxx_copy_of(MAX, 3, cum), 2)  # the one leaked point of 2^32
        self.assertEqual(rr.copy_of(MAX, 3, cum), 1)  # the router keeps it on the share-1 copy
        for p in (0, 1, MAX - 1):
            self.assertNotEqual(cxx_copy_of(p, 3, cum), 0)

    def test_slots_are_deterministic_and_route_unreplicated_experts_home(self):
        rng = random.Random(11)
        expert_slots, shares, e, s = random_layout(rng)
        packed, _ = build_tables(expert_slots, shares, e, s)
        ids = [[rng.randrange(-1, e + 2) for _ in range(6)] for _ in range(64)]
        first = rr.slots_of(ids, 3, packed, e, s)
        self.assertEqual(first, rr.slots_of(ids, 3, packed, e, s))
        for row, out in zip(ids, first):
            for x, slot in zip(row, out):
                if not 0 <= x < e:
                    self.assertEqual(slot, -1)
                elif len(expert_slots[x]) == 1:
                    self.assertEqual(slot, expert_slots[x][0])
                else:
                    self.assertIn(slot, expert_slots[x])
                    self.assertGreater(
                        shares[slot] if sum(shares[q] for q in expert_slots[x]) else 1, 0
                    )

    def test_record_counts_live_rows_only_while_enabled(self):
        ids = [[0, 1], [1, 2], [2, 3], [3, 4]]
        word = (4 << 16) | 2  # 2 real of 4 requests: the first half of the rows are live
        rows, forwards = [0.0] * 5, [0]
        rr.record(rows, forwards, ids, 4, word, enabled=True, first=True)
        self.assertEqual(rows, [1.0, 2.0, 1.0, 0.0, 0.0])
        self.assertEqual(forwards, [1])
        rr.record(rows, forwards, ids, 4, word, enabled=True, first=False)
        self.assertEqual(forwards, [1])
        always_on = list(rows)
        rr.record(rows, forwards, ids, 4, word, enabled=False, first=True)
        self.assertEqual(rows, always_on, "nothing is counted outside a window")
        self.assertEqual(rr.live_rows(None, 7), 7)
        self.assertEqual(rr.live_rows((3 << 16) | 3, 6), 6)
        self.assertEqual(rr.live_rows((4 << 16) | 1, 6), 6)  # rows not a multiple of requests: all

    def test_realized_split_error_is_bounded(self):
        """(c): rows of each copy against its share over random tokens of one rank's bucket."""
        rng = random.Random(5)
        worst = []
        for tokens in (8, 32, 64, 128, 256):
            errors = []
            for _ in range(300):
                copies = rng.randrange(2, 5)
                weights = [rng.random() for _ in range(copies)]
                if rng.random() < 0.3:
                    weights = [1.0 if k == 0 else 0.0 for k in range(copies)]
                expert_slots = [list(range(copies))]
                shares = weights + [0.0] * 4
                _, (counts, starts, slots, cumulative) = build_tables(
                    expert_slots, shares, 1, copies + 4
                )
                cum = cumulative[:copies]
                picks = sorted(rng.sample(range(tokens), rng.randrange(1, tokens + 1)))
                ep_rank = rng.randrange(32)
                got = [0] * copies
                for t in picks:
                    got[rr.copy_of(rr.point(0, t, ep_rank), copies, cum)] += 1
                total = sum(weights)
                for k in range(copies):
                    errors.append(abs(got[k] - len(picks) * weights[k] / total))
                    if weights[k] == 0.0:
                        self.assertEqual(got[k], 0)
            errors.sort()
            worst.append((tokens, errors[-1], errors[int(0.99 * (len(errors) - 1))]))
        for tokens, max_rows, p99_rows in worst:
            # A copy is off its share by at most a few rows at every bucket size.
            self.assertLessEqual(max_rows, max(4.0, 0.2 * tokens), (tokens, max_rows, p99_rows))
        print(
            "router split error rows (tokens, max, p99):",
            [(t, round(m, 2), round(p, 2)) for t, m, p in worst],
        )


if __name__ == "__main__":
    unittest.main()
