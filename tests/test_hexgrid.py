from __future__ import annotations

import random
from collections import deque

import pytest

from atlas.errors import Invalid
from atlas.hexgrid import disk, distance, from_point, make, neighbors, ring, to_point

ORIGIN = (0, 0, 0)


def _bfs(a, b):
    seen = {a: 0}
    queue = deque([a])
    while queue:
        h = queue.popleft()
        if h == b:
            return seen[h]
        for n in neighbors(h):
            if n not in seen:
                seen[n] = seen[h] + 1
                queue.append(n)
    return None


class TestRingsAndDisks:
    def test_a_ring_holds_six_k_cells(self):
        for k in range(1, 8):
            assert len(ring(ORIGIN, k)) == 6 * k

    def test_a_disk_holds_one_plus_three_k_k_plus_one(self):
        for k in range(6):
            assert len(disk(ORIGIN, k)) == 1 + 3 * k * (k + 1)

    def test_ring_cells_are_distinct_and_at_exactly_distance_k(self):
        for k in range(1, 8):
            cells = ring(ORIGIN, k)
            assert len(set(cells)) == len(cells)
            assert all(distance(ORIGIN, c) == k for c in cells)

    def test_the_zero_ring_is_the_center(self):
        assert ring(ORIGIN, 0) == [ORIGIN]


class TestDistance:
    def test_six_neighbors_at_distance_one(self):
        ns = neighbors(ORIGIN)
        assert len(ns) == 6
        assert all(distance(ORIGIN, n) == 1 for n in ns)

    def test_the_three_way_max_matches_breadth_first_search(self):
        rng = random.Random(81)
        for _ in range(2000):
            q1, r1 = rng.randint(-8, 8), rng.randint(-8, 8)
            q2, r2 = rng.randint(-8, 8), rng.randint(-8, 8)
            a, b = make(q1, r1, -q1 - r1), make(q2, r2, -q2 - r2)
            assert distance(a, b) == _bfs(a, b)


class TestPointConversion:
    def test_roundtrip_and_jitter_stay_in_the_hex(self):
        rng = random.Random(82)
        for _ in range(5000):
            q, r = rng.randint(-20, 20), rng.randint(-20, 20)
            h = make(q, r, -q - r)
            x, y = to_point(h, 2.0)
            assert from_point(x, y, 2.0) == h
            jx, jy = x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)
            assert from_point(jx, jy, 2.0) == h


class TestRefusals:
    def test_coordinates_must_sum_to_zero(self):
        with pytest.raises(Invalid):
            make(1, 1, 1)

    def test_a_negative_ring_is_refused(self):
        with pytest.raises(Invalid):
            ring(ORIGIN, -1)

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(Invalid):
            from_point(0, 0, 0)
