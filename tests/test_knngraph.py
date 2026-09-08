from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.knngraph import KnnGraph


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TestNeighbors:
    def test_neighbor_distances_match_brute_force(self):
        rng = random.Random(121)
        for _ in range(300):
            n = rng.randint(6, 40)
            pts = list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})
            k = rng.randint(1, 5)
            g = KnnGraph(pts, k)
            for i, p in enumerate(g.points):
                got = sorted(_dist(g.points[j], p) for j in g.out[i])
                expected = sorted(_dist(q, p) for q in g.points if q != p)[:k]
                assert got == pytest.approx(expected)

    def test_out_degree_is_exactly_k_and_in_degree_is_not(self):
        rng = random.Random(121)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(500)]
        g = KnnGraph(pts, 5)
        assert all(g.out_degree(i) == 5 for i in range(500))
        ins = g.in_degrees()
        assert sum(ins) / len(ins) == pytest.approx(5.0)
        assert min(ins) < 5 < max(ins)  # measured 0 to 10: hubs and orphans exist


class TestDirectedness:
    def test_uniform_points_have_a_high_mutual_fraction_within_bounds(self):
        rng = random.Random(121)
        pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(500)]
        g = KnnGraph(pts, 5)
        assert g.mutual_fraction() == pytest.approx(0.804, abs=0.02)
        assert 500 * 5 // 2 <= g.symmetrized_edges() <= 500 * 5

    def test_clustering_with_stragglers_lowers_the_mutual_fraction(self):
        rng = random.Random(122)
        uniform = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(500)]
        clustered = [(rng.gauss(50, 2), rng.gauss(50, 2)) for _ in range(450)]
        clustered += [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(50)]
        assert KnnGraph(clustered, 5).mutual_fraction() < KnnGraph(uniform, 5).mutual_fraction()


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            KnnGraph(None, 3)

    def test_a_non_positive_k_is_refused(self):
        with pytest.raises(Invalid):
            KnnGraph([(0, 0), (1, 1)], 0)

    def test_k_must_be_below_the_distinct_point_count(self):
        with pytest.raises(Invalid):
            KnnGraph([(0, 0), (1, 1), (2, 2)], 3)
