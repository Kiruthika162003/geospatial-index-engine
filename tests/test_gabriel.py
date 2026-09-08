from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.gabriel import (
    delaunay_edges,
    edge_length,
    gabriel_edges,
    is_connected,
    relative_neighborhood_edges,
)


def _random_sets(seed, count=150):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(5, 60)
        yield list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})


class TestNesting:
    def test_relative_neighborhood_within_gabriel_within_delaunay(self):
        for pts in _random_sets(125):
            d, g, r = delaunay_edges(pts), gabriel_edges(pts), relative_neighborhood_edges(pts)
            assert r <= g <= d

    def test_all_three_graphs_stay_connected(self):
        for pts in _random_sets(125):
            assert is_connected(pts, delaunay_edges(pts))
            assert is_connected(pts, gabriel_edges(pts))
            assert is_connected(pts, relative_neighborhood_edges(pts))


class TestThinning:
    def test_edges_per_point_fall_along_the_chain(self):
        totals = [0.0, 0.0, 0.0]
        count = 0
        for pts in _random_sets(125):
            n = len(pts)
            totals[0] += len(delaunay_edges(pts)) / n
            totals[1] += len(gabriel_edges(pts)) / n
            totals[2] += len(relative_neighborhood_edges(pts)) / n
            count += 1
        means = [t / count for t in totals]
        # measured 2.55, 1.58, 1.09 edges per point
        assert means[0] == pytest.approx(2.55, abs=0.1)
        assert means[1] == pytest.approx(1.58, abs=0.1)
        assert means[2] == pytest.approx(1.09, abs=0.1)
        assert means[0] > means[1] > means[2]

    def test_a_square_with_its_center(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
        assert len(delaunay_edges(pts)) == 8
        assert len(gabriel_edges(pts)) == 8  # the center sits exactly on each side's circle
        assert len(relative_neighborhood_edges(pts)) == 4
        assert ((0, 0), (10, 0)) in gabriel_edges(pts)

    def test_edge_length(self):
        assert edge_length(((0, 0), (3, 4))) == pytest.approx(5.0)


class TestRefusals:
    def test_fewer_than_three_distinct_points_is_refused(self):
        with pytest.raises(Invalid):
            delaunay_edges([(0, 0), (1, 1)])
