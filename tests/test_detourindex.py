from __future__ import annotations

import math
import random

import pytest

from atlas.detourindex import (
    bridged,
    delaunay_network,
    detour_ratios,
    dijkstra,
    edge_count,
    grid_network,
    manhattan_law,
    random_pairs,
    sparsified,
    summary,
    uniform_points,
    without_long_edges,
)
from atlas.errors import Invalid


@pytest.fixture(scope="module")
def town():
    points = uniform_points(300, random.Random(320))
    return points, delaunay_network(points), random_pairs(300, 500, random.Random(321))


class TestDelaunay:
    def test_six_percent_on_average_and_a_quarter_at_worst(self, town):
        points, graph, pairs = town
        assert edge_count(graph) == 881
        read = summary(detour_ratios(points, graph, pairs))
        assert read["mean"] == pytest.approx(1.0609, abs=1e-4)
        assert read["median"] == pytest.approx(1.057, abs=1e-3)
        assert read["max"] == pytest.approx(1.2595, abs=1e-4)
        assert read["disconnected"] == 0.0

    @pytest.mark.parametrize(("n", "mean"), [(50, 1.0698), (200, 1.0555)])
    def test_the_mean_barely_moves_with_size(self, n, mean):
        points = uniform_points(n, random.Random(324))
        graph = delaunay_network(points)
        read = summary(detour_ratios(points, graph, random_pairs(n, 300, random.Random(325))))
        assert read["mean"] == pytest.approx(mean, abs=1e-4)
        assert read["max"] < 1.4


class TestGrid:
    def test_the_manhattan_law(self):
        points, graph = grid_network(20)
        read = summary(detour_ratios(points, graph, random_pairs(400, 500, random.Random(323))))
        assert read["mean"] == pytest.approx(1.2708, abs=1e-4)
        assert read["mean"] == pytest.approx(manhattan_law(), abs=3e-3)
        assert read["max"] == pytest.approx(math.sqrt(2), abs=1e-9)
        assert read["median"] == pytest.approx(1.3077, abs=1e-4)


class TestThinning:
    @pytest.mark.parametrize(
        ("keep", "edges", "mean", "worst", "lost"),
        [
            (0.9, 797, 1.1008, 2.2547, 0.0),
            (0.7, 608, 1.2149, 3.6725, 0.018),
            (0.5, 446, 1.5355, 4.5125, 0.086),
        ],
    )
    def test_random_thinning(self, town, keep, edges, mean, worst, lost):
        points, graph, pairs = town
        thin = sparsified(graph, keep, random.Random(322))
        assert edge_count(thin) == edges
        read = summary(detour_ratios(points, thin, pairs))
        assert (read["mean"], read["max"], read["disconnected"]) == pytest.approx(
            (mean, worst, lost), abs=1e-4
        )

    @pytest.mark.parametrize(
        ("factor", "edges", "lost", "mean"),
        [
            (1.0, 441, 0.93, 1.3903),
            (1.5, 696, 0.032, 1.3261),
            (2.0, 815, 0.0, 1.0758),
            (3.0, 866, 0.0, 1.061),
        ],
    )
    def test_cutting_long_edges(self, town, factor, edges, lost, mean):
        points, graph, pairs = town
        cut = without_long_edges(graph, factor)
        assert edge_count(cut) == edges
        read = summary(detour_ratios(points, cut, pairs))
        assert read["disconnected"] == pytest.approx(lost, abs=1e-4)
        assert read["mean"] == pytest.approx(mean, abs=1e-4)


class TestTheBridge:
    @pytest.mark.parametrize(
        ("bridge_y", "mean", "median", "worst"),
        [(50.0, 1.4099, 1.2437, 5.0706), (10.0, 1.8454, 1.5335, 10.5838)],
    )
    def test_cross_river_pairs_pay_for_the_bridge(self, town, bridge_y, mean, median, worst):
        points, graph, pairs = town
        graph = bridged(points, graph, 50.0, bridge_y)
        assert edge_count(graph) == 840
        ratios = detour_ratios(points, graph, pairs)
        west = [points[i][0] < 50 for i in range(len(points))]
        cross = [x for (i, j), x in zip(pairs, ratios, strict=True) if west[i] != west[j]]
        same = [x for (i, j), x in zip(pairs, ratios, strict=True) if west[i] == west[j]]
        read = summary(cross)
        got = (read["mean"], read["median"], read["max"])
        assert got == pytest.approx((mean, median, worst), abs=1e-4)
        assert summary(same)["mean"] == pytest.approx(1.0642, abs=1e-4)


class TestRefusals:
    def test_bad_networks_pairs_and_cuts(self):
        with pytest.raises(Invalid):
            delaunay_network([(0, 0), (1, 1)])
        with pytest.raises(Invalid):
            grid_network(1)
        _, graph = grid_network(3)
        with pytest.raises(Invalid):
            dijkstra(graph, 99)
        with pytest.raises(Invalid):
            detour_ratios([(0, 0)] * 9, graph, [])
        with pytest.raises(Invalid):
            detour_ratios([(0, 0)] * 9, graph, [(1, 1)])
        with pytest.raises(Invalid):
            sparsified(graph, 0, random.Random(1))
        with pytest.raises(Invalid):
            summary([math.inf])
        with pytest.raises(Invalid):
            bridged([(0, 0), (1, 0), (2, 0)], {0: {}, 1: {}, 2: {}}, 5.0, 0.0)
