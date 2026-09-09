from __future__ import annotations

import math
import random

import pytest

from atlas.detourindex import grid_network
from atlas.errors import Invalid
from atlas.streetorientation import (
    bearing,
    edge_bearings,
    entropy,
    histogram,
    network_order,
    orientation_order,
    radial_network,
    rotated,
    two_grids,
    uniform_web,
    with_diagonals,
)


@pytest.fixture(scope="module")
def grid():
    return grid_network(20)


class TestTheGrid:
    def test_ln_four_and_order_one(self, grid):
        points, graph = grid
        h, order = network_order(points, graph)
        assert h == pytest.approx(math.log(4), abs=1e-9)
        assert order == 1.0

    @pytest.mark.parametrize("degrees", [3, 4.9, 5.1, 17])
    def test_a_turn_inside_a_slice_is_harmless(self, grid, degrees):
        points, graph = grid
        assert network_order(rotated(points, degrees), graph)[1] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("degrees", "h", "order"), [(5, 1.4118, 0.9999), (45, 1.9515, 0.9338)]
    )
    def test_a_turn_onto_a_slice_edge_splits(self, grid, degrees, h, order):
        read = network_order(rotated(points := grid[0], degrees), grid[1])
        assert read == pytest.approx((h, order), abs=1e-4)
        assert points is grid[0]

    @pytest.mark.parametrize(
        ("fraction", "by_count", "by_length"),
        [(0.1, 0.9948, 0.9919), (0.5, 0.9727, 0.9679), (1.0, 0.966, 0.9676)],
    )
    def test_diagonals_lower_the_order(self, grid, fraction, by_count, by_length):
        points, graph = grid
        cut = with_diagonals(points, graph, 20, fraction, random.Random(351))
        assert network_order(points, cut)[1] == pytest.approx(by_count, abs=1e-4)
        assert network_order(points, cut, True)[1] == pytest.approx(by_length, abs=1e-4)


class TestWebsAndWheels:
    @pytest.mark.parametrize(
        ("n", "h", "order", "weighted"),
        [(100, 3.5571, 0.0239, 0.0928), (300, 3.5727, 0.0098, 0.0505)],
    )
    def test_a_delaunay_web_is_near_uniform(self, n, h, order, weighted):
        points, graph = uniform_web(n, random.Random(352))
        read_h, read_order = network_order(points, graph)
        assert read_h == pytest.approx(h, abs=1e-4)
        assert read_order == pytest.approx(order, abs=1e-4)
        assert network_order(points, graph, True)[1] == pytest.approx(weighted, abs=1e-4)
        assert read_h < math.log(36)

    @pytest.mark.parametrize(
        ("rings", "spokes", "order"), [(3, 8, 0.6019), (3, 12, 0.3152), (6, 36, 0.008)]
    )
    def test_radial_networks(self, rings, spokes, order):
        points, graph = radial_network(rings, spokes)
        assert len(points) == 1 + rings * spokes
        assert network_order(points, graph)[1] == pytest.approx(order, abs=1e-4)

    @pytest.mark.parametrize(("angle", "order"), [(30, 0.9005), (45, 0.7802), (90, 1.0)])
    def test_two_grids(self, angle, order):
        points, graph = two_grids(10, angle)
        assert network_order(points, graph)[1] == pytest.approx(order, abs=1e-4)


class TestPieces:
    def test_bearings_histogram_and_entropy(self):
        assert bearing((0, 0), (0, 1)) == 0.0
        assert bearing((0, 0), (1, 0)) == 90.0
        assert bearing((0, 0), (0, -1)) == 180.0
        counts = histogram([(0.0, 1.0), (4.9, 1.0), (355.1, 1.0), (90.0, 2.0)], 36)
        assert counts[0] == 3 and counts[9] == 1
        weighted = histogram([(90.0, 2.0)], 36, True)
        assert weighted[9] == 2.0
        assert entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))
        assert orientation_order([1, 1, 1, 1] + [0] * 32) == pytest.approx(1.0)
        assert orientation_order([1] * 36) == pytest.approx(0.0)

    def test_refusals(self):
        with pytest.raises(Invalid):
            bearing((1, 1), (1, 1))
        with pytest.raises(Invalid):
            edge_bearings([(0, 0), (1, 1)], {0: {}, 1: {}})
        with pytest.raises(Invalid):
            histogram([(0.0, 1.0)], 0)
        with pytest.raises(Invalid):
            entropy([0, 0])
        with pytest.raises(Invalid):
            orientation_order([1, 1, 1])
        with pytest.raises(Invalid):
            radial_network(0, 8)
        with pytest.raises(Invalid):
            with_diagonals([(0, 0)], {0: {}}, 1, 2.0, random.Random(1))
