from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.quadtreeoccupancy import (
    balanced_leaves,
    build,
    bytes_estimate,
    clustered,
    histogram,
    near_duplicates,
    occupancy,
    uniform,
)


class TestUniform:
    @pytest.mark.parametrize(
        ("capacity", "leaves", "empty", "fill", "depth", "nodes"),
        [
            (4, 2122, 0.138, 0.471, 7, 2829),
            (8, 1054, 0.026, 0.474, 6, 1405),
            (16, 568, 0.007, 0.44, 5, 757),
        ],
    )
    def test_leaves_sit_under_half_full(self, capacity, leaves, empty, fill, depth, nodes):
        tree = build(uniform(4000, random.Random(660)), capacity)
        read = occupancy(tree, capacity)
        assert read["leaves"] == leaves
        assert read["empty_fraction"] == pytest.approx(empty, abs=1e-3)
        assert read["mean_fill"] == pytest.approx(fill, abs=1e-3)
        assert read["max_depth"] == depth
        assert read["nodes"] == nodes
        assert read["mean_fill"] < 0.5

    def test_the_histogram_and_the_scaling(self):
        tree = build(uniform(4000, random.Random(660)), 8)
        assert histogram(tree) == {
            0: 27,
            1: 76,
            2: 169,
            3: 223,
            4: 207,
            5: 151,
            6: 103,
            7: 68,
            8: 30,
        }
        for n, leaves, depth, per_point in (
            (200, 61, 3, 41.9),
            (4000, 1060, 6, 38.6),
            (16000, 4210, 7, 38.5),
        ):
            tree = build(uniform(n, random.Random(663)), 8)
            read = occupancy(tree, 8)
            assert read["leaves"] == leaves
            assert read["max_depth"] == depth
            assert bytes_estimate(tree) / n == pytest.approx(per_point, abs=0.1)
        assert balanced_leaves(4000, 8) == 1024
        assert balanced_leaves(3, 8) == 1


class TestSkew:
    @pytest.mark.parametrize(
        ("capacity", "empty", "fill", "depth"),
        [(4, 0.187, 0.427, 10), (8, 0.092, 0.424, 9), (16, 0.085, 0.382, 8)],
    )
    def test_clusters_leave_siblings_empty(self, capacity, empty, fill, depth):
        tree = build(clustered(4000, 8, 15.0, random.Random(661)), capacity)
        read = occupancy(tree, capacity)
        assert read["empty_fraction"] == pytest.approx(empty, abs=1e-3)
        assert read["mean_fill"] == pytest.approx(fill, abs=1e-3)
        assert read["max_depth"] == depth

    @pytest.mark.parametrize(
        ("capacity", "empty", "fill", "depth"),
        [(4, 0.477, 0.291, 24), (8, 0.542, 0.212, 23), (16, 0.727, 0.142, 23)],
    )
    def test_near_duplicates_dig_deep(self, capacity, empty, fill, depth):
        tree = build(near_duplicates(200, random.Random(662)), capacity)
        read = occupancy(tree, capacity)
        assert read["empty_fraction"] == pytest.approx(empty, abs=1e-3)
        assert read["mean_fill"] == pytest.approx(fill, abs=1e-3)
        assert read["max_depth"] == depth


class TestRefusals:
    def test_bad_trees(self):
        with pytest.raises(Invalid):
            build([], 0)
