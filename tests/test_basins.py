from __future__ import annotations

import math
import random

import pytest

from atlas.basins import basin_sizes, is_connected, label_basins
from atlas.errors import Invalid
from atlas.flowdirection import flow_accumulation, flow_directions, sinks


class TestBowls:
    def test_a_bowl_is_one_basin_holding_every_cell(self):
        n, c0 = 21, 10
        bowl = [[math.hypot(r - c0, c - c0) for c in range(n)] for r in range(n)]
        labels = label_basins(flow_directions(bowl))
        assert basin_sizes(labels) == {0: n * n}

    def test_two_bowls_split_along_the_ridge_between_them(self):
        two = [
            [min(math.hypot(r - 10, c - 5), math.hypot(r - 10, c - 25)) for c in range(31)]
            for r in range(21)
        ]
        labels = label_basins(flow_directions(two))
        assert len(basin_sizes(labels)) == 2
        west, east = labels[10][3], labels[10][27]
        assert west != east
        assert all(labels[r][c] == west for r in range(21) for c in range(14))
        assert all(labels[r][c] == east for r in range(21) for c in range(17, 31))


class TestPartition:
    def test_labels_cover_the_grid_match_the_sinks_and_agree_with_accumulation(self):
        rng = random.Random(157)
        for _ in range(30):
            rows, cols = rng.randint(8, 25), rng.randint(8, 25)
            grid = [[rng.uniform(0, 100) for _ in range(cols)] for _ in range(rows)]
            directions = flow_directions(grid)
            labels = label_basins(directions)
            sizes = basin_sizes(labels)
            assert -1 not in sizes
            assert sum(sizes.values()) == rows * cols
            sink_cells = sinks(directions)
            assert len(sizes) == len(sink_cells)
            acc = flow_accumulation(directions)
            for r, c in sink_cells:
                assert sizes[labels[r][c]] == acc[r][c]
            for label in sizes:
                assert is_connected(labels, label)


class TestRefusals:
    def test_an_empty_direction_grid_is_refused(self):
        with pytest.raises(Invalid):
            label_basins([])

    def test_a_cycle_is_refused(self):
        with pytest.raises(Invalid):
            label_basins([[(0, 1), (0, 0)]])

    def test_an_absent_label_is_not_connected(self):
        assert not is_connected([[0, 0], [0, 0]], 7)
