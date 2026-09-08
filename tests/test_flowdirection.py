from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.flowdirection import (
    NOWHERE,
    flow_accumulation,
    flow_directions,
    path_to_sink,
    sinks,
)


class TestBowl:
    def test_every_cell_drains_to_the_center_which_collects_them_all(self):
        n, c0 = 21, 10
        bowl = [[math.hypot(r - c0, c - c0) for c in range(n)] for r in range(n)]
        d = flow_directions(bowl)
        assert sinks(d) == [(c0, c0)]
        assert flow_accumulation(d)[c0][c0] == n * n
        assert max(path_to_sink(d, (r, c)) for r in range(n) for c in range(n)) == 10


class TestConservation:
    def test_a_tilted_plane_drains_to_one_low_corner_sink(self):
        rows, cols = 12, 15
        plane = [[r * 1.0 + c * 0.3 for c in range(cols)] for r in range(rows)]
        d = flow_directions(plane)
        s = sinks(d)
        assert len(s) == 1
        assert s[0][0] == 0
        acc = flow_accumulation(d)
        assert sum(acc[r][c] for r, c in s) == rows * cols

    def test_sink_accumulations_sum_to_the_cell_count_on_rough_ground(self):
        rng = random.Random(151)
        rough = [[rng.uniform(0, 100) for _ in range(30)] for _ in range(30)]
        d = flow_directions(rough)
        acc = flow_accumulation(d)
        s = sinks(d)
        assert len(s) == 105
        assert sum(acc[r][c] for r, c in s) == 900

    def test_the_drainage_graph_has_no_cycles(self):
        rng = random.Random(151)
        rough = [[rng.uniform(0, 100) for _ in range(30)] for _ in range(30)]
        d = flow_directions(rough)
        for r in range(30):
            for c in range(30):
                assert path_to_sink(d, (r, c)) >= 0


class TestTheDiagonalRule:
    def test_an_equal_drop_goes_straight_not_diagonal(self):
        g = [[2, 1, 2], [1, 3, 1], [2, 1, 2]]
        target = flow_directions(g)[1][1]
        assert target in ((0, 1), (1, 0), (1, 2), (2, 1))
        assert target != NOWHERE


class TestRefusals:
    def test_an_empty_grid_is_refused(self):
        with pytest.raises(Invalid):
            flow_directions([])

    def test_a_cycle_is_reported_rather_than_looped(self):
        cyclic = [[(0, 1), (0, 0)]]
        with pytest.raises(Invalid):
            path_to_sink(cyclic, (0, 0))
