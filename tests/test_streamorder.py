from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.flowdirection import NOWHERE, flow_directions
from atlas.streamorder import order_bound, stream_order


def _binary_tree(k):
    width, rows = 2**k, k + 1
    d = [[NOWHERE] * width for _ in range(rows)]
    for level in range(k):
        step = 2**level
        for c in range(0, width, step * 2):
            d[level][c] = (level + 1, c)
            d[level][c + step] = (level + 1, c)
    return d


class TestTheRule:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
    def test_a_symmetric_binary_tree_climbs_one_order_per_level(self, depth):
        order = stream_order(_binary_tree(depth))
        assert order[depth][0] == depth + 1

    def test_a_stem_fed_by_first_order_tributaries_stays_at_two(self):
        n = 12
        chain = [[NOWHERE] * 2 for _ in range(n)]
        for r in range(n - 1):
            chain[r][0] = (r + 1, 0)
        for r in range(n):
            chain[r][1] = (r, 0)
        order = stream_order(chain)
        assert [order[r][0] for r in range(n)] == [1] + [2] * (n - 1)
        assert all(order[r][1] == 1 for r in range(n))


class TestTheBound:
    def test_order_never_exceeds_one_plus_log2_of_the_cell_count(self):
        rng = random.Random(159)
        for _ in range(40):
            rows, cols = rng.randint(10, 30), rng.randint(10, 30)
            grid = [[rng.uniform(0, 100) for _ in range(cols)] for _ in range(rows)]
            order = stream_order(flow_directions(grid))
            assert max(max(row) for row in order) <= order_bound(rows * cols)

    def test_the_bound_itself(self):
        assert order_bound(1) == 1
        assert order_bound(2) == 2
        assert order_bound(900) == 10


class TestRefusals:
    def test_an_empty_grid_is_refused(self):
        with pytest.raises(Invalid):
            stream_order([])

    def test_a_cycle_is_refused(self):
        with pytest.raises(Invalid):
            stream_order([[(0, 1), (0, 0)]])

    def test_a_non_positive_cell_count_is_refused(self):
        with pytest.raises(Invalid):
            order_bound(0)
