from __future__ import annotations

import random

import pytest

from atlas.costdistance import brute_cost_distance, cheapest_path, cost_distance, octile
from atlas.errors import Invalid

N = 15
FLAT = [[1.0] * N for _ in range(N)]
SOURCE = (7, 7)


class TestFlatCost:
    def test_cost_distance_is_octile_distance_cell_for_cell(self):
        d = cost_distance(FLAT, SOURCE)
        for r in range(N):
            for c in range(N):
                assert d[r][c] == pytest.approx(octile(SOURCE, (r, c)))

    def test_the_path_to_the_source_is_the_source(self):
        assert cheapest_path(FLAT, SOURCE, SOURCE) == [SOURCE]


class TestBarrier:
    def test_the_cheapest_route_detours_around_an_expensive_wall(self):
        wall = [[100.0 if c == 7 and 2 <= r <= 12 else 1.0 for c in range(N)] for r in range(N)]
        d = cost_distance(wall, (7, 2))
        assert d[7][12] == pytest.approx(16.142, abs=0.01)
        assert d[7][12] / octile((7, 2), (7, 12)) == pytest.approx(1.614, abs=0.01)
        path = cheapest_path(wall, (7, 2), (7, 12))
        assert all(wall[r][c] < 100 for r, c in path)
        assert path[0] == (7, 2)
        assert path[-1] == (7, 12)


class TestAgainstBrute:
    def test_dijkstra_matches_exhaustive_relaxation(self):
        rng = random.Random(155)
        for _ in range(60):
            rows, cols = rng.randint(3, 8), rng.randint(3, 8)
            prices = [1.0, 1.0, 1.0, 5.0, 20.0]
            cost = [[rng.choice(prices) for _ in range(cols)] for _ in range(rows)]
            s = (rng.randrange(rows), rng.randrange(cols))
            a = cost_distance(cost, s)
            b = brute_cost_distance(cost, s)
            for r in range(rows):
                for c in range(cols):
                    assert a[r][c] == pytest.approx(b[r][c])


class TestRefusals:
    def test_an_empty_grid_is_refused(self):
        with pytest.raises(Invalid):
            cost_distance([], (0, 0))

    def test_a_non_positive_cost_is_refused(self):
        with pytest.raises(Invalid):
            cost_distance([[1.0, 0.0]], (0, 0))

    def test_an_off_grid_source_or_target_is_refused(self):
        with pytest.raises(Invalid):
            cost_distance(FLAT, (50, 50))
        with pytest.raises(Invalid):
            cheapest_path(FLAT, SOURCE, (50, 50))
