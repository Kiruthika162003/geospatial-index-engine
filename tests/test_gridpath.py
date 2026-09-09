from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Missing
from atlas.gridpath import maze_grid, octile, open_grid, random_cost_grid, search, walled_grid


class TestOpenGround:
    @pytest.mark.parametrize("size", [20, 40, 80])
    def test_a_star_expands_the_diagonal_and_dijkstra_the_square(self, size):
        grid = open_grid(size)
        start, goal = (0, 0), (size - 1, size - 1)
        d_cost, _, d_expanded = search(grid, start, goal, 0.0)
        a_cost, path, a_expanded = search(grid, start, goal, 1.0)
        assert d_expanded == size * size
        assert a_expanded == size
        assert a_cost == pytest.approx(d_cost) == pytest.approx(octile(start, goal))
        assert path[0] == start and path[-1] == goal
        assert search(grid, start, goal, 1.5)[0] == pytest.approx(d_cost)


class TestWallsAndMazes:
    def test_the_wall_halves_the_saving_and_the_maze_removes_it(self):
        start, goal = (0, 0), (39, 39)
        d = search(walled_grid(40), start, goal, 0.0)
        a = search(walled_grid(40), start, goal, 1.0)
        w = search(walled_grid(40), start, goal, 1.5)
        assert (d[2], a[2], w[2]) == (1120, 566, 147)
        assert a[0] == pytest.approx(d[0]) == pytest.approx(65.113, abs=1e-3)
        assert w[0] == pytest.approx(d[0])
        d = search(maze_grid(40), start, goal, 0.0)
        a = search(maze_grid(40), start, goal, 1.0)
        assert (d[2], a[2]) == (862, 849)
        assert a[0] == pytest.approx(d[0]) == pytest.approx(685.154, abs=1e-3)
        assert search(maze_grid(40), start, goal, 1.5)[0] == pytest.approx(d[0])


class TestRandomCosts:
    def test_the_unit_cost_heuristic_barely_helps_and_weighting_costs_little(self):
        rng = random.Random(254)
        ratios, longer = [], []
        for _ in range(20):
            grid = random_cost_grid(30, rng)
            d = search(grid, (0, 0), (29, 29), 0.0)
            a = search(grid, (0, 0), (29, 29), 1.0)
            w = search(grid, (0, 0), (29, 29), 1.5)
            assert a[0] == pytest.approx(d[0], abs=1e-9)
            ratios.append(d[2] / a[2])
            longer.append(w[0] / d[0] - 1)
        assert sum(ratios) / 20 == pytest.approx(1.05, abs=0.01)
        assert max(ratios) == pytest.approx(1.29, abs=0.01)
        assert 100 * sum(longer) / 20 == pytest.approx(0.03, abs=0.01)
        assert 100 * max(longer) == pytest.approx(0.23, abs=0.01)


class TestRefusals:
    def test_no_path_and_bad_inputs(self):
        blocked = open_grid(6)
        for c in range(6):
            blocked[3][c] = math.inf
        with pytest.raises(Missing):
            search(blocked, (0, 0), (5, 5))
        with pytest.raises(Invalid):
            search([], (0, 0), (0, 0))
        with pytest.raises(Invalid):
            search(open_grid(5), (0, 0), (9, 9))
        with pytest.raises(Invalid):
            search(walled_grid(5), (0, 0), (2, 0))
        with pytest.raises(Invalid):
            search(open_grid(5), (0, 0), (4, 4), -1)
