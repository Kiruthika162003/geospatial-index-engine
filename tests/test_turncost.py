from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.turncost import length, noisy_cost, route, straightness, travel_cost, turns, wall

SIZE = 61
A, B = (30, 5), (30, 55)


class TestNoisySurfaces:
    @pytest.mark.parametrize(
        ("sigma", "readings"),
        [
            (0.1, ((0.0, 7, 52.49, 49.798), (0.1, 3, 50.83, 50.0), (1.0, 0, 50.0, 51.342))),
            (
                0.3,
                (
                    (0.0, 22, 59.7, 42.01),
                    (0.1, 13, 54.14, 42.878),
                    (1.0, 5, 52.49, 44.667),
                    (3.0, 0, 50.0, 54.025),
                ),
            ),
            (0.6, ((0.0, 33, 64.53, 30.366), (1.0, 5, 52.49, 39.554), (10.0, 0, 50.0, 58.941))),
        ],
    )
    def test_turns_fall_and_travel_cost_rises_with_the_penalty(self, sigma, readings):
        cost = noisy_cost(SIZE, sigma, random.Random(760))
        previous = math.inf
        for penalty, bends, span, travel in readings:
            path, _ = route(SIZE, A, B, penalty, cost=cost)
            assert turns(path) == bends
            assert length(path) == pytest.approx(span, abs=0.01)
            assert travel_cost(path, cost) == pytest.approx(travel, abs=1e-3)
            assert bends <= previous
            previous = bends
        assert straightness(route(SIZE, A, B, readings[-1][0], cost=cost)[0]) == 1.0


class TestOpenGrid:
    def test_the_penalty_changes_the_shape_but_never_the_length(self):
        free, _ = route(SIZE, (5, 10), (55, 40), 0.0)
        bent, _ = route(SIZE, (5, 10), (55, 40), 0.1)
        assert (turns(free), turns(bent)) == (3, 1)
        assert length(free) == pytest.approx(62.426, abs=1e-3)
        assert length(bent) == pytest.approx(length(free), abs=1e-9)
        assert math.dist((5, 10), (55, 40)) == pytest.approx(58.31, abs=1e-2)
        for penalty in (0.0, 3.0):
            path, _ = route(SIZE, A, B, penalty)
            assert turns(path) == 0 and length(path) == 50.0

    def test_a_wall_forces_two_turns_at_any_penalty(self):
        blocked = wall(SIZE, 30, 45)
        for penalty in (0.0, 3.0):
            path, _ = route(SIZE, A, B, penalty, blocked)
            assert turns(path) == 2
            assert length(path) == pytest.approx(61.598, abs=1e-3)


class TestRefusals:
    def test_bad_grids_penalties_cells_and_walls(self):
        with pytest.raises(Invalid):
            route(1, (0, 0), (0, 0), 0.0)
        with pytest.raises(Invalid):
            route(5, (0, 0), (4, 4), -1.0)
        with pytest.raises(Invalid):
            route(5, (0, 0), (9, 9), 0.0)
        with pytest.raises(Invalid):
            route(5, (0, 0), (4, 4), 0.0, {(r, 2) for r in range(5)})
        with pytest.raises(Invalid):
            noisy_cost(5, -1.0, random.Random(1))
        with pytest.raises(Invalid):
            straightness([(0, 0)])
