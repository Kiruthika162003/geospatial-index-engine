from __future__ import annotations

import math
import random

import pytest

from atlas.emptycircle import grid_search, largest_empty_circle
from atlas.errors import Invalid


def _random_sets(seed, count=60):
    rng = random.Random(seed)
    for _ in range(count):
        n = rng.randint(5, 25)
        yield list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})


class TestExactness:
    def test_the_grid_never_beats_the_exact_answer(self):
        for sites in _random_sets(129):
            _, r = largest_empty_circle(sites)
            _, gr = grid_search(sites, 60)
            assert gr <= r + 1e-9

    def test_the_interior_only_search_is_beaten_a_quarter_of_the_time(self):
        # the refuted guess: scoring only interior Voronoi vertices loses to a grid
        # whenever the optimum sits on the hull boundary; measured 16 of 60
        beaten = 0
        for sites in _random_sets(129):
            _, ri = largest_empty_circle(sites, interior_only=True)
            _, gr = grid_search(sites, 60)
            beaten += gr > ri + 1e-9
        assert beaten == 16

    def test_the_circle_is_empty_and_touches_its_pinning_sites(self):
        for sites in _random_sets(129):
            (cx, cy), r = largest_empty_circle(sites)
            distances = [math.hypot(cx - sx, cy - sy) for sx, sy in sites]
            assert min(distances) >= r - 1e-9
            assert sum(1 for d in distances if abs(d - r) < 1e-6) >= 2


class TestTheGridClosesFromBelow:
    def test_finer_grids_approach_the_exact_radius(self):
        rng = random.Random(130)
        sites = list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(20)})
        _, r = largest_empty_circle(sites)
        shortfalls = [(r - grid_search(sites, cells)[1]) / r for cells in (10, 30, 100, 300)]
        assert shortfalls == sorted(shortfalls, reverse=True)
        assert shortfalls[-1] < 0.01
        assert all(s >= -1e-9 for s in shortfalls)


class TestKnownCase:
    def test_a_square_of_sites_is_centered_at_half_the_diagonal(self):
        (cx, cy), r = largest_empty_circle([(0, 0), (10, 0), (10, 10), (0, 10)])
        assert (cx, cy) == pytest.approx((5.0, 5.0))
        assert r == pytest.approx(math.sqrt(50))


class TestRefusals:
    def test_fewer_than_three_sites_is_refused(self):
        with pytest.raises(Invalid):
            largest_empty_circle([(0, 0), (1, 1)])

    def test_collinear_sites_are_refused(self):
        with pytest.raises(Invalid):
            largest_empty_circle([(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_a_non_positive_grid_is_refused(self):
        with pytest.raises(Invalid):
            grid_search([(0, 0), (1, 0), (0, 1)], 0)
