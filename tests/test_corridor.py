from __future__ import annotations

import random

import pytest

from atlas.corridor import (
    band_width_law,
    barrier_cost,
    best_cost,
    cell_count,
    components,
    corridor,
    corridor_field,
    ellipse_area,
    octile_corridor_law,
    rough_cost,
    uniform_cost,
    width_profile,
)
from atlas.errors import Invalid

A, B = (50, 20), (50, 80)


@pytest.fixture(scope="module")
def uniform():
    return corridor_field(uniform_cost(101), A, B)


class TestUniform:
    @pytest.mark.parametrize(
        ("tolerance", "cells", "width", "ellipse", "octile"),
        [
            (0.0, 61, 1, 0.0, 0.0),
            (1.0, 179, 3, 527.0, 150.4),
            (5.0, 761, 13, 1276.3, 715.3),
            (10.0, 1451, 25, 1982.3, 1411.2),
            (20.0, 2777, 49, 3324.7, 2757.8),
        ],
    )
    def test_a_band_not_an_ellipse(self, uniform, tolerance, cells, width, ellipse, octile):
        assert best_cost(uniform) == pytest.approx(60.0)
        mask = corridor(uniform, tolerance)
        assert cell_count(mask) == cells
        assert components(mask) == 1
        profile = width_profile(mask, A, B, 5)
        assert profile[2] == width
        assert ellipse_area(60.0, tolerance) == pytest.approx(ellipse, abs=0.1)
        assert octile_corridor_law(60.0, tolerance) == pytest.approx(octile, abs=0.1)
        if tolerance >= 5:
            assert abs(octile / cells - 1) < 0.07
            assert band_width_law(tolerance) == pytest.approx(width, abs=0.4)


class TestRoughAndBarrier:
    def test_the_rough_field(self):
        field = corridor_field(rough_cost(101, random.Random(650)), A, B)
        assert best_cost(field) == pytest.approx(136.57, abs=0.01)
        assert components(corridor(field, 0.0)) == 3
        for tolerance, cells in ((5.0, 519), (10.0, 1064), (40.0, 3164)):
            mask = corridor(field, tolerance)
            assert cell_count(mask) == cells
            assert components(mask) == 1

    def test_the_wall_pinches_the_corridor(self):
        field = corridor_field(barrier_cost(101, 50, 30), A, B)
        assert best_cost(field) == pytest.approx(74.912, abs=1e-3)
        assert width_profile(corridor(field, 2.0), A, B, 5) == [16, 16, 3, 16, 16]
        assert width_profile(corridor(field, 10.0), A, B, 5) == [29, 29, 5, 29, 29]


class TestRefusals:
    def test_bad_tolerances_samples_and_ends(self, uniform):
        with pytest.raises(Invalid):
            corridor(uniform, -1.0)
        with pytest.raises(Invalid):
            width_profile(corridor(uniform, 1.0), A, B, 0)
        with pytest.raises(Invalid):
            width_profile(corridor(uniform, 1.0), A, A)
