from __future__ import annotations

import math
from itertools import pairwise

import pytest

from atlas.errors import Invalid
from atlas.terrain import FLAT_ASPECT, aspect_degrees, cone, plane, slope_degrees


def _cone_error(cell, east, north, true=30.0):
    n = int(24 / cell) + 1
    grid = cone(n, n, cell, true)
    c0 = (n - 1) // 2
    r, c = c0 + round(north / cell), c0 + round(east / cell)
    return abs(slope_degrees(grid, cell, r, c) - true)


def _gaussian_error(cell):
    n = int(24 / cell) + 1
    c0 = (n - 1) // 2
    s2 = 50.0
    grid = [
        [
            10 * math.exp(-((r - c0) ** 2 + (c - c0) ** 2) * cell * cell / (2 * s2))
            for c in range(n)
        ]
        for r in range(n)
    ]
    r, c = c0 + round(3 / cell), c0 + round(4 / cell)
    x, y = (c - c0) * cell, (r - c0) * cell
    z = 10 * math.exp(-(x * x + y * y) / (2 * s2))
    true_slope = math.degrees(math.atan(math.hypot(-x / s2 * z, -y / s2 * z)))
    return abs(slope_degrees(grid, cell, r, c) - true_slope)


class TestPlanesAreExact:
    def test_slope_and_aspect_are_recovered_to_floating_point(self):
        for s in (5, 15, 30, 45, 60):
            for a in (0, 45, 90, 135, 180, 225, 270, 315, 33.3):
                g = plane(9, 9, 2.5, s, a)
                for r in range(1, 8):
                    for c in range(1, 8):
                        assert slope_degrees(g, 2.5, r, c) == pytest.approx(s, abs=1e-9)
                        gap = abs((aspect_degrees(g, 2.5, r, c) - a + 180) % 360 - 180)
                        assert gap < 1e-9

    def test_flat_ground_has_zero_slope_and_the_sentinel_aspect(self):
        g = plane(5, 5, 1.0, 0, 0)
        assert slope_degrees(g, 1.0, 2, 2) == 0.0
        assert aspect_degrees(g, 1.0, 2, 2) == FLAT_ASPECT


class TestCurvedSurfaces:
    def test_a_cone_is_exact_along_its_axis_where_it_is_locally_planar(self):
        # the refuted setup: no decay because the axis row is linear and symmetric
        for cell in (2, 1, 0.5, 0.25):
            assert _cone_error(cell, 5, 0) < 1e-12

    def test_off_axis_the_cone_error_quarters_per_halving(self):
        errors = [_cone_error(cell, 5, 3) for cell in (2, 1, 0.5, 0.25)]
        assert errors == pytest.approx([0.7909, 0.1426, 0.0355, 0.0089], abs=0.002)
        assert errors[1] / errors[2] == pytest.approx(4.0, abs=0.1)
        assert errors[2] / errors[3] == pytest.approx(4.0, abs=0.1)

    def test_a_gaussian_hill_shows_the_quadratic_rate_everywhere(self):
        errors = [_gaussian_error(cell) for cell in (2, 1, 0.5, 0.25)]
        for earlier, later in pairwise(errors):
            assert earlier / later == pytest.approx(4.0, abs=0.1)


class TestRefusals:
    def test_a_border_cell_is_refused(self):
        with pytest.raises(Invalid):
            slope_degrees(plane(5, 5, 1.0, 10, 90), 1.0, 0, 2)

    def test_a_non_positive_cell_is_refused(self):
        with pytest.raises(Invalid):
            slope_degrees(plane(5, 5, 1.0, 10, 90), 0, 2, 2)

    def test_a_grid_too_small_is_refused(self):
        with pytest.raises(Invalid):
            plane(2, 2, 1.0, 10, 90)
        with pytest.raises(Invalid):
            cone(2, 5, 1.0, 10)
