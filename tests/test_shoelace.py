from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.shoelace import (
    area,
    centroid,
    is_counterclockwise,
    signed_area,
)


def _fan_area(poly):
    a = 0.0
    for i in range(1, len(poly) - 1):
        x0, y0 = poly[0]
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        a += abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2
    return a


class TestKnownShapes:
    def test_unit_square(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert area(sq) == 1.0
        assert signed_area(sq) == 1.0
        assert is_counterclockwise(sq)
        assert centroid(sq) == (0.5, 0.5)

    def test_triangle_area_and_centroid(self):
        tri = [(0, 0), (4, 0), (0, 3)]
        assert area(tri) == 6.0
        cx, cy = centroid(tri)
        assert cx == pytest.approx(4 / 3)
        assert cy == pytest.approx(1.0)


class TestOrientation:
    def test_reversing_flips_the_sign_not_the_magnitude(self):
        sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert signed_area(sq) == -signed_area(sq[::-1])
        assert not is_counterclockwise(sq[::-1])


class TestAgainstFanDecomposition:
    def test_area_matches_a_triangle_fan(self):
        rng = random.Random(23)
        for _ in range(5000):
            k = rng.randint(3, 10)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            cx, cy = rng.uniform(-5, 5), rng.uniform(-5, 5)
            r = rng.uniform(1, 5)
            poly = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            assert area(poly) == pytest.approx(_fan_area(poly))
            assert signed_area(poly) == pytest.approx(-signed_area(poly[::-1]))


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            signed_area([(0, 0), (1, 1)])

    def test_a_degenerate_polygon_has_no_centroid(self):
        with pytest.raises(Degenerate):
            centroid([(0, 0), (1, 0), (2, 0)])  # collinear, zero area
