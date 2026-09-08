from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from atlas.alphashape import alpha_area, alpha_triangles, boundary_edges, circumradius
from atlas.errors import Invalid
from atlas.jarvismarch import convex_hull
from atlas.shoelace import area as poly_area


def _crescent(seed=131, count=400):
    rng = random.Random(seed)
    pts = []
    for _ in range(count):
        a = math.radians(rng.uniform(30, 330))
        r = rng.uniform(8, 10)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


class TestCircumradius:
    def test_a_three_four_five_triangle(self):
        assert circumradius(((0, 0), (3, 0), (0, 4))) == pytest.approx(2.5)

    def test_collinear_is_infinite(self):
        assert circumradius(((0, 0), (1, 1), (2, 2))) == math.inf


class TestTheAlphaKnob:
    def test_alpha_zero_is_the_convex_hull(self):
        pts = _crescent()
        assert alpha_area(pts, 0) == pytest.approx(poly_area(convex_hull(pts)))

    def test_area_falls_monotonically_with_alpha(self):
        pts = _crescent()
        areas = [alpha_area(pts, a) for a in (0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0)]
        for earlier, later in pairwise(areas):
            assert later <= earlier + 1e-9
        assert areas[-1] < areas[0] * 0.1  # measured 12.5 against 294.9

    def test_a_moderate_alpha_carves_the_bay_the_hull_fills(self):
        pts = _crescent()
        true_area = (300 / 360) * math.pi * (100 - 64)
        hull_area = poly_area(convex_hull(pts))
        carved = alpha_area(pts, 0.2)
        assert hull_area / true_area > 3.0  # measured 3.13x overshoot
        assert carved == pytest.approx(true_area, rel=0.1)  # measured 89.2 vs 94.2

    def test_the_outline_is_a_closed_set_of_single_owner_edges(self):
        pts = _crescent()
        edges = boundary_edges(alpha_triangles(pts, 0.2))
        assert len(edges) == 70
        degree: dict = {}
        for a, b in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        assert all(d % 2 == 0 for d in degree.values())  # every outline vertex has even degree


class TestRefusals:
    def test_fewer_than_three_points_is_refused(self):
        with pytest.raises(Invalid):
            alpha_triangles([(0, 0), (1, 1)], 0.1)

    def test_a_negative_alpha_is_refused(self):
        with pytest.raises(Invalid):
            alpha_triangles([(0, 0), (1, 0), (0, 1)], -1)
