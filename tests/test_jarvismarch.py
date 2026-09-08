from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.jarvismarch import convex_hull


def _andrew(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def half(ps):
        h = []
        for p in ps:
            while len(h) >= 2 and (
                (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])
            ) <= 0:
                h.pop()
            h.append(p)
        return h

    return half(pts)[:-1] + half(pts[::-1])[:-1]


class TestHullCorrectness:
    def test_a_square_drops_its_interior(self):
        assert set(convex_hull([(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)])) == {
            (0, 0), (4, 0), (4, 4), (0, 4),
        }

    def test_collinear_collapses(self):
        assert convex_hull([(0, 0), (1, 1), (2, 2), (3, 3)]) == [(0, 0), (3, 3)]

    def test_it_matches_andrews_monotone_chain(self):
        rng = random.Random(55)
        for _ in range(3000):
            pts = [
                (rng.randint(-30, 30), rng.randint(-30, 30))
                for _ in range(rng.randint(1, 40))
            ]
            assert set(convex_hull(pts)) == set(_andrew(pts))


class TestOutputSensitivity:
    def test_a_disk_has_a_small_hull(self):
        rng = random.Random(1)
        disk = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(2000)]
        # only a handful of the 2000 points reach the boundary
        assert len(convex_hull(disk)) < 40

    def test_points_on_a_circle_are_all_hull_vertices(self):
        circle = [
            (math.cos(2 * math.pi * i / 500), math.sin(2 * math.pi * i / 500))
            for i in range(500)
        ]
        assert len(convex_hull(circle)) == 500  # worst case h == n


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            convex_hull(None)
