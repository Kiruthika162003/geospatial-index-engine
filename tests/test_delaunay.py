from __future__ import annotations

import random

import pytest

from atlas.delaunay import in_circumcircle, is_delaunay, triangulate
from atlas.errors import Invalid
from atlas.jarvismarch import convex_hull


def _on_segment(p, a, b):
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return (
        cross == 0
        and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _boundary_count(pts):
    # h for Euler's formula counts every point on the hull boundary, collinear ones too
    hull = convex_hull(pts)
    m = len(hull)
    edges = [(hull[i], hull[(i + 1) % m]) for i in range(m)]
    return sum(1 for p in pts if any(_on_segment(p, a, b) for a, b in edges))


def _random_sets(seed, count=300):
    rng = random.Random(seed)
    sets = []
    for _ in range(count):
        n = rng.randint(4, 40)
        pts = list({(rng.randint(0, 1000), rng.randint(0, 1000)) for _ in range(n)})
        if len(pts) >= 4:
            sets.append(pts)
    return sets


class TestCircumcircle:
    def test_the_circumcenter_is_inside_and_a_far_point_is_not(self):
        tri = ((0, 0), (4, 0), (0, 4))
        assert in_circumcircle(tri, (2, 2))
        assert not in_circumcircle(tri, (10, 10))

    def test_every_triangle_has_an_empty_circumcircle(self):
        for pts in _random_sets(119):
            assert is_delaunay(triangulate(pts), pts)


class TestEulerCount:
    def test_a_square_gives_two_triangles(self):
        assert len(triangulate([(0, 0), (1, 0), (1, 1), (0, 1)])) == 2

    def test_a_perturbed_grid_matches_the_formula(self):
        rng = random.Random(119)
        grid = [
            (x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1))
            for x in range(6)
            for y in range(6)
        ]
        tris = triangulate(grid)
        assert len(tris) == 2 * 36 - _boundary_count(grid) - 2
        assert is_delaunay(tris, grid)

    def test_the_count_is_two_n_minus_h_minus_two_on_every_random_set(self):
        for pts in _random_sets(119):
            assert len(triangulate(pts)) == 2 * len(pts) - _boundary_count(pts) - 2

    def test_a_merely_large_super_triangle_silently_drops_hull_triangles(self):
        # the refuted guess: 50x the spread looked generous; it lost triangles on 11 of 300
        lost = 0
        for pts in _random_sets(119):
            target = 2 * len(pts) - _boundary_count(pts) - 2
            got = len(triangulate(pts, super_scale=50))
            assert got <= target  # always below, never above
            lost += got < target
        assert lost == 11


class TestRefusals:
    def test_fewer_than_three_distinct_points_is_refused(self):
        with pytest.raises(Invalid):
            triangulate([(0, 0), (1, 1), (1, 1)])

    def test_a_super_triangle_no_bigger_than_the_spread_is_refused(self):
        with pytest.raises(Invalid):
            triangulate([(0, 0), (1, 0), (0, 1)], super_scale=1)
