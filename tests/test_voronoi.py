from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.voronoi import Voronoi, circumcenter


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TestCircumcenter:
    def test_a_right_triangle_centers_on_its_hypotenuse_midpoint(self):
        assert circumcenter(((0, 0), (4, 0), (0, 4))) == pytest.approx((2.0, 2.0))

    def test_collinear_points_have_no_circumcenter(self):
        with pytest.raises(Degenerate):
            circumcenter(((0, 0), (1, 1), (2, 2)))


class TestVertices:
    def test_every_vertex_is_equidistant_from_its_three_sites_and_nothing_is_nearer(self):
        rng = random.Random(123)
        for _ in range(100):
            n = rng.randint(5, 30)
            sites = list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})
            v = Voronoi(sites)
            for tri, c in zip(v.triangles, v.vertices, strict=True):
                d = [_dist(c, p) for p in tri]
                assert max(d) - min(d) < 1e-6
                r = min(d)
                assert all(_dist(c, s) >= r - 1e-6 for s in v.sites if s not in tri)

    def test_a_vertex_has_exactly_three_nearest_sites_in_general_position(self):
        v = Voronoi([(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)])
        assert len(v.vertices) == 4
        assert len(v.edges) == 4
        assert len(v.nearest_sites(v.vertices[0])) == 3


class TestEdges:
    def test_every_finite_edge_lies_on_the_bisector_of_its_shared_delaunay_edge(self):
        rng = random.Random(123)
        for _ in range(100):
            n = rng.randint(5, 30)
            sites = list({(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)})
            v = Voronoi(sites)
            for i, j, (a, b) in v.edges:
                p, q = v.vertices[i], v.vertices[j]
                m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                assert _dist(m, a) == pytest.approx(_dist(m, b), abs=1e-6)


class TestRefusals:
    def test_fewer_than_three_distinct_sites_is_refused(self):
        with pytest.raises(Invalid):
            Voronoi([(0, 0), (1, 1), (1, 1)])
