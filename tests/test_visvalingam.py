from __future__ import annotations

import random

import pytest

from atlas.douglaspeucker import simplify as dp_simplify
from atlas.errors import Invalid
from atlas.visvalingam import simplify_by_area, simplify_to


class TestSimplify:
    def test_a_straight_line_collapses_to_endpoints(self):
        line = [(i, 0) for i in range(50)]
        assert simplify_to(line, 2) == [(0, 0), (49, 0)]

    def test_endpoints_are_always_kept(self):
        pts = [(i, (i * 7) % 5) for i in range(30)]
        out = simplify_to(pts, 6)
        assert out[0] == pts[0]
        assert out[-1] == pts[-1]
        assert len(out) == 6

    def test_area_threshold_removes_small_triangles(self):
        pts = [(0, 0), (1, 0.001), (2, 0), (3, 5), (4, 0)]
        # the (1, 0.001) point forms a near-zero triangle and goes
        out = simplify_by_area(pts, 0.1)
        assert (1, 0.001) not in out
        assert (3, 5) in out  # the tall triangle survives


class TestContrastWithDouglasPeucker:
    def test_visvalingam_drops_a_thin_spike_douglas_peucker_keeps(self):
        spike = [(0, 0), (5, 0), (5.01, 10), (5.02, 0), (10, 0)]
        vv = simplify_to(spike, 4)
        dp = dp_simplify(spike, 1.0)
        assert (5.01, 10) not in vv  # area-based: a thin sliver is dropped
        assert (5.01, 10) in dp  # distance-based: the far spike is kept

    def test_the_two_keep_different_vertices_at_the_same_budget(self):
        rng = random.Random(53)
        distances = []
        for _ in range(200):
            path = [(i, rng.uniform(0, 10)) for i in range(60)]
            vv = set(simplify_to(path, 20))
            lo, hi = 0.0, 20.0
            for _ in range(40):
                mid = (lo + hi) / 2
                if len(dp_simplify(path, mid)) > 20:
                    lo = mid
                else:
                    hi = mid
            dp = set(dp_simplify(path, hi))
            distances.append(1 - len(vv & dp) / len(vv | dp))
        # measured mean Jaccard distance ~0.42: the vertex sets differ a lot
        assert sum(distances) / len(distances) > 0.25


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            simplify_to(None, 3)

    def test_keeping_fewer_than_two_is_refused(self):
        with pytest.raises(Invalid):
            simplify_to([(0, 0), (1, 1), (2, 0)], 1)
