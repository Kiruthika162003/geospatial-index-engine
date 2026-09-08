from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.segmentintersect import intersection_point, segments_intersect


def _brute(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def onseg(p, q, r):
        return (
            min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
            and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
        )

    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    if (d1 > 0) != (d2 > 0) and d1 and d2 and (d3 > 0) != (d4 > 0) and d3 and d4:
        return True
    if d1 == 0 and onseg(c, d, a):
        return True
    if d2 == 0 and onseg(c, d, b):
        return True
    if d3 == 0 and onseg(a, b, c):
        return True
    return bool(d4 == 0 and onseg(a, b, d))


class TestCases:
    def test_a_clean_crossing(self):
        assert segments_intersect((0, 0), (4, 4), (0, 4), (4, 0))
        assert intersection_point((0, 0), (4, 4), (0, 4), (4, 0)) == (2.0, 2.0)

    def test_parallel_segments_do_not_cross(self):
        assert not segments_intersect((0, 0), (4, 0), (0, 1), (4, 1))

    def test_collinear_overlap_crosses(self):
        assert segments_intersect((0, 0), (4, 0), (2, 0), (6, 0))

    def test_collinear_disjoint_does_not(self):
        assert not segments_intersect((0, 0), (2, 0), (4, 0), (6, 0))

    def test_a_touching_endpoint_counts(self):
        assert segments_intersect((0, 0), (4, 0), (2, 0), (2, 4))

    def test_a_non_crossing_pair_has_no_point(self):
        assert intersection_point((0, 0), (1, 0), (0, 1), (1, 1)) is None


class TestAgainstBrute:
    def test_boolean_matches_brute_on_a_dense_grid(self):
        rng = random.Random(33)
        for _ in range(200000):
            pts = [(rng.randint(0, 5), rng.randint(0, 5)) for _ in range(4)]
            a, b, c, d = pts
            if a == b or c == d:
                continue
            assert segments_intersect(a, b, c, d) == _brute(a, b, c, d)


class TestRefusals:
    def test_a_collinear_point_request_is_refused(self):
        with pytest.raises(Invalid):
            intersection_point((0, 0), (4, 0), (2, 0), (6, 0))
