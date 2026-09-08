from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.snaptopath import longest_segment, snap, snap_to_vertex

PATH = [(0, 0), (10, 0), (10, 10)]


class TestSnap:
    def test_beside_a_segment_snaps_to_its_interior(self):
        foot, dist, along = snap((5, 3), PATH)
        assert foot == (5.0, 0.0)
        assert dist == pytest.approx(3.0)
        assert along == pytest.approx(5.0)

    def test_beyond_the_end_clamps_to_the_last_vertex(self):
        foot, dist, along = snap((15, 12), PATH)
        assert foot == (10.0, 10.0)
        assert dist == pytest.approx((25 + 4) ** 0.5)
        assert along == pytest.approx(20.0)

    def test_along_path_position_accumulates_segment_lengths(self):
        assert snap((10, 5), PATH)[2] == pytest.approx(15.0)


class TestAgainstVertexSnap:
    def test_segment_snap_is_never_farther_and_the_gap_is_bounded(self):
        rng = random.Random(95)
        gaps = []
        for _ in range(5000):
            n = rng.randint(2, 8)
            path = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            q = (rng.uniform(-10, 110), rng.uniform(-10, 110))
            _, ds, _ = snap(q, path)
            _, dv = snap_to_vertex(q, path)
            assert ds <= dv + 1e-9
            assert dv - ds <= longest_segment(path) / 2 + 1e-9
            gaps.append(dv - ds)
        assert sum(gaps) / len(gaps) > 2  # measured mean 4.0, max 42.5

    def test_the_bound_is_reached_beside_a_long_straight_run(self):
        long_run = [(0, 0), (100, 0)]
        _, ds, along = snap((50, 0.001), long_run)
        _, dv = snap_to_vertex((50, 0.001), long_run)
        assert ds == pytest.approx(0.001)
        assert dv == pytest.approx(50.0)
        assert dv - ds == pytest.approx(50.0, abs=0.01)  # half the longest segment
        assert along == pytest.approx(50.0)


class TestRefusals:
    def test_a_one_vertex_path_is_refused(self):
        with pytest.raises(Invalid):
            snap((0, 0), [(1, 1)])

    def test_an_empty_path_is_refused_for_vertex_snap(self):
        with pytest.raises(Invalid):
            snap_to_vertex((0, 0), [])
