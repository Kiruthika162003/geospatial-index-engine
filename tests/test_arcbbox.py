from __future__ import annotations

import random

import pytest

from atlas.arcbbox import (
    arc_box,
    bulge_km,
    endpoint_box,
    passes_vertex,
    sampled_box,
    vertex_latitude,
)
from atlas.bearing import initial_bearing
from atlas.errors import Invalid

PARIS = (48.86, 2.35)
VANCOUVER = (49.28, -123.12)


class TestParisToVancouver:
    def test_the_route_peaks_at_68_over_greenland(self):
        box = arc_box(PARIS, VANCOUVER)
        assert box == pytest.approx((48.86, -123.12, 68.336, 2.35), abs=1e-3)
        assert sampled_box(PARIS, VANCOUVER, 20000) == pytest.approx(box, abs=1e-3)
        assert endpoint_box(PARIS, VANCOUVER) == (48.86, -123.12, 49.28, 2.35)
        assert bulge_km(PARIS, VANCOUVER) == pytest.approx(2118.9, abs=0.5)
        bearing = initial_bearing(*PARIS, *VANCOUVER)
        assert vertex_latitude(48.86, bearing) == pytest.approx(68.336, abs=1e-3)


class TestAgainstSampledArcs:
    def test_random_arcs_match_the_sampled_box_and_the_endpoint_box_often_misses(self):
        rng = random.Random(225)
        agree = missed = 0
        worst_miss = 0.0
        for _ in range(200):
            a = (rng.uniform(-70, 70), rng.uniform(-180, 180))
            b = (rng.uniform(-70, 70), rng.uniform(-180, 180))
            box, sampled, plain = arc_box(a, b), sampled_box(a, b, 4000), endpoint_box(a, b)
            assert box[0] == pytest.approx(sampled[0], abs=1e-4)
            assert box[2] == pytest.approx(sampled[2], abs=1e-4)
            assert box[3] - box[1] == pytest.approx(sampled[3] - sampled[1], abs=1e-9)
            agree += box == plain
            miss = max(sampled[2] - plain[2], plain[0] - sampled[0])
            missed += miss > 1
            worst_miss = max(worst_miss, miss)
        assert 90 < agree < 120
        assert 60 < missed < 90
        assert worst_miss > 40


class TestTheBulgeAlongAParallel:
    @pytest.mark.parametrize(
        ("span", "north"),
        [(30, 45.993), (60, 49.107), (90, 54.736), (150, 75.489), (179, 89.5)],
    )
    def test_the_bulge_grows_with_the_span(self, span, north):
        a, b = (45.0, -span / 2), (45.0, span / 2)
        assert arc_box(a, b)[2] == pytest.approx(north, abs=1e-3)
        assert sampled_box(a, b, 20000)[2] == pytest.approx(north, abs=1e-3)


class TestSpecialArcs:
    def test_meridional_and_short_arcs_give_the_endpoint_box(self):
        assert arc_box((10, 20), (50, 20)) == (10, 20, 50, 20.0)
        assert arc_box((45, 0), (45.5, 1)) == (45, 0, 45.5, 1.0)

    def test_the_pacific_box_is_the_short_way_round(self):
        assert arc_box((10, 170), (20, -170)) == (10, 170, 20, 190.0)
        assert sampled_box((10, 170), (20, -170)) == pytest.approx((10, 170, 20, 190), abs=1e-3)

    def test_a_southern_route_dips_by_the_mirrored_rule(self):
        box = arc_box((-48, 2), (-49, -123))
        assert box[0] == pytest.approx(-67.783, abs=1e-3)
        assert sampled_box((-48, 2), (-49, -123), 20000)[0] == pytest.approx(-67.783, abs=1e-3)

    def test_passes_vertex_reads_the_bearing_crossing(self):
        assert passes_vertex(60, 120)
        assert not passes_vertex(60, 80)
        assert passes_vertex(300, 240)
        assert not passes_vertex(60, 300)

    def test_a_degenerate_arc_is_refused(self):
        with pytest.raises(Invalid):
            arc_box((1, 1), (1, 1))
