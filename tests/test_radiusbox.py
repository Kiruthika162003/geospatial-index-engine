from __future__ import annotations

import random

import pytest

from atlas.bearing import destination
from atlas.errors import Outside
from atlas.radiusbox import contains, edge_cosine_box, naive_box, radius_box

OFF_AXIS = list(range(5, 360, 10))


def _rim_misses(box_fn, rng, trials, lat_lo, lat_hi, bearings):
    missed = total = 0
    for _ in range(trials):
        lat, lon, r = rng.uniform(lat_lo, lat_hi), rng.uniform(-100, 100), rng.uniform(1, 800)
        box = box_fn(lat, lon, r)
        for b in bearings:
            total += 1
            if not contains(box, *destination(lat, lon, b, r)):
                missed += 1
    return missed, total


class TestContainment:
    def test_the_exact_box_contains_every_off_axis_rim_point(self):
        missed, _ = _rim_misses(radius_box, random.Random(89), 3000, -70, 70, OFF_AXIS)
        assert missed == 0

    def test_the_exact_north_and_south_rim_points_are_inside_by_construction(self):
        # they land on the latitude edge within 6e-14 degrees; the pad absorbs the tie
        missed, _ = _rim_misses(radius_box, random.Random(88), 3000, -75, 75, (0, 180))
        assert missed == 0

    def test_the_edge_cosine_box_also_contains_the_rim(self):
        missed, _ = _rim_misses(edge_cosine_box, random.Random(89), 3000, -70, 70, OFF_AXIS)
        assert missed == 0

    def test_the_naive_center_cosine_box_leaks_at_high_latitude(self):
        missed, total = _rim_misses(naive_box, random.Random(87), 2000, 60, 80, OFF_AXIS)
        assert missed / total > 0.05  # measured about 9 percent


class TestTightness:
    def test_the_exact_box_cannot_shrink_one_percent(self):
        rng = random.Random(89)
        escapes = total = 0
        for _ in range(1000):
            lat, lon, r = rng.uniform(-70, 70), rng.uniform(-100, 100), rng.uniform(1, 800)
            mn, mnl, mx, mxl = radius_box(lat, lon, r)
            half = (mxl - mnl) / 2 * 0.99
            c = (mnl + mxl) / 2
            shrunk = (mn, c - half, mx, c + half)
            for b in OFF_AXIS:
                total += 1
                if not contains(shrunk, *destination(lat, lon, b, r)):
                    escapes += 1
        assert escapes / total > 0.05  # measured about 10 percent

    def test_the_edge_cosine_box_overshoots_the_true_spread(self):
        exact = radius_box(70, 0, 800)
        edge = edge_cosine_box(70, 0, 800)
        exact_half = (exact[3] - exact[1]) / 2
        edge_half = (edge[3] - edge[1]) / 2
        assert exact_half == pytest.approx(21.48, abs=0.05)
        assert edge_half == pytest.approx(32.46, abs=0.05)

    def test_the_overshoot_is_systematic_across_random_circles(self):
        rng = random.Random(89)
        ratios = []
        for _ in range(3000):
            lat, lon, r = rng.uniform(-70, 70), rng.uniform(-100, 100), rng.uniform(1, 800)
            e = radius_box(lat, lon, r)
            c = edge_cosine_box(lat, lon, r)
            ratios.append((c[3] - c[1]) / (e[3] - e[1]))
        assert min(ratios) >= 1.0 - 1e-9  # never narrower than the truth
        assert max(ratios) > 1.3  # measured max 1.46
        assert sum(ratios) / len(ratios) > 1.03  # measured mean 1.066


class TestPoleAndDateline:
    def test_a_circle_over_the_pole_spans_all_longitudes(self):
        box = radius_box(88, 0, 500)
        assert box[1] == -180.0
        assert box[3] == 180.0
        assert box[2] == 90.0

    def test_a_dateline_box_wraps(self):
        box = radius_box(0, 179.5, 100)
        assert box[1] > box[3]  # min_lon > max_lon signals the wrap
        assert contains(box, 0, 179.9)
        assert contains(box, 0, -179.9)
        assert not contains(box, 0, 170)


class TestRefusals:
    def test_a_negative_radius_is_refused(self):
        with pytest.raises(Outside):
            radius_box(0, 0, -1)

    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            radius_box(95, 0, 10)
