from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.sphericalpolygon import (
    disagreement,
    planar_contains,
    polar_cap,
    shifted,
    spherical_contains,
)

COMPACT = [(40.0, 10.0), (40.0, 30.0), (55.0, 30.0), (55.0, 10.0)]
WIDE = [(40.0, -20.0), (40.0, 20.0), (60.0, 20.0), (60.0, -20.0)]


class TestTheThreeFailures:
    def test_a_compact_box_disagrees_only_along_its_bowed_parallels(self):
        rng = random.Random(261)
        points = [(rng.uniform(30, 65), rng.uniform(0, 40)) for _ in range(2000)]
        assert disagreement(COMPACT, points) == pytest.approx(0.005, abs=1e-3)
        inside = sum(spherical_contains(COMPACT, *p) for p in points) / 2000
        assert inside == pytest.approx(0.216, abs=1e-3)

    def test_a_dateline_straddling_box_is_inverted_inside_its_latitude_band(self):
        rng = random.Random(261)
        for _ in range(2000):
            rng.uniform(30, 65)
            rng.uniform(0, 40)
        box = shifted(COMPACT, 160.0)
        assert box == [(40.0, 170.0), (40.0, -170.0), (55.0, -170.0), (55.0, 170.0)]
        points = []
        for _ in range(2000):
            lat, lon = rng.uniform(30, 65), rng.uniform(160, 200)
            points.append((lat, ((lon + 180) % 360) - 180))
        assert disagreement(box, points) == pytest.approx(15 / 35, abs=0.03)
        # clear of the parallels' bulge, which is under a degree for a 20-degree span
        in_band = [p for p in points if 42 < p[0] < 53]
        assert all(planar_contains(box, *p) != spherical_contains(box, *p) for p in in_band)
        # beyond the band and clear of the northern parallel's bulge
        beyond = [p for p in points if p[0] < 39 or p[0] > 56.5]
        assert not any(planar_contains(box, *p) or spherical_contains(box, *p) for p in beyond)

    def test_a_polar_cap_is_missed_entirely_by_the_flat_test(self):
        cap = polar_cap(60.0)
        rng = random.Random(262)
        polar = [(rng.uniform(61, 89), rng.uniform(-180, 180)) for _ in range(300)]
        assert all(spherical_contains(cap, *p) for p in polar)
        assert not any(planar_contains(cap, *p) for p in polar)
        assert spherical_contains(cap, 90, 0) and not planar_contains(cap, 90, 0)
        below = [(rng.uniform(0, 59), rng.uniform(-180, 180)) for _ in range(300)]
        assert not any(spherical_contains(cap, *p) for p in below)

    def test_long_edges_bow_by_a_degree_or_two(self):
        south = [40 + k * 0.05 for k in range(100)]
        first_inside = next(lat for lat in south if spherical_contains(WIDE, lat, 0.0))
        assert first_inside == pytest.approx(41.8, abs=0.06)
        north = [60 + k * 0.05 for k in range(100)]
        last_inside = max(lat for lat in north if spherical_contains(WIDE, lat, 0.0))
        assert last_inside == pytest.approx(61.5, abs=0.06)
        grid = [(lat, lon) for lat in range(38, 63) for lon in range(-22, 23)]
        assert disagreement(WIDE, grid) == pytest.approx(0.1333, abs=1e-3)


class TestTheSphericalTestItself:
    def test_orientation_and_vertices(self):
        rng = random.Random(263)
        backwards = list(reversed(COMPACT))
        for _ in range(100):
            p = (rng.uniform(30, 65), rng.uniform(0, 40))
            assert spherical_contains(backwards, *p) == spherical_contains(COMPACT, *p)
        assert spherical_contains(COMPACT, 40.0, 10.0)

    def test_refusals(self):
        with pytest.raises(Invalid):
            spherical_contains([(0, 0), (1, 1)], 0.5, 0.5)
        with pytest.raises(Outside):
            spherical_contains(COMPACT, 91, 0)
        with pytest.raises(Invalid):
            disagreement(COMPACT, [])
