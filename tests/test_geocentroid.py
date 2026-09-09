from __future__ import annotations

import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.geocentroid import centroid, chord_cost, concentration, coordinate_mean
from atlas.haversine import haversine

DATELINE_PAIR = [(0.0, 179.0), (0.0, -179.0)]
MERIDIAN_PAIR = [(0.0, -1.0), (0.0, 1.0)]
RING = [(60.0, float(lon)) for lon in range(-90, 91, 10)]


class TestTheDateline:
    def test_coordinate_averaging_lands_on_the_far_side_of_the_globe(self):
        assert centroid(DATELINE_PAIR) == pytest.approx((0.0, 180.0))
        assert coordinate_mean(DATELINE_PAIR) == pytest.approx((0.0, 0.0))
        gap = haversine(*centroid(DATELINE_PAIR), *coordinate_mean(DATELINE_PAIR))
        assert gap == pytest.approx(20015.114, abs=1e-3)

    def test_the_same_pair_shifted_to_the_prime_meridian_agrees(self):
        assert centroid(MERIDIAN_PAIR) == pytest.approx((0.0, 0.0), abs=1e-9)
        assert coordinate_mean(MERIDIAN_PAIR) == pytest.approx((0.0, 0.0))


class TestTheRing:
    def test_the_vector_mean_sits_poleward_of_the_ring(self):
        lat, lon = centroid(RING)
        assert lat == pytest.approx(70.847, abs=1e-3)
        assert lon == pytest.approx(0.0, abs=1e-9)
        assert coordinate_mean(RING) == pytest.approx((60.0, 0.0))
        assert haversine(lat, lon, 60.0, 0.0) == pytest.approx(1206.1, abs=0.1)

    def test_the_vector_mean_costs_less_than_the_coordinate_mean(self):
        assert chord_cost(RING, centroid(RING)) == pytest.approx(3.163, abs=1e-3)
        assert chord_cost(RING, coordinate_mean(RING)) == pytest.approx(3.785, abs=1e-3)
        assert concentration(RING) == pytest.approx(0.9168, abs=1e-4)


class TestMinimization:
    def test_no_perturbation_lowers_the_chord_cost(self):
        rng = random.Random(173)
        for _ in range(300):
            count = rng.randint(2, 12)
            pts = [(rng.uniform(-80, 80), rng.uniform(-180, 180)) for _ in range(count)]
            try:
                v = centroid(pts)
            except Degenerate:
                continue
            base = chord_cost(pts, v)
            for _ in range(20):
                cand = (max(-90.0, min(90.0, v[0] + rng.gauss(0, 5))), v[1] + rng.gauss(0, 5))
                assert chord_cost(pts, cand) >= base - 1e-9


class TestConcentration:
    def test_a_single_spot_is_fully_concentrated(self):
        assert concentration([(10.0, 20.0)]) == pytest.approx(1.0)

    def test_a_balanced_spread_has_no_direction(self):
        octahedron = [(90, 0), (-90, 0), (0, 0), (0, 90), (0, 180), (0, -90)]
        assert concentration(octahedron) < 1e-12
        with pytest.raises(Degenerate):
            centroid(octahedron)
        assert concentration([(0, 0), (0, 180)]) < 1e-12


class TestRefusals:
    def test_no_points_is_refused(self):
        with pytest.raises(Invalid):
            centroid([])
        with pytest.raises(Invalid):
            coordinate_mean([])
