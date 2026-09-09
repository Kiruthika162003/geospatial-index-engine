from __future__ import annotations

import random

import pytest

from atlas.antipode import antipode, half_circumference_km, identity_residual, is_antipodal
from atlas.errors import Outside
from atlas.haversine import haversine


class TestAntipode:
    def test_known_antipodes(self):
        assert antipode(51.5, -0.13) == pytest.approx((-51.5, 179.87))
        assert antipode(0, 180) == pytest.approx((0, 0.0))
        assert antipode(90, 45) == pytest.approx((-90, -135.0))

    def test_the_distance_to_your_own_antipode_is_half_the_circumference(self):
        assert haversine(30, 40, *antipode(30, 40)) == pytest.approx(half_circumference_km())

    def test_antipoding_twice_returns_home(self):
        assert antipode(*antipode(12.5, 34.5)) == pytest.approx((12.5, 34.5))


class TestTheIdentity:
    def test_any_third_point_splits_the_half_circumference_with_the_antipode(self):
        rng = random.Random(169)
        for _ in range(5000):
            la, lo = rng.uniform(-90, 90), rng.uniform(-180, 180)
            tl, to = rng.uniform(-90, 90), rng.uniform(-180, 180)
            assert abs(identity_residual(la, lo, tl, to)) < 1e-6  # measured worst 6.6e-11 km

    def test_no_point_is_farther_than_the_antipode(self):
        rng = random.Random(170)
        for _ in range(2000):
            la, lo = rng.uniform(-90, 90), rng.uniform(-180, 180)
            tl, to = rng.uniform(-90, 90), rng.uniform(-180, 180)
            assert haversine(la, lo, tl, to) <= haversine(la, lo, *antipode(la, lo)) + 1e-9


class TestIsAntipodal:
    def test_exact_and_near_misses(self):
        assert is_antipodal(0, 0, 0, 180)
        assert not is_antipodal(0, 0, 0, 179)


class TestRefusals:
    def test_a_point_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            antipode(91, 0)
        with pytest.raises(Outside):
            antipode(0, 181)
