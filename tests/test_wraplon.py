from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.wraplon import normalize, to_unit_vector, wrap_longitude


class TestLongitude:
    def test_wrapping_around_the_globe(self):
        assert wrap_longitude(200) == -160.0
        assert wrap_longitude(-540) == -180.0
        assert wrap_longitude(0) == 0.0

    def test_the_seam_follows_one_fixed_convention(self):
        # positive 180 maps to negative 180; never depends on floating residue
        assert wrap_longitude(180) == -180.0
        assert wrap_longitude(-180) == -180.0


class TestLatitude:
    def test_past_the_north_pole_reflects_and_flips_longitude(self):
        assert normalize(100, 30) == (80.0, -150.0)

    def test_past_the_south_pole_reflects_and_flips_longitude(self):
        assert normalize(-100, 30) == (-80.0, -150.0)

    def test_in_range_coordinates_are_untouched(self):
        assert normalize(45.5, -120.25) == (45.5, -120.25)


class TestInvariants:
    def test_idempotent_in_range_and_location_preserving(self):
        rng = random.Random(109)
        for _ in range(20000):
            lat, lon = rng.uniform(-1000, 1000), rng.uniform(-3000, 3000)
            once = normalize(lat, lon)
            twice = normalize(*once)
            assert once == pytest.approx(twice)
            assert -90 <= once[0] <= 90
            assert -180 <= once[1] < 180
            raw = to_unit_vector(lat, lon)
            clean = to_unit_vector(*once)
            assert math.dist(raw, clean) < 1e-12  # measured worst 4e-15


class TestRefusals:
    def test_non_finite_input_is_refused(self):
        with pytest.raises(Invalid):
            wrap_longitude(float("nan"))
        with pytest.raises(Invalid):
            normalize(float("inf"), 0)
