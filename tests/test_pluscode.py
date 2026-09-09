from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.haversine import haversine
from atlas.pluscode import cell_size_m, contains, decode, encode, shared_prefix

ZURICH = (47.365590, 8.524997)


class TestThePublishedExample:
    def test_zurich_encodes_and_decodes_as_the_standard_says(self):
        assert encode(*ZURICH) == "8FVC9G8F+6X"
        assert encode(*ZURICH, 11) == "8FVC9G8F+6XQ"
        south, west, north, east = decode("8FVC9G8F+6X")
        assert (south, west) == pytest.approx((47.3655, 8.524875))
        assert (north, east) == pytest.approx((47.365625, 8.525))


class TestContainment:
    def test_every_random_point_lies_in_its_own_cell_at_every_length(self):
        rng = random.Random(223)
        for _ in range(1000):
            lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
            for length in (2, 4, 6, 8, 10, 11):
                assert contains(encode(lat, lon, length), lat, lon)

    def test_the_poles_and_the_dateline_stay_in_the_last_and_first_cells(self):
        assert encode(90, 0) == "CFX2X2X2+X2"
        assert encode(0, 180) == "62G22222+22"
        assert encode(-90, -180) == "22222222+22"


class TestCellSizes:
    @pytest.mark.parametrize(
        ("length", "degrees"), [(2, 20.0), (4, 1.0), (6, 0.05), (8, 0.0025), (10, 0.000125)]
    )
    def test_the_schedule(self, length, degrees):
        south, west, north, east = decode(encode(0.0, 0.0, length))
        assert north - south == pytest.approx(degrees)
        assert east - west == pytest.approx(degrees)

    @pytest.mark.parametrize(("lat", "width"), [(0, 13.9), (30, 12.04), (60, 6.95), (80, 2.41)])
    def test_width_follows_the_cosine_of_latitude(self, lat, width):
        height, measured = cell_size_m(encode(lat, 10.0))
        assert height == pytest.approx(13.9, abs=0.01)
        assert measured == pytest.approx(width, abs=0.01)

    def test_the_eleventh_character_refines_five_by_four(self):
        height, width = cell_size_m(encode(0.0, 0.0, 11))
        assert (height, width) == pytest.approx((2.78, 3.47), abs=0.01)
        south, west, north, east = decode(encode(0.0, 0.0, 11))
        assert north - south == pytest.approx(0.000125 / 5)
        assert east - west == pytest.approx(0.000125 / 4)


class TestNamesAreNotDistances:
    def test_a_shared_prefix_bounds_distance_from_above(self):
        rng = random.Random(223)
        worst = 0.0
        for _ in range(1000):
            lat, lon = rng.uniform(-60, 60), rng.uniform(-170, 170)
            lat2, lon2 = lat + rng.uniform(-0.01, 0.01), lon + rng.uniform(-0.01, 0.01)
            if shared_prefix(encode(lat, lon), encode(lat2, lon2)) >= 8:
                south, west, north, east = decode(encode(lat, lon, 8))
                diagonal = haversine(south, west, north, east)
                worst = max(worst, haversine(lat, lon, lat2, lon2) / diagonal)
        assert 0.5 < worst < 1.0

    def test_neighbours_across_a_boundary_share_nothing(self):
        south, _, _, _ = decode(encode(47.0, 8.0, 2))
        a, b = (south - 1e-6, 8.0), (south + 1e-6, 8.0)
        assert shared_prefix(encode(*a), encode(*b)) == 0
        assert haversine(*a, *b) * 1000 < 0.3


class TestRefusals:
    def test_bad_codes_lengths_and_places_are_refused(self):
        for bad in ("8FVC9G8F+6Y", "8FVC9G8", ""):
            with pytest.raises(Invalid):
                decode(bad)
        with pytest.raises(Invalid):
            encode(0, 0, 5)
        with pytest.raises(Outside):
            encode(91, 0)
