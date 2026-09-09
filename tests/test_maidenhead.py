from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.haversine import haversine
from atlas.maidenhead import aspect_on_ground, cell_size_m, center, contains, decode, encode


class TestKnownLocators:
    def test_munich_and_the_cell_of_a_subsquare(self):
        assert encode(48.1, 11.6) == "JN58TC"
        assert encode(48.1, 11.6, 4) == "JN58"
        south, west, north, east = decode("JN58td")
        assert (south, north) == pytest.approx((48.125, 48.1666667))
        assert (west, east) == pytest.approx((11.5833333, 11.6666667))

    def test_the_poles_land_in_the_first_and_last_cells(self):
        assert encode(-90, -180) == "AA00AA"
        assert encode(90, 180) == "RR99XX"


class TestContainmentAndCentering:
    def test_every_point_is_inside_its_cell_and_the_center_re_encodes(self):
        rng = random.Random(224)
        worst = 0.0
        for _ in range(1000):
            lat, lon = rng.uniform(-90, 90), rng.uniform(-180, 180)
            for length in (2, 4, 6, 8):
                locator = encode(lat, lon, length)
                assert contains(locator, lat, lon)
                clat, clon = center(locator)
                assert encode(clat, clon, length) == locator
                south, west, north, east = decode(locator)
                lat_offset = abs(clat - lat) / (north - south)
                lon_offset = abs(clon - lon) / (east - west)
                worst = max(worst, lat_offset, lon_offset)
        assert worst <= 0.5 + 1e-9


class TestCellShapes:
    @pytest.mark.parametrize(
        ("lat", "width_km", "aspect"),
        [(0, 9.27, 2.0), (30, 8.02, 1.732), (60, 4.63, 0.999), (75, 2.4, 0.517)],
    )
    def test_subsquares_are_square_on_the_ground_only_at_sixty(self, lat, width_km, aspect):
        locator = encode(lat, 10.0, 6)
        height, width = cell_size_m(locator)
        assert height / 1000 == pytest.approx(4.63, abs=0.01)
        assert width / 1000 == pytest.approx(width_km, abs=0.01)
        assert aspect_on_ground(locator) == pytest.approx(aspect, abs=2e-3)

    def test_the_degree_schedule(self):
        for length, lat_deg, lon_deg in ((2, 10.0, 20.0), (4, 1.0, 2.0), (6, 1 / 24, 1 / 12)):
            south, west, north, east = decode(encode(0, 0, length))
            assert north - south == pytest.approx(lat_deg)
            assert east - west == pytest.approx(lon_deg)

    def test_field_centers_and_subsquare_diagonals(self):
        assert haversine(*center("JJ"), *center("KJ")) == pytest.approx(2215.4, abs=0.1)
        assert haversine(*center("JJ"), *center("JK")) == pytest.approx(1112.0, abs=0.1)
        assert haversine(*center("RR"), *center("QR")) == pytest.approx(192.9, abs=0.1)
        for lat, diagonal in ((0, 10.36), (60, 6.55)):
            south, west, north, east = decode(encode(lat, 10.0, 6))
            assert haversine(south, west, north, east) == pytest.approx(diagonal, abs=0.01)


class TestRefusals:
    def test_bad_locators_lengths_and_places_are_refused(self):
        for bad in ("SS", "JN58yz", "JNAA", "J"):
            with pytest.raises(Invalid):
                decode(bad)
        with pytest.raises(Invalid):
            encode(0, 0, 3)
        with pytest.raises(Outside):
            encode(0, 181)
