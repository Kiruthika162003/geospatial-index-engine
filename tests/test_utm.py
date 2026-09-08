from __future__ import annotations

import pytest

from atlas.errors import Outside
from atlas.utm import (
    CENTRAL_SCALE,
    band_letter,
    central_meridian,
    point_scale,
    zone_number,
)


class TestZones:
    @pytest.mark.parametrize(
        ("lat", "lon", "zone", "band"),
        [
            (51.5074, -0.1278, 30, "U"),  # London
            (40.7128, -74.0060, 18, "T"),  # New York
            (-33.8688, 151.2093, 56, "H"),  # Sydney
            (59.9139, 10.7522, 32, "V"),  # Oslo
        ],
    )
    def test_known_cities(self, lat, lon, zone, band):
        assert zone_number(lat, lon) == zone
        assert band_letter(lat) == band

    def test_the_norway_exception_widens_zone_32(self):
        # Bergen at lon 5.32 would fall in zone 31 by arithmetic alone
        assert zone_number(60.39, 5.32) == 32

    def test_central_meridians(self):
        assert central_meridian(1) == -177.0
        assert central_meridian(30) == -3.0
        assert central_meridian(60) == 177.0

    def test_bands_skip_i_and_o(self):
        letters = {band_letter(lat) for lat in range(-80, 85)}
        assert "I" not in letters
        assert "O" not in letters


class TestScaleFactor:
    def test_the_meridian_is_scaled_by_0_9996(self):
        assert point_scale(0, central_meridian(31)) == pytest.approx(CENTRAL_SCALE)

    def test_the_equatorial_edge_rises_to_about_a_tenth_of_a_percent(self):
        edge = point_scale(0, central_meridian(31) + 2.999)
        assert edge == pytest.approx(1.000969, abs=1e-5)

    def test_the_whole_zone_stays_within_a_tenth_of_a_percent(self):
        cm = central_meridian(31)
        worst = max(abs(point_scale(0, cm + d / 100) - 1) for d in range(-299, 300))
        assert worst < 0.001  # measured 0.000961

    def test_high_latitude_edges_barely_stretch(self):
        # the meridian offset shrinks with cos(lat), so the edge stays under one
        assert point_scale(60, central_meridian(31) + 2.999) < 1.0


class TestRefusals:
    def test_a_polar_latitude_is_refused(self):
        with pytest.raises(Outside):
            zone_number(85, 0)

    def test_a_bad_zone_is_refused(self):
        with pytest.raises(Outside):
            central_meridian(61)
