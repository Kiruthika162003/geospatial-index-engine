from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from atlas.errors import Invalid, Missing, Outside
from atlas.sunposition import (
    elevation_profile,
    position,
    rise_and_set_azimuths,
    solar_coordinates,
    solar_noon_utc_minutes,
)

EQUINOX = datetime(2024, 3, 20, tzinfo=UTC)
SOLSTICE = datetime(2024, 6, 21, tzinfo=UTC)


class TestTheEquinox:
    @pytest.mark.parametrize(
        ("lat", "elevation"), [(0, 89.83), (30, 60.15), (51.5, 38.65), (-33.9, 55.95)]
    )
    def test_noon_elevation_is_ninety_minus_latitude_within_the_days_declination(
        self, lat, elevation
    ):
        noon = solar_noon_utc_minutes(EQUINOX, lat, 0.0)
        _, el = position(EQUINOX + timedelta(minutes=noon), lat, 0.0)
        assert el == pytest.approx(elevation, abs=0.02)
        assert abs(el - (90 - abs(lat))) < 0.2

    def test_the_declination_is_a_seventh_of_a_degree_nine_hours_after_the_instant(self):
        assert solar_coordinates(EQUINOX.replace(hour=12, minute=7))[1] == pytest.approx(
            0.15, abs=0.01
        )

    @pytest.mark.parametrize("lat", [0, 30, 51.5, -33.9])
    def test_the_sun_rises_due_east_and_sets_due_west(self, lat):
        rise, set_ = rise_and_set_azimuths(EQUINOX, lat, 0.0)
        assert rise == pytest.approx(90.0, abs=0.35)
        assert set_ == pytest.approx(270.0, abs=0.4)

    def test_noon_azimuth_is_south_in_the_north_and_north_in_the_south(self):
        noon = solar_noon_utc_minutes(EQUINOX, 51.5, 0.0)
        assert position(EQUINOX + timedelta(minutes=noon), 51.5, 0.0)[0] == pytest.approx(
            180.0, abs=0.3
        )
        noon = solar_noon_utc_minutes(EQUINOX, -33.9, 0.0)
        assert position(EQUINOX + timedelta(minutes=noon), -33.9, 0.0)[0] == pytest.approx(
            0.0, abs=0.3
        )


class TestTheSolstice:
    def test_the_tropic_sees_the_sun_overhead(self):
        noon = solar_noon_utc_minutes(SOLSTICE, 23.44, 0.0)
        assert position(SOLSTICE + timedelta(minutes=noon), 23.44, 0.0)[1] == pytest.approx(
            89.98, abs=0.02
        )

    def test_the_arctic_sun_never_sets(self):
        lowest = min(el for _, (_, el) in elevation_profile(SOLSTICE, 70.0, 0.0, 60))
        assert lowest == pytest.approx(3.44, abs=0.02)
        with pytest.raises(Missing):
            rise_and_set_azimuths(SOLSTICE, 70.0, 0.0)


class TestTheEquationOfTime:
    def test_the_swing_matches_the_almanac_within_the_scan_resolution(self):
        february = 720 - solar_noon_utc_minutes(datetime(2024, 2, 11, tzinfo=UTC), 51.5, 0.0)
        november = 720 - solar_noon_utc_minutes(datetime(2024, 11, 3, tzinfo=UTC), 51.5, 0.0)
        april = 720 - solar_noon_utc_minutes(datetime(2024, 4, 15, tzinfo=UTC), 51.5, 0.0)
        assert february == pytest.approx(-15.0, abs=1.0)
        assert november == pytest.approx(17.0, abs=1.0)
        assert abs(april) <= 1.0


class TestRefusals:
    def test_naive_times_and_off_globe_places_are_refused(self):
        with pytest.raises(Invalid):
            position(datetime(2024, 1, 1), 0, 0)
        with pytest.raises(Outside):
            position(EQUINOX, 91, 0)
        with pytest.raises(Outside):
            position(EQUINOX, 0, 181)
        with pytest.raises(Invalid):
            elevation_profile(EQUINOX, 0, 0, 0)
