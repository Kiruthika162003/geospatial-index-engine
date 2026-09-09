from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.solarradiation import (
    EQUINOX_SPRING,
    SOLSTICE_SUMMER,
    SOLSTICE_WINTER,
    annual_beam,
    best_tilt,
    daily_beam,
    day_length_hours,
    declination,
    noon_elevation,
    shaded_all_day,
    surface_normal,
)


class TestSlopesAtLatitude45:
    @pytest.mark.parametrize(
        ("day", "flat", "south", "ratio"),
        [
            (SOLSTICE_WINTER, 10.066, 30.931, 3.073),
            (EQUINOX_SPRING, 26.174, 37.429, 1.43),
            (SOLSTICE_SUMMER, 43.141, 34.341, 0.796),
        ],
    )
    def test_a_south_slope_triples_winter_and_loses_a_fifth_in_summer(
        self, day, flat, south, ratio
    ):
        assert daily_beam(45, day) == pytest.approx(flat, abs=1e-3)
        assert daily_beam(45, day, 45, 180) == pytest.approx(south, abs=1e-3)
        read = daily_beam(45, day, 45, 180) / daily_beam(45, day)
        assert read == pytest.approx(ratio, abs=1e-3)

    def test_a_north_slope_steeper_than_the_noon_sun_is_dark_in_winter(self):
        assert noon_elevation(45, SOLSTICE_WINTER) == pytest.approx(21.56, abs=0.01)
        assert shaded_all_day(45, SOLSTICE_WINTER, 30, 0)
        assert shaded_all_day(45, SOLSTICE_WINTER, 25, 0)
        assert not shaded_all_day(45, SOLSTICE_WINTER, 20, 0)
        assert day_length_hours(45, SOLSTICE_WINTER) == pytest.approx(8.574, abs=1e-3)
        assert day_length_hours(45, SOLSTICE_SUMMER) == pytest.approx(15.426, abs=1e-3)


class TestOtherLatitudes:
    def test_the_far_north_in_winter(self):
        assert day_length_hours(65, SOLSTICE_WINTER) == pytest.approx(2.88, abs=1e-3)
        assert noon_elevation(65, SOLSTICE_WINTER) == pytest.approx(1.56, abs=0.01)
        flat = daily_beam(65, SOLSTICE_WINTER)
        south = daily_beam(65, SOLSTICE_WINTER, 45, 180)
        assert flat == pytest.approx(0.256, abs=1e-3)
        assert south / flat == pytest.approx(39.23, abs=0.01)
        assert day_length_hours(65, SOLSTICE_SUMMER) == pytest.approx(21.12, abs=1e-3)
        assert day_length_hours(70, SOLSTICE_SUMMER) == 24.0
        assert day_length_hours(70, SOLSTICE_WINTER) == 0.0

    def test_the_equator(self):
        december = daily_beam(0, SOLSTICE_WINTER, 45, 180) / daily_beam(0, SOLSTICE_WINTER)
        june = daily_beam(0, SOLSTICE_SUMMER, 45, 180) / daily_beam(0, SOLSTICE_SUMMER)
        assert december == pytest.approx(1.19, abs=1e-3)
        assert june == pytest.approx(0.2931, abs=1e-3)
        assert day_length_hours(0, EQUINOX_SPRING) == pytest.approx(12.0, abs=1e-3)


class TestTheYear:
    @pytest.mark.parametrize(
        ("lat", "dry", "hazy"), [(0, 0, 0), (30, 30, 25), (45, 45, 35), (60, 55, 45)]
    )
    def test_the_best_tilt_is_the_latitude_only_without_air(self, lat, dry, hazy):
        cheap = {"day_step": 7, "step_minutes": 10}
        assert best_tilt(lat, **cheap)[0] == dry
        assert best_tilt(lat, transmittance=0.7, **cheap)[0] == hazy
        assert hazy <= dry

    def test_annual_totals(self):
        cheap = {"day_step": 7, "step_minutes": 10}
        assert annual_beam(0, **cheap) == pytest.approx(13085.8, abs=0.1)
        assert annual_beam(0) == pytest.approx(13096.2, abs=0.1)
        dry = annual_beam(90, **cheap) / annual_beam(0, **cheap)
        assert dry == pytest.approx(0.4127, abs=1e-3)
        pole = annual_beam(90, transmittance=0.7, **cheap)
        hazy = pole / annual_beam(0, transmittance=0.7, **cheap)
        assert hazy == pytest.approx(0.2154, abs=1e-3)

    def test_the_step_and_the_declination(self):
        fine = daily_beam(45, SOLSTICE_SUMMER, 45, 180, step_minutes=1)
        usual = daily_beam(45, SOLSTICE_SUMMER, 45, 180, step_minutes=2)
        hourly = daily_beam(45, SOLSTICE_SUMMER, 45, 180, step_minutes=60)
        assert usual == pytest.approx(fine, abs=2e-4)
        assert hourly / fine == pytest.approx(0.9943, abs=1e-3)
        readings = [round(declination(d), 3) for d in (1, 80, 172, 355)]
        assert readings == [-23.002, -0.403, 23.44, -23.44]
        assert surface_normal(0, 0) == pytest.approx((0.0, 0.0, 1.0))


class TestRefusals:
    def test_bad_days_latitudes_slopes_and_steps(self):
        with pytest.raises(Invalid):
            declination(0)
        with pytest.raises(Invalid):
            daily_beam(91, 100)
        with pytest.raises(Invalid):
            daily_beam(45, 100, 95, 180)
        with pytest.raises(Invalid):
            daily_beam(45, 100, step_minutes=7)
        with pytest.raises(Invalid):
            daily_beam(45, 100, transmittance=0)
        with pytest.raises(Invalid):
            best_tilt(45, step=0)
        with pytest.raises(Invalid):
            annual_beam(45, day_step=0)
