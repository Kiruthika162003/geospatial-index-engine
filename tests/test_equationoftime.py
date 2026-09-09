from __future__ import annotations

import pytest

from atlas.equationoftime import (
    analemma,
    analemma_crossing_declination,
    analemma_width_minutes,
    declination,
    equation_of_time_minutes,
    extremes,
    measured_from_positions,
    solar_noon_clock_offset_minutes,
    worst_disagreement,
    zero_crossings,
)
from atlas.errors import Invalid


class TestTheCurve:
    def test_extremes_crossings_and_the_analemma(self):
        (high_day, high), (low_day, low) = extremes()
        assert (high_day, low_day) == (303, 44)
        assert high == pytest.approx(16.453, abs=1e-3)
        assert low == pytest.approx(-14.600, abs=1e-3)
        assert zero_crossings() == [106, 165, 243, 358]
        assert analemma_width_minutes() == pytest.approx(31.053, abs=1e-3)
        assert analemma_crossing_declination() == pytest.approx(1.678, abs=1e-3)
        crossings = [round(declination(d), 2) for d in zero_crossings()]
        assert crossings == [9.78, 23.26, 8.1, -23.4]
        assert len(analemma(5)) == 73

    def test_sample_days(self):
        days = (1, 45, 100, 172, 250, 306, 355)
        readings = [round(equation_of_time_minutes(d), 2) for d in days]
        assert readings == [-3.61, -14.59, -1.59, -1.5, 2.72, 16.4, 1.03]


class TestAgainstTheScan:
    def test_the_formula_disagrees_with_the_minute_scan_by_under_two_minutes(self):
        worst, mean = worst_disagreement(2024)
        assert worst == pytest.approx(1.786, abs=1e-3)
        assert mean == pytest.approx(0.584, abs=1e-3)
        scanned = dict(measured_from_positions(2024))
        assert (scanned[45], scanned[172], scanned[306]) == (-14.0, -2.0, 16.0)


class TestClocks:
    @pytest.mark.parametrize(
        ("day", "lon", "zone", "minutes"),
        [(45, 0.0, 0, 14.59), (172, 13.4, 1, 7.9), (172, -3.7, 1, 76.3), (303, -3.7, 1, 58.35)],
    )
    def test_solar_noon_by_zone_and_longitude(self, day, lon, zone, minutes):
        read = solar_noon_clock_offset_minutes(day, lon, zone)
        assert read == pytest.approx(minutes, abs=0.01)

    def test_refusals(self):
        with pytest.raises(Invalid):
            equation_of_time_minutes(0)
        with pytest.raises(Invalid):
            declination(367)
        with pytest.raises(Invalid):
            solar_noon_clock_offset_minutes(10, 200, 0)
        with pytest.raises(Invalid):
            analemma(0)
