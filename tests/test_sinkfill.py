from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.sinkfill import (
    bowl,
    bowl_volume_law,
    crater_on_slope,
    fill_summary,
    interior_sink_count,
    noisy,
    priority_flood,
    spill_height,
    tilted_plane,
)


class TestNoise:
    def test_a_plane_takes_no_fill(self):
        plane = tilted_plane(61, 1.0)
        assert interior_sink_count(plane) == 0
        assert fill_summary(plane, priority_flood(plane))["volume"] == 0.0

    @pytest.mark.parametrize(
        ("sigma", "before", "flats", "raised", "mean", "worst", "volume"),
        [
            (0.3, 1, 1, 0.0003, 0.1447, 0.1447, 0.1447),
            (1.0, 211, 244, 0.0656, 0.5351, 2.7472, 130.5595),
            (3.0, 374, 654, 0.1758, 1.8697, 8.2155, 1222.7727),
        ],
    )
    def test_flats_outnumber_sinks_until_an_epsilon_tilts_them(
        self, sigma, before, flats, raised, mean, worst, volume
    ):
        dem = noisy(tilted_plane(61, 1.0), sigma, random.Random(420))
        assert interior_sink_count(dem) == before
        filled = priority_flood(dem)
        assert interior_sink_count(filled) == flats
        assert interior_sink_count(priority_flood(dem, 1e-6)) == 0
        summary = fill_summary(dem, filled)
        assert summary["raised_fraction"] == pytest.approx(raised, abs=1e-4)
        assert summary["mean_depth"] == pytest.approx(mean, abs=1e-4)
        assert summary["max_depth"] == pytest.approx(worst, abs=1e-4)
        assert summary["volume"] == pytest.approx(volume, abs=1e-3)


class TestBowlsAndCraters:
    @pytest.mark.parametrize(("depth", "volume"), [(5.0, 7068.133), (20.0, 28272.533)])
    def test_a_bowl_fills_to_its_rim_at_the_law(self, depth, volume):
        dem = bowl(61, depth)
        filled = priority_flood(dem)
        summary = fill_summary(dem, filled)
        assert summary["volume"] == pytest.approx(volume, abs=1e-3)
        assert bowl_volume_law(61, depth) == pytest.approx(volume, abs=1e-3)
        assert summary["max_depth"] == pytest.approx(depth, abs=1e-9)
        assert summary["raised_fraction"] == pytest.approx(0.7549, abs=1e-4)
        assert interior_sink_count(filled) == 3481

    @pytest.mark.parametrize(
        ("slope", "volume", "deepest", "spill"),
        [(0.0, 2260.556, 10.0, 0.0), (0.1, 1765.958, 8.8306, 1.8), (0.5, 543.028, 4.8889, 9.0)],
    )
    def test_a_crater_keeps_the_water_below_its_low_rim(self, slope, volume, deepest, spill):
        dem = crater_on_slope(61, slope, 10.0, 12.0)
        filled = priority_flood(dem)
        summary = fill_summary(dem, filled)
        assert summary["volume"] == pytest.approx(volume, abs=1e-3)
        assert summary["max_depth"] == pytest.approx(deepest, abs=1e-4)
        assert spill_height(filled, 30, 30) == pytest.approx(spill, abs=1e-9)
        assert spill == pytest.approx(dem[30][18], abs=1e-9)


class TestRefusals:
    def test_bad_grids_and_epsilons(self):
        with pytest.raises(Invalid):
            priority_flood([])
        with pytest.raises(Invalid):
            priority_flood([[1.0]], -1.0)
        with pytest.raises(Invalid):
            interior_sink_count([])
