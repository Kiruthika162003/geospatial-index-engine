from __future__ import annotations

import random

import pytest

from atlas.coordprecision import (
    area_change,
    collapsed,
    damage_rates,
    decimals_for,
    float32_shift_m,
    float32_ulp_m,
    measured_shift_m,
    rounded,
    self_intersects,
    small_ring,
    worst_shift_m,
)
from atlas.errors import Invalid


class TestDecimals:
    @pytest.mark.parametrize(
        ("decimals", "law", "worst", "mean"),
        [(2, 787.148, 784.84, 424.184), (4, 7.871, 7.771, 4.288), (5, 0.787, 0.773, 0.427)],
    )
    def test_the_formula_holds_within_a_third_of_a_percent(self, decimals, law, worst, mean):
        assert worst_shift_m(0, decimals) == pytest.approx(law, abs=1e-3)
        read_worst, read_mean = measured_shift_m(0, decimals, random.Random(340 + decimals))
        assert read_worst == pytest.approx(worst, abs=1e-3)
        assert read_mean == pytest.approx(mean, abs=1e-3)
        assert read_worst / law > 0.98
        assert read_mean / read_worst == pytest.approx(0.545, abs=0.01)

    def test_high_latitudes_shrink_the_east_axis(self):
        assert worst_shift_m(80, 3) / worst_shift_m(0, 3) == pytest.approx(0.7177, abs=1e-3)
        worst, _ = measured_shift_m(80, 3, random.Random(343))
        assert worst == pytest.approx(56.275, abs=1e-3)

    def test_decimals_for_a_tolerance(self):
        assert decimals_for(1.0) == 5
        assert decimals_for(1.0, 60) == 5
        assert decimals_for(10.0) == 4
        assert decimals_for(0.001) == 8
        assert rounded(12.3456789, -0.9876543, 5) == (12.34568, -0.98765)


class TestFloat32:
    @pytest.mark.parametrize(
        ("lon", "ulp", "worst"),
        [
            (0.5, 0.0066, 0.0036),
            (10.0, 0.1062, 0.0527),
            (100.0, 0.8493, 0.423),
            (179.9, 1.6986, 0.8479),
        ],
    )
    def test_the_last_place_grows_with_the_longitude(self, lon, ulp, worst):
        assert float32_ulp_m(lon) == pytest.approx(ulp, abs=1e-4)
        rng = random.Random(341)
        probes = [(rng.uniform(-0.5, 0.5), lon + rng.uniform(0, 0.001)) for _ in range(500)]
        read = max(float32_shift_m(lat, x) for lat, x in probes)
        assert read == pytest.approx(worst, abs=1e-4)

    def test_high_latitudes_shift_too(self):
        rng = random.Random(342)
        read = max(float32_shift_m(rng.uniform(80, 89), 0.0) for _ in range(500))
        assert read == pytest.approx(0.4232, abs=1e-4)


class TestRings:
    @pytest.mark.parametrize(
        ("radius", "repeated", "fallen"),
        [
            (2e-3, 0.278, 0.0),
            (1e-3, 0.99, 0.0),
            (5e-4, 1.0, 0.508),
            (2e-4, 1.0, 0.932),
            (1e-2, 0.0, 0.0),
        ],
    )
    def test_rounding_repeats_and_collapses_but_does_not_cross(self, radius, repeated, fallen):
        rates = damage_rates(radius, 3, random.Random(343))
        assert rates["crossed"] == 0.0
        assert rates["repeated"] == pytest.approx(repeated, abs=1e-3)
        assert rates["collapsed"] == pytest.approx(fallen, abs=1e-3)

    def test_twenty_vertices_cross_in_a_few_per_thousand(self):
        assert damage_rates(5e-3, 3, random.Random(345), vertices=20)["crossed"] == 0.004
        assert damage_rates(2e-3, 3, random.Random(345), vertices=20)["crossed"] == 0.002

    def test_area_change_by_cells_across(self):
        rng = random.Random(344)
        readings = []
        for radius, decimals in ((1e-2, 3), (1e-3, 3), (1e-3, 4), (1e-4, 5)):
            changes = []
            for _ in range(300):
                centre = (rng.uniform(-60, 60), rng.uniform(-180, 180))
                changes.append(abs(area_change(small_ring(centre, radius, 8, rng), decimals)))
            readings.append((round(sum(changes) / 300, 4), round(max(changes), 4)))
        assert readings[0] == (0.0223, 0.0769)
        assert readings[1] == (0.2266, 0.974)
        assert readings[2] == (0.0228, 0.0807)
        assert readings[3] == (0.0233, 0.0791)

    def test_the_geometry_helpers(self):
        bow = [(0, 0), (1, 1), (1, 0), (0, 1)]
        assert self_intersects(bow)
        assert not self_intersects([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert collapsed([(0, 0), (1, 1), (2, 2)])
        assert collapsed([(0, 0), (0, 0), (1, 1)])


class TestRefusals:
    def test_bad_decimals_rings_and_tolerances(self):
        with pytest.raises(Invalid):
            rounded(0, 0, -1)
        with pytest.raises(Invalid):
            small_ring((0, 0), 1e-3, 2, random.Random(1))
        with pytest.raises(Invalid):
            area_change([(0, 0), (1, 1), (2, 2)], 3)
        with pytest.raises(Invalid):
            decimals_for(0)
