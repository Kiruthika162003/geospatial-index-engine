from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.shoelace import area
from atlas.wkt import read, worst_error_m, write_linestring, write_point, write_polygon


class TestReadingAndWriting:
    def test_the_three_types_write_and_read(self):
        assert write_point((30, 10)) == "POINT (30.0 10.0)"
        assert write_linestring([(30, 10), (10, 30)]) == "LINESTRING (30.0 10.0, 10.0 30.0)"
        polygon = write_polygon([[(30, 10), (40, 40), (20, 40), (10, 20)]])
        assert polygon == "POLYGON ((30.0 10.0, 40.0 40.0, 20.0 40.0, 10.0 20.0, 30.0 10.0))"
        assert read("POINT (30 10)") == ("POINT", (30.0, 10.0))
        line = read("  linestring(30 10,10 30) ")
        assert line == ("LINESTRING", [(30.0, 10.0), (10.0, 30.0)])
        text = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10), (1 1, 2 2, 1 2, 1 1))"
        kind, rings = read(text)
        assert kind == "POLYGON"
        assert len(rings) == 2
        assert rings[1] == [(1.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]

    def test_empties(self):
        assert read("POINT EMPTY") == ("POINT", None)
        assert read("POLYGON EMPTY") == ("POLYGON", None)
        assert write_polygon([]) == "POLYGON EMPTY"


class TestPrecision:
    def test_shortest_repr_round_trips_losslessly(self):
        rng = random.Random(236)
        for _ in range(1000):
            p = (rng.uniform(-180, 180), rng.uniform(-90, 90))
            assert read(write_point(p))[1] == p
            line = [(rng.uniform(-180, 180), rng.uniform(-90, 90)) for _ in range(5)]
            assert read(write_linestring(line))[1] == line

    @pytest.mark.parametrize(("decimals", "meters"), [(6, 0.0556), (4, 5.56), (2, 556.0)])
    def test_each_dropped_digit_costs_tenfold_half_a_unit(self, decimals, meters):
        rng = random.Random(236 + decimals)
        worst = 0.0
        for _ in range(2000):
            p = (rng.uniform(-180, 180), rng.uniform(-90, 90))
            q = read(write_point(p, decimals))[1]
            worst = max(worst, abs(p[0] - q[0]), abs(p[1] - q[1]))
        assert worst <= 0.5 * 10**-decimals + 1e-12
        assert worst > 0.49 * 10**-decimals
        assert worst_error_m(decimals) == pytest.approx(meters, rel=1e-2)

    def test_area_drift_depends_on_the_geometry_size(self):
        rng = random.Random(236)
        for _ in range(6000):
            rng.uniform(-180, 180)
            rng.uniform(-90, 90)
        country = [(10 + rng.uniform(-3, 3), 50 + rng.uniform(-3, 3)) for _ in range(3)]
        building = []
        for _ in range(3):
            building.append((10 + rng.uniform(-3e-4, 3e-4), 50 + rng.uniform(-3e-4, 3e-4)))
        drifts = []
        for ring in (country, building):
            back = read(write_polygon([ring], 6))[1][0][:-1]
            drifts.append(abs(area(back) - area(ring)) / area(ring))
        assert drifts[0] < 1e-5
        assert drifts[1] > 1e-4
        assert drifts[1] / drifts[0] > 100


class TestRefusals:
    @pytest.mark.parametrize(
        "text",
        [
            "POLYGON ((0 0, 1 0, 1 1, 0 0.5))",
            "LINESTRING (1 1)",
            "CIRCLE (1 1)",
            "POINT (1 1 1)",
            "POINT (a b)",
            "nothing",
        ],
    )
    def test_malformed_text_is_refused(self, text):
        with pytest.raises(Invalid):
            read(text)

    def test_bad_writes_are_refused(self):
        with pytest.raises(Invalid):
            write_linestring([(0, 0)])
        with pytest.raises(Invalid):
            write_polygon([[(0, 0), (1, 1)]])
        with pytest.raises(Invalid):
            worst_error_m(-1)
