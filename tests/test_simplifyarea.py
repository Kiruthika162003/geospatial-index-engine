from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.simplifyarea import (
    area,
    crossing_rate,
    noisy_circle,
    polygon_area_law,
    report,
    self_intersects,
    simplify_ring,
    star,
)


class TestCircles:
    @pytest.mark.parametrize(
        ("tolerance", "vertices", "shrink", "perimeter"),
        [
            (0.1, 104, -0.068, -0.017),
            (1.0, 32, -0.639, -0.16),
            (10.0, 8, -9.964, -2.549),
            (30.0, 4, -36.335, -9.967),
        ],
    )
    def test_a_clean_circle_always_loses_area(self, tolerance, vertices, shrink, perimeter):
        ring = noisy_circle(360, 100.0, 0.0, random.Random(821))
        assert area(ring) == pytest.approx(polygon_area_law(360, 100.0), abs=1e-6)
        read = report(ring, tolerance)
        assert read["vertices"] == vertices
        assert 100 * read["area_change"] == pytest.approx(shrink, abs=1e-3)
        assert 100 * read["perimeter_change"] == pytest.approx(perimeter, abs=1e-3)
        assert read["crossed"] == 0.0

    @pytest.mark.parametrize(
        ("tolerance", "vertices", "area_change", "perimeter"),
        [
            (0.5, 290, 0.01, -0.251),
            (2.0, 159, -0.108, -8.399),
            (5.0, 41, 1.291, -32.042),
            (10.0, 11, -0.501, -41.282),
        ],
    )
    def test_a_rough_ring_sheds_perimeter_not_area(
        self, tolerance, vertices, area_change, perimeter
    ):
        ring = noisy_circle(360, 100.0, 2.0, random.Random(821))
        read = report(ring, tolerance)
        assert read["vertices"] == vertices
        assert 100 * read["area_change"] == pytest.approx(area_change, abs=1e-3)
        assert 100 * read["perimeter_change"] == pytest.approx(perimeter, abs=1e-3)
        assert abs(read["area_change"]) < abs(read["perimeter_change"])


class TestCrossingsAndStars:
    def test_no_rough_ring_crossed(self):
        rings = [noisy_circle(200, 100.0, 3.0, random.Random(830 + k)) for k in range(40)]
        for tolerance in (1.0, 5.0, 20.0):
            assert crossing_rate(rings, tolerance) == 0.0
        assert sum(report(r, 20.0)["vertices"] for r in rings) / 40 == pytest.approx(
            7.8, abs=0.05
        )

    def test_the_star_keeps_its_spikes_until_the_tolerance_passes_them(self):
        ring = star(8, 100.0, 40.0)
        assert [int(report(ring, t)["vertices"]) for t in (1.0, 30.0, 60.0)] == [16, 16, 4]
        assert 100 * report(ring, 60.0)["area_change"] == pytest.approx(63.32, abs=0.01)
        assert simplify_ring(ring, 1.0) == ring

    def test_pieces_and_refusals(self):
        bow = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)]
        assert self_intersects(bow)
        assert not self_intersects([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        assert simplify_ring([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], 5.0) == [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ]
        with pytest.raises(Invalid):
            area([(0.0, 0.0), (1.0, 1.0)])
        with pytest.raises(Invalid):
            noisy_circle(2, 1.0, 0.0, random.Random(1))
        with pytest.raises(Invalid):
            crossing_rate([], 1.0)
