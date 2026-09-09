from __future__ import annotations

import math

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.minkowski import (
    area,
    buffer_area,
    is_convex,
    minkowski_sum,
    mixed_area,
    perimeter,
    regular,
)

TRIANGLE = [(0.0, 0.0), (4.0, 0.0), (1.0, 3.0)]
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
HEXAGON = regular(6, 2.0)


class TestAreaLaws:
    def test_p_plus_p_quadruples_the_area_with_mixed_area_equal_to_the_area(self):
        doubled = minkowski_sum(TRIANGLE, TRIANGLE)
        assert area(doubled) == pytest.approx(24.0)
        assert mixed_area(TRIANGLE, TRIANGLE) == pytest.approx(6.0)
        assert len(doubled) == 3

    def test_a_square_adds_the_side_times_the_extents_not_half_the_perimeter(self):
        assert area(minkowski_sum(TRIANGLE, SQUARE)) == pytest.approx(14.0)
        guessed = area(TRIANGLE) + perimeter(TRIANGLE) * 0.5 + 1.0
        assert guessed == pytest.approx(12.7025, abs=1e-3)
        assert area(TRIANGLE) + 1.0 * (4.0 + 3.0) + 1.0 == pytest.approx(14.0)
        hexagon_sum = area(minkowski_sum(HEXAGON, SQUARE))
        assert hexagon_sum == pytest.approx(18.8564, abs=1e-3)
        assert hexagon_sum == pytest.approx(area(HEXAGON) + (4.0 + 2 * math.sqrt(3)) + 1.0)


class TestVertices:
    def test_parallel_edges_merge(self):
        assert len(minkowski_sum(TRIANGLE, SQUARE)) == 6
        assert len(minkowski_sum(HEXAGON, SQUARE)) == 8
        assert len(minkowski_sum(SQUARE, SQUARE)) == 4
        assert area(minkowski_sum(SQUARE, SQUARE)) == pytest.approx(4.0)

    def test_commutative_and_translation_covariant(self):
        def canon(polygon, dx: float = 0.0, dy: float = 0.0):
            return sorted((round(x + dx, 9), round(y + dy, 9)) for x, y in polygon)

        a = minkowski_sum(TRIANGLE, HEXAGON)
        b = minkowski_sum(HEXAGON, TRIANGLE)
        assert canon(a) == canon(b)
        assert len(a) == 8
        shifted = [(x + 3.0, y - 2.0) for x, y in TRIANGLE]
        assert canon(minkowski_sum(shifted, HEXAGON)) == canon(a, 3.0, -2.0)


class TestBufferApproximation:
    @pytest.mark.parametrize(
        ("sides", "shortfall"),
        [(8, 3.1417), (16, 0.637), (32, 0.1898), (64, 0.0434), (256, 0.0024)],
    )
    def test_the_shortfall_quarters_per_doubling_of_sides(self, sides, shortfall):
        total = area(minkowski_sum(TRIANGLE, regular(sides, 1.5)))
        measured = 100 * (1 - total / buffer_area(TRIANGLE, 1.5))
        assert measured == pytest.approx(shortfall, abs=1e-3)


class TestRefusals:
    def test_non_convex_and_degenerate_inputs_are_refused(self):
        assert is_convex(TRIANGLE)
        assert not is_convex([(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)])
        with pytest.raises(Invalid):
            minkowski_sum([(0, 0), (2, 0), (1, 0.5), (2, 2), (0, 2)], SQUARE)
        with pytest.raises(Invalid):
            minkowski_sum([(0, 0), (1, 1)], SQUARE)
        with pytest.raises(Degenerate):
            minkowski_sum([(0, 0), (1, 1), (2, 2)], SQUARE)
        with pytest.raises(Invalid):
            regular(2, 1.0)
