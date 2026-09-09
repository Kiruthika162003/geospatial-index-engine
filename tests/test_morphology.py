from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.morphology import (
    area,
    blank,
    closing,
    cross_element,
    dilate,
    erode,
    opening,
    paint_square,
    square_element,
)

E1 = square_element(1)


def _square() -> list[list[bool]]:
    return paint_square(blank(30, 30), 10, 10, 10)


class TestDilationAndErosion:
    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_areas_follow_the_perimeter_and_corner_formula(self, radius):
        square = _square()
        element = square_element(radius)
        assert area(dilate(square, element)) == (10 + 2 * radius) ** 2
        assert area(erode(square, element)) == (10 - 2 * radius) ** 2
        assert area(dilate(square, element)) - 100 == 40 * radius + 4 * radius * radius

    def test_the_two_invert_each_other_on_a_coarse_shape(self):
        square = _square()
        assert erode(dilate(square, E1), E1) == square
        assert dilate(erode(square, E1), E1) == square

    def test_dilation_is_not_idempotent(self):
        square = _square()
        assert area(dilate(dilate(square, E1), E1)) == 196
        assert area(dilate(square, E1)) == 144

    def test_the_border_counts_as_unset(self):
        corner = paint_square(blank(10, 10), 0, 0, 5)
        assert area(erode(corner, E1)) == 9


class TestOpeningAndClosing:
    def test_opening_removes_a_spur_and_a_small_patch(self):
        spur = _square()
        spur[15][20] = True
        assert area(spur) == 101
        assert opening(spur, E1) == _square()
        small = blank(30, 30)
        small[5][5] = small[5][6] = True
        assert area(opening(small, E1)) == 0

    def test_closing_fills_a_gap_and_a_hole(self):
        two = paint_square(paint_square(blank(30, 30), 10, 5, 10), 10, 16, 10)
        assert area(two) == 200
        assert area(closing(two, E1)) == 210
        holed = _square()
        holed[15][15] = False
        assert area(closing(holed, E1)) == 100

    def test_both_are_idempotent(self):
        spur = _square()
        spur[15][20] = True
        two = paint_square(paint_square(blank(30, 30), 10, 5, 10), 10, 16, 10)
        assert opening(opening(spur, E1), E1) == opening(spur, E1)
        assert closing(closing(two, E1), E1) == closing(two, E1)


class TestElements:
    def test_a_lone_cell_grows_to_nine_or_five(self):
        lone = blank(9, 9)
        lone[4][4] = True
        assert area(dilate(lone, E1)) == 9
        assert area(dilate(lone, cross_element(1))) == 5
        assert area(dilate(lone, square_element(2))) == 25
        assert area(dilate(lone, cross_element(2))) == 13

    def test_refusals(self):
        with pytest.raises(Invalid):
            dilate([], E1)
        with pytest.raises(Invalid):
            square_element(-1)
        with pytest.raises(Invalid):
            cross_element(-1)
