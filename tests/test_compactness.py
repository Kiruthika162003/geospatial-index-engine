from __future__ import annotations

import math

import pytest

from atlas.compactness import (
    comb,
    convex_hull_ratio,
    perimeter,
    polsby_popper,
    rectangle,
    regular_polygon,
    reock,
    schwartzberg,
    scores,
)
from atlas.errors import Degenerate, Invalid
from atlas.shoelace import signed_area


class TestConvexFigures:
    def test_the_square(self):
        s = scores(rectangle(1, 1))
        assert s["polsby_popper"] == pytest.approx(math.pi / 4)
        assert s["schwartzberg"] == pytest.approx(math.sqrt(math.pi / 4))
        assert s["reock"] == pytest.approx(2 / math.pi)
        assert s["convex_hull_ratio"] == pytest.approx(1.0)

    def test_the_ten_to_one_rectangle(self):
        s = scores(rectangle(10, 1))
        assert s["polsby_popper"] == pytest.approx(0.2596, abs=1e-4)
        assert s["reock"] == pytest.approx(0.1261, abs=1e-4)
        assert s["convex_hull_ratio"] == pytest.approx(1.0)

    def test_schwartzberg_is_the_square_root_of_polsby_popper(self):
        for ring in (rectangle(1, 1), rectangle(10, 1), regular_polygon(3), comb(1.0, 4, 0.8)):
            assert schwartzberg(ring) == pytest.approx(math.sqrt(polsby_popper(ring)))


class TestTheComb:
    def test_four_teeth_at_depth_0_8(self):
        ring = comb(1.0, 4, 0.8)
        assert abs(signed_area(ring)) == pytest.approx(0.6)
        assert perimeter(ring) == pytest.approx(10.4)
        s = scores(ring)
        assert s["polsby_popper"] == pytest.approx(0.0697, abs=1e-4)
        assert s["reock"] == pytest.approx(0.6 * 2 / math.pi)  # exactly the area fraction
        assert s["convex_hull_ratio"] == pytest.approx(0.6)

    def test_eight_teeth_at_depth_0_9(self):
        s = scores(comb(1.0, 8, 0.9))
        assert s["polsby_popper"] == pytest.approx(0.0204, abs=1e-4)
        assert s["reock"] == pytest.approx(0.3501, abs=1e-4)
        assert s["convex_hull_ratio"] == pytest.approx(0.55)


class TestTheCircle:
    def test_regular_polygons_approach_one(self):
        assert polsby_popper(regular_polygon(8)) == pytest.approx(0.9481, abs=1e-4)
        assert polsby_popper(regular_polygon(16)) == pytest.approx(0.9871, abs=1e-4)
        assert polsby_popper(regular_polygon(32)) == pytest.approx(0.9968, abs=1e-4)
        assert reock(regular_polygon(32)) == pytest.approx(0.9936, abs=1e-4)
        firsts = [n for n in (8, 16, 32, 64) if polsby_popper(regular_polygon(n)) > 0.99]
        assert firsts[0] == 32
        firsts = [n for n in (8, 16, 32, 64) if reock(regular_polygon(n)) > 0.99]
        assert firsts[0] == 32
        assert polsby_popper(regular_polygon(256)) > 0.9999

    def test_the_hexagon_and_triangle(self):
        assert polsby_popper(regular_polygon(6)) == pytest.approx(0.9069, abs=1e-4)
        assert polsby_popper(regular_polygon(3)) == pytest.approx(0.6046, abs=1e-4)
        assert convex_hull_ratio(regular_polygon(3)) == pytest.approx(1.0)


class TestRefusals:
    def test_bad_shapes_are_refused(self):
        with pytest.raises(Invalid):
            polsby_popper([(0, 0), (1, 1)])
        with pytest.raises(Degenerate):
            polsby_popper([(0, 0), (1, 1), (2, 2)])
        with pytest.raises(Invalid):
            comb(1, 0, 0.5)
        with pytest.raises(Invalid):
            regular_polygon(2)
