from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.s2cell import (
    area_spread,
    cell_area_km2,
    cell_center,
    edge_km,
    encode,
    inverse_quadratic,
    quadratic,
)


class TestContainment:
    def test_every_cell_center_re_encodes_to_its_own_cell_on_every_face(self):
        rng = random.Random(252)
        faces = {}
        for _ in range(5000):
            lat, lon = math.degrees(math.asin(rng.uniform(-1, 1))), rng.uniform(-180, 180)
            level = rng.randint(1, 20)
            face, lv, i, j = encode(lat, lon, level)
            faces[face] = faces.get(face, 0) + 1
            assert encode(*cell_center(face, level, i, j), level) == (face, lv, i, j)
        assert set(faces) == set(range(6))
        assert min(faces.values()) > 700

    def test_the_poles_and_the_dateline_sit_inside_cells(self):
        assert encode(90, 0, 4) == (2, 4, 8, 8)
        assert encode(-90, 0, 4) == (5, 4, 8, 8)
        face, _, i, j = encode(0, 180, 4)
        assert face == 3
        assert cell_center(face, 4, i, j) == pytest.approx((-2.458, 177.54), abs=1e-2)
        assert encode(0, 0, 4) == (0, 4, 8, 8)


class TestAreas:
    @pytest.mark.parametrize(
        ("level", "plain", "transformed"), [(3, 3.781, 1.56), (5, 4.86, 1.939)]
    )
    def test_the_transform_halves_the_area_spread(self, level, plain, transformed):
        assert area_spread(level, False) == pytest.approx(plain, abs=1e-3)
        assert area_spread(level, True) == pytest.approx(transformed, abs=1e-3)

    def test_the_six_faces_sum_to_the_sphere(self):
        cells = [(f, i, j) for f in range(6) for i in range(8) for j in range(8)]
        total = sum(cell_area_km2(f, 3, i, j) for f, i, j in cells)
        assert total / (4 * math.pi * 6371.0088**2) == pytest.approx(1.0, abs=1e-9)

    def test_the_edge_schedule(self):
        assert edge_km(0) == 10007.5
        assert edge_km(12) == pytest.approx(2.443, abs=1e-3)
        assert edge_km(30) * 1e6 == pytest.approx(9.32, abs=0.01)


class TestTheTransform:
    def test_it_fixes_the_center_and_the_corners_and_inverts(self):
        assert (quadratic(-1), quadratic(0), quadratic(1)) == (-1.0, 0.0, 1.0)
        for k in range(-100, 101):
            u = k / 100
            assert inverse_quadratic(quadratic(u)) == pytest.approx(u, abs=1e-12)


class TestRefusals:
    def test_bad_levels_and_places_are_refused(self):
        with pytest.raises(Invalid):
            encode(0, 0, 31)
        with pytest.raises(Outside):
            encode(91, 0, 3)
