from __future__ import annotations

import math

import pytest

from atlas.equalarea import (
    PROJECTIONS,
    area_factor,
    axis_ratio,
    cell_polygon,
    cell_true_area,
    cylindrical_omega,
    hemisphere_check,
    lambert_cylindrical,
    mercator_inflation,
    mollweide_theta,
    omega,
    projected_area,
    sinusoidal,
    worst_omega,
)
from atlas.errors import Invalid, Outside
from atlas.tissot import mercator_sphere

LATS = [0, 30, 45, 60, 75, 85]


class TestArea:
    @pytest.mark.parametrize(
        ("name", "bound"),
        [("cylindrical", 1e-5), ("sinusoidal", 1e-9), ("azimuthal", 1e-6), ("mollweide", 1e-5)],
    )
    def test_the_indicatrix_reads_one_to_the_steps_residue(self, name, bound):
        forward = PROJECTIONS[name]
        worst = max(abs(area_factor(forward, lat, lon) - 1) for lat in LATS for lon in (0, 90))
        assert worst < bound

    def test_the_cylinder_is_exact_without_densification(self):
        for south in (0, 40, 70):
            polygon = cell_polygon(south, south + 10, 20, 30)
            truth = cell_true_area(south, south + 10, 20, 30)
            assert projected_area(lambert_cylindrical, polygon) / truth == pytest.approx(
                1.0, abs=1e-14
            )
        assert hemisphere_check(lambert_cylindrical, 1) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("name", "south", "first"),
        [
            ("sinusoidal", 0, -2.54e-3),
            ("sinusoidal", 70, -2.54e-3),
            ("azimuthal", 40, -5.07e-3),
            ("mollweide", 0, -1.58e-3),
            ("mollweide", 70, -1.87e-2),
        ],
    )
    def test_straight_edges_fall_short_and_densification_recovers_sixteenfold(
        self, name, south, first
    ):
        forward = PROJECTIONS[name]
        polygon = cell_polygon(south, south + 10, 20, 30)
        truth = cell_true_area(south, south + 10, 20, 30)
        errors = [projected_area(forward, polygon, n) / truth - 1 for n in (1, 4, 16)]
        assert errors[0] == pytest.approx(first, rel=0.01)
        assert errors[0] / errors[1] == pytest.approx(16, rel=0.02)
        assert errors[1] / errors[2] == pytest.approx(16, rel=0.02)

    def test_the_hemisphere_ring(self):
        assert hemisphere_check(sinusoidal, 1) == pytest.approx(math.pi / 4, abs=1e-5)
        azimuthal, mollweide = PROJECTIONS["azimuthal"], PROJECTIONS["mollweide"]
        assert hemisphere_check(azimuthal, 6) == pytest.approx(3 / math.pi, abs=1e-5)
        assert hemisphere_check(mollweide, 1) == pytest.approx(2 / math.pi, abs=1e-3)
        assert hemisphere_check(sinusoidal, 36) == pytest.approx(0.99984, abs=1e-5)


class TestAngle:
    def test_the_cylinders_omega_is_its_law_and_its_ratio_is_mercators_inflation(self):
        readings = [omega(lambert_cylindrical, lat, 0) for lat in LATS]
        assert readings == pytest.approx([0, 16.426, 38.942, 73.74, 121.957, 160.076], abs=1e-3)
        assert readings == pytest.approx([cylindrical_omega(lat) for lat in LATS], abs=1e-3)
        ratios = [axis_ratio(lambert_cylindrical, lat, 0) for lat in LATS[:-1]]
        assert ratios == pytest.approx([mercator_inflation(lat) for lat in LATS[:-1]], abs=1e-3)

    def test_the_sinusoidal_is_clean_on_its_meridian_and_shears_off_it(self):
        assert all(omega(sinusoidal, lat, 0) < 1e-6 for lat in LATS)
        readings = [omega(sinusoidal, lat, 90) for lat in LATS]
        assert readings == pytest.approx([0, 42.88, 58.092, 68.445, 74.371, 76.08], abs=1e-3)

    def test_the_azimuthal_and_mollweide(self):
        azimuthal = PROJECTIONS["azimuthal"]
        readings = [omega(azimuthal, lat, 0) for lat in LATS]
        assert readings == pytest.approx([38.942, 16.426, 9.063, 3.972, 0.985, 0.109], abs=1e-3)
        assert omega(azimuthal, 45, 90) == pytest.approx(omega(azimuthal, 45, 0), abs=1e-9)
        mollweide = PROJECTIONS["mollweide"]
        central = [omega(mollweide, lat, 0) for lat in LATS]
        assert central == pytest.approx([12.011, 5.756, 2.954, 17.455, 43.238, 79.68], abs=1e-3)
        assert worst_omega(mollweide, LATS, [0, 45, 90])[0] == pytest.approx(108.383, abs=1e-3)
        assert omega(mercator_sphere, 60, 0) < 2e-4

    def test_newton_steps_climb_toward_the_pole(self):
        steps = [mollweide_theta(lat)[1] for lat in (0, 30, 60, 80, 89, 89.9, 90)]
        assert steps == [1, 5, 6, 7, 11, 14, 0]
        assert mollweide_theta(90)[0] == pytest.approx(math.pi / 2)


class TestRefusals:
    def test_bad_coordinates_and_polygons(self):
        with pytest.raises(Outside):
            lambert_cylindrical(91, 0)
        with pytest.raises(Outside):
            sinusoidal(0, 181)
        with pytest.raises(Invalid):
            projected_area(sinusoidal, [(0, 0), (1, 1)])
        with pytest.raises(Invalid):
            projected_area(sinusoidal, cell_polygon(0, 1, 0, 1), 0)
