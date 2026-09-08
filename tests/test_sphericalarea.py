from __future__ import annotations

import math

import pytest

from atlas.errors import Degenerate, Invalid, Outside
from atlas.shoelace import area as flat_area
from atlas.sphericalarea import (
    EARTH_RADIUS_KM,
    band_area_km2,
    band_signed_area_km2,
    sphere_surface_km2,
    triangle_area_km2,
)

OCTANT = [(0, 0), (0, 90), (90, 0)]


class TestTheOctantTellsTheFormulasApart:
    def test_bands_halve_the_octant(self):
        # the refuted guess said 1.0; the band shoelace returns exactly 1/16 of the sphere
        assert band_area_km2(OCTANT) / (sphere_surface_km2() / 8) == pytest.approx(0.5)

    def test_spherical_excess_gets_the_octant_exactly(self):
        assert triangle_area_km2(*OCTANT) / (sphere_surface_km2() / 8) == pytest.approx(1.0)


class TestWhereBandsAreExact:
    def test_a_graticule_rectangle_is_exact(self):
        rect = [(0, 0), (0, 90), (45, 90), (45, 0)]
        exact = EARTH_RADIUS_KM**2 * (math.pi / 2) * math.sin(math.radians(45))
        assert band_area_km2(rect) == pytest.approx(exact, rel=1e-9)

    def test_a_small_patch_agrees_with_the_flat_shoelace(self):
        km = math.pi * EARTH_RADIUS_KM / 180
        patch = [(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)]
        flat = flat_area([(lon * km, lat * km) for lat, lon in patch])
        assert band_area_km2(patch) == pytest.approx(flat, rel=1e-5)

    def test_a_small_triangle_agrees_between_the_two_formulas(self):
        tri = [(0, 0), (0, 0.1), (0.1, 0)]
        assert band_area_km2(tri) == pytest.approx(triangle_area_km2(*tri), rel=1e-5)


class TestSignAndDateline:
    def test_reversing_flips_the_sign(self):
        assert band_signed_area_km2(OCTANT) == pytest.approx(
            -band_signed_area_km2(OCTANT[::-1])
        )

    def test_a_dateline_straddling_patch_is_not_the_whole_globe(self):
        straddle = [(0, 179.95), (0, -179.95), (0.1, -179.95), (0.1, 179.95)]
        patch = [(0, 0), (0, 0.1), (0.1, 0.1), (0.1, 0)]
        assert band_area_km2(straddle) == pytest.approx(band_area_km2(patch), rel=1e-9)


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            band_area_km2([(0, 0), (1, 1)])

    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            band_area_km2([(91, 0), (0, 0), (0, 1)])

    def test_a_degenerate_triangle_is_refused(self):
        with pytest.raises(Degenerate):
            triangle_area_km2((0, 0), (0, 0), (0, 0))
