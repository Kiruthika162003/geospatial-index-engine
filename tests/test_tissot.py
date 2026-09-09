from __future__ import annotations

import math

import pytest

from atlas.azimuthalequidistant import forward as azimuthal
from atlas.errors import Invalid
from atlas.gnomonic import forward as gnomonic
from atlas.tissot import axes_angle, equirectangular, indicatrix, mercator_sphere, scales


class TestMercator:
    @pytest.mark.parametrize("lat", [0, 30, 60, 80])
    def test_conformal_with_both_scales_the_secant(self, lat):
        ind = indicatrix(mercator_sphere, lat, 10.0)
        secant = 1 / math.cos(math.radians(lat))
        assert abs(ind["h"] - ind["k"]) / ind["h"] < 1e-5
        assert ind["h"] == pytest.approx(secant, rel=1e-5)
        assert ind["area"] == pytest.approx(secant * secant, rel=1e-5)
        assert ind["omega"] < 3e-4
        assert axes_angle(mercator_sphere, lat, 10.0) == pytest.approx(90.0, abs=1e-6)


class TestEquirectangular:
    @pytest.mark.parametrize(
        ("lat", "omega"), [(0, 0.0), (30, 8.234), (60, 38.942), (80, 89.512)]
    )
    def test_angular_distortion_is_twice_the_arcsine_of_the_cosine_contrast(self, lat, omega):
        ind = indicatrix(equirectangular, lat, 10.0)
        cos = math.cos(math.radians(lat))
        assert ind["h"] == pytest.approx(1.0, rel=1e-6)
        assert ind["k"] == pytest.approx(1 / cos, rel=1e-6)
        assert ind["omega"] == pytest.approx(omega, abs=1e-3)
        assert ind["omega"] == pytest.approx(
            2 * math.degrees(math.asin((1 - cos) / (1 + cos))), abs=1e-6
        )


class TestAzimuthalEquidistant:
    @pytest.mark.parametrize(
        ("c", "transverse", "omega"),
        [
            (10, 1.005095, 0.291),
            (45, 1.110721, 6.014),
            (90, 1.570796, 25.657),
            (135, 3.332162, 65.141),
        ],
    )
    def test_radial_scale_one_and_transverse_c_over_sin_c(self, c, transverse, omega):
        ind = indicatrix(lambda lat, lon: azimuthal(lat, lon, 0.0, 0.0), 0.0, float(c))
        assert ind["k"] == pytest.approx(1.0, rel=1e-6)
        assert ind["h"] == pytest.approx(transverse, rel=1e-6)
        assert ind["h"] == pytest.approx(math.radians(c) / math.sin(math.radians(c)), rel=1e-6)
        assert ind["omega"] == pytest.approx(omega, abs=1e-3)
        assert ind["area"] == pytest.approx(transverse, rel=1e-5)


class TestGnomonic:
    @pytest.mark.parametrize(("c", "omega"), [(10, 0.877), (45, 19.759), (60, 38.943)])
    def test_radial_sec_squared_transverse_sec_area_sec_cubed(self, c, omega):
        ind = indicatrix(lambda lat, lon: gnomonic(lat, lon, 0.0, 0.0), 0.0, float(c))
        cos = math.cos(math.radians(c))
        assert ind["k"] == pytest.approx(1 / cos**2, rel=1e-5)
        assert ind["h"] == pytest.approx(1 / cos, rel=1e-5)
        assert ind["area"] == pytest.approx(1 / cos**3, rel=1e-5)
        assert ind["omega"] == pytest.approx(omega, abs=1e-3)


class TestRefusals:
    def test_poles_and_bad_steps_are_refused(self):
        with pytest.raises(Invalid):
            scales(mercator_sphere, 89.9999, 0)
        with pytest.raises(Invalid):
            scales(mercator_sphere, 10, 0, step_deg=0)
