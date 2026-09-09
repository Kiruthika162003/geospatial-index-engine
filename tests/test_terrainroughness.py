from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.terrainroughness import (
    field,
    hilltop,
    mean_of,
    noisy_plane,
    plane,
    sine_hills,
    tpi,
    tpi_classes,
    tpi_summit_law,
    tri,
    tri_noise_law,
    tri_plane_law,
    window_std,
)


class TestPlanesAndNoise:
    @pytest.mark.parametrize("slope", [0.1, 0.5, 1.0])
    def test_a_plane_reads_root_six_times_the_slope(self, slope):
        grid = plane(41, slope)
        assert mean_of(field(grid, tri)) == pytest.approx(tri_plane_law(slope), abs=1e-9)
        assert tri_plane_law(slope) == pytest.approx(math.sqrt(6) * slope, abs=1e-12)
        assert mean_of(field(grid, tpi)) == pytest.approx(0.0, abs=1e-12)
        deviation = mean_of(field(grid, window_std))
        assert deviation == pytest.approx(slope * math.sqrt(2 / 3), abs=1e-9)

    @pytest.mark.parametrize(
        ("sigma", "reading"), [(0.1, 0.3751), (1.0, 3.7511), (3.0, 11.2534)]
    )
    def test_white_noise_reads_0_938_of_the_law(self, sigma, reading):
        grid = noisy_plane(41, 0.0, sigma, random.Random(410))
        read = mean_of(field(grid, tri))
        assert read == pytest.approx(reading, abs=1e-4)
        assert read / tri_noise_law(sigma) == pytest.approx(0.9378, abs=1e-4)
        assert mean_of(field(grid, window_std)) / sigma == pytest.approx(0.915, abs=1e-3)
        absolute = mean_of([[abs(v) for v in row] for row in field(grid, tpi)])
        assert absolute / sigma == pytest.approx(0.844, abs=1e-3)


class TestHills:
    @pytest.mark.parametrize(
        ("wavelength", "mean_tri", "peak", "law", "ring1", "ring5"),
        [
            (10.0, 5.1411, 6.1007, 7.6953, 0.5211, 1.8414),
            (40.0, 1.3073, 1.9002, 1.9238, 0.0381, 0.5115),
        ],
    )
    def test_sine_hills(self, wavelength, mean_tri, peak, law, ring1, ring5):
        grid = sine_hills(81, wavelength, 5.0)
        values = field(grid, tri)
        assert mean_of(values) == pytest.approx(mean_tri, abs=1e-4)
        assert max(v for row in values for v in row) == pytest.approx(peak, abs=1e-4)
        assert 2 * math.pi * 5.0 / wavelength * math.sqrt(6) == pytest.approx(law, abs=1e-4)
        near = mean_of([[abs(v) for v in row] for row in field(grid, tpi)])
        far = mean_of([[abs(v) for v in row] for row in field(grid, tpi, 5, radius=5)])
        assert (near, far) == pytest.approx((ring1, ring5), abs=1e-4)

    @pytest.mark.parametrize(
        ("radius", "summit", "classes"), [(1, 0.2, (0, 96, 1425)), (3, 1.0889, (509, 320, 396))]
    )
    def test_the_hilltop_summit_law(self, radius, summit, classes):
        grid = hilltop(41, 15.0, 30.0)
        values = field(grid, tpi, radius, radius=radius)
        assert values[20 - radius][20 - radius] == pytest.approx(summit, abs=1e-4)
        assert tpi_summit_law(radius, 15.0, 30.0) == pytest.approx(summit, abs=1e-4)
        assert tpi_classes(values, 0.5) == classes


class TestRefusals:
    def test_bad_grids_cells_and_radii(self):
        with pytest.raises(Invalid):
            tri([], 0, 0)
        with pytest.raises(Invalid):
            tri(plane(5, 1.0), 9, 0)
        with pytest.raises(Invalid):
            tpi(plane(5, 1.0), 2, 2, 0)
        with pytest.raises(Invalid):
            window_std(plane(5, 1.0), -1, 0)
        with pytest.raises(Invalid):
            field(plane(5, 1.0), tri, 3)
