from __future__ import annotations

import math
import random

import pytest

from atlas.aspectnoise import (
    aspect_error_law,
    aspect_field,
    circular_error,
    flat_slope_law,
    gradient_noise,
    horn,
    mean,
    noisy,
    plane,
    slope_field,
    smoothed,
)
from atlas.errors import Invalid

SIZE = 41


class TestClean:
    @pytest.mark.parametrize("aspect", [0.0, 90.0, 200.0, 315.0])
    def test_a_plane_reads_its_aspect_and_slope_exactly(self, aspect):
        grid = plane(SIZE, 0.3, aspect)
        assert mean(aspect_field(grid)) == pytest.approx(aspect, abs=1e-6)
        assert mean(slope_field(grid)) == pytest.approx(math.degrees(math.atan(0.3)), abs=1e-6)
        assert gradient_noise(1.0) == pytest.approx(math.sqrt(12) / 8)


class TestNoise:
    @pytest.mark.parametrize(
        ("slope", "sigma", "ratio", "error", "law", "circular", "smooth"),
        [
            (0.5, 0.1, 11.55, 3.94, 3.95, 4.96, 1.62),
            (0.5, 0.3, 3.85, 12.05, 11.62, 15.36, 4.89),
            (0.5, 1.0, 1.15, 42.81, 32.63, 54.73, 17.4),
            (0.05, 0.1, 1.15, 42.81, 32.63, 54.73, 17.4),
            (2.0, 1.0, 4.62, 9.97, 9.75, 12.67, 4.06),
        ],
    )
    def test_the_error_depends_on_the_ratio_alone(
        self, slope, sigma, ratio, error, law, circular, smooth
    ):
        grid = noisy(plane(SIZE, slope, 135.0), sigma, random.Random(480))
        assert slope / gradient_noise(sigma) == pytest.approx(ratio, abs=0.01)
        read, spread = circular_error(aspect_field(grid), 135.0)
        assert read == pytest.approx(error, abs=0.01)
        assert spread == pytest.approx(circular, abs=0.01)
        assert aspect_error_law(slope, sigma) == pytest.approx(law, abs=0.01)
        assert circular_error(aspect_field(smoothed(grid)), 135.0)[0] == pytest.approx(
            smooth, abs=0.01
        )
        if ratio > 4:
            assert abs(read / law - 1) < 0.03

    def test_noise_only_adds_slope(self):
        grid = noisy(plane(SIZE, 0.1, 135.0), 0.3, random.Random(480))
        assert mean(slope_field(grid)) == pytest.approx(10.329, abs=1e-3)
        for sigma, reading, law, smooth in (
            (0.1, 3.183, 3.106, 1.29),
            (1.0, 27.635, 28.489, 12.497),
        ):
            flat = noisy(plane(SIZE, 0.0, 0.0), sigma, random.Random(481))
            assert mean(slope_field(flat)) == pytest.approx(reading, abs=1e-3)
            assert flat_slope_law(sigma) == pytest.approx(law, abs=1e-3)
            assert mean(slope_field(smoothed(flat))) == pytest.approx(smooth, abs=1e-3)
            assert circular_error(aspect_field(flat), 0.0)[0] == pytest.approx(89.55, abs=0.01)

    @pytest.mark.parametrize(
        ("cell", "error", "law"), [(1.0, 56.66, 41.82), (30.0, 1.98, 1.98)]
    )
    def test_the_cell_size_enters_through_the_gradient(self, cell, error, law):
        grid = noisy(plane(SIZE, 0.1, 135.0, cell), 0.3, random.Random(482))
        assert circular_error(aspect_field(grid, cell), 135.0)[0] == pytest.approx(
            error, abs=0.01
        )
        assert aspect_error_law(0.1, 0.3, cell) == pytest.approx(law, abs=0.01)


class TestRefusals:
    def test_edges_aspects_and_empties(self):
        with pytest.raises(Invalid):
            horn(plane(5, 0.1, 0.0), 0, 2, 1.0)
        with pytest.raises(Invalid):
            plane(5, 0.1, 360.0)
        with pytest.raises(Invalid):
            circular_error([], 0.0)
        assert aspect_error_law(0.0, 1.0) == 90.0
