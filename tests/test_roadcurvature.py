from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.roadcurvature import (
    arc,
    corner,
    curvatures,
    menger,
    noisy_line,
    random_walk,
    tortuosity,
)


class TestCircles:
    @pytest.mark.parametrize("radius", [1.0, 5.0, 100.0])
    def test_curvature_is_one_over_r_and_a_half_circle_is_pi_over_two(self, radius):
        half = arc(radius, 180.0, 37)
        values = curvatures(half)
        assert min(values) == pytest.approx(1 / radius, abs=1e-9)
        assert max(values) == pytest.approx(1 / radius, abs=1e-9)
        assert tortuosity(half) == pytest.approx(1.5703, abs=1e-4)

    def test_a_quarter_circle(self):
        quarter = math.pi / (2 * math.sqrt(2))
        assert tortuosity(arc(1.0, 90.0, 91)) == pytest.approx(quarter, abs=1e-3)


class TestCorners:
    @pytest.mark.parametrize("spacing", [1.0, 0.5, 0.25, 0.1])
    def test_a_60_degree_corner_reads_one_over_the_spacing(self, spacing):
        values = curvatures(corner(60.0, spacing))
        assert max(values) == pytest.approx(1 / spacing)
        assert min(values) == pytest.approx(0.0, abs=1e-12)


class TestWalksAndNoise:
    def test_a_random_walks_tortuosity_grows_roughly_as_root_two_per_doubling(self):
        rng = random.Random(265)
        means = []
        for n in (100, 200, 400, 800, 1600):
            means.append(sum(tortuosity(random_walk(n, rng)) for _ in range(40)) / 40)
        assert means == pytest.approx([21.44, 22.74, 28.46, 42.9, 57.23], abs=0.05)
        assert means[-1] / means[0] > 2

    def test_noise_fakes_curvature_at_about_twice_noise_over_spacing_squared(self):
        rng = random.Random(265)
        for n in (100, 200, 400, 800, 1600):
            for _ in range(40):
                random_walk(n, rng)
        for noise, spacing in ((0.01, 1.0), (0.01, 0.25), (0.1, 0.25)):
            rng = random.Random(int(noise * 1000) + int(spacing * 100))
            values = []
            for _ in range(20):
                values.extend(curvatures(noisy_line(50, spacing, noise, rng)))
            factor = (sum(values) / len(values)) / (noise / spacing**2)
            assert 1.2 < factor < 2.4
            if noise == 0.01:
                assert factor == pytest.approx(2.0, abs=0.25)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            menger((0, 0), (0, 0), (1, 1))
        with pytest.raises(Invalid):
            curvatures([(0, 0), (1, 1)])
        with pytest.raises(Invalid):
            tortuosity([(0, 0)])
        with pytest.raises(Invalid):
            tortuosity([(0, 0), (1, 1), (0, 0)])
        with pytest.raises(Invalid):
            arc(1, 90, 1)
