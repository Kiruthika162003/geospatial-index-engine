from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.rastershift import (
    correlation_surface,
    error,
    find_shift,
    integer_peak,
    noisy,
    parabolic_peak,
    quadratic_peak,
    sample,
    shifted,
    smooth_field,
)


@pytest.fixture(scope="module")
def fields():
    return {
        "smooth": smooth_field(48, random.Random(500), 6, 0.3),
        "textured": smooth_field(48, random.Random(500), 6, 0.8),
    }


class TestRecovery:
    @pytest.mark.parametrize(
        ("truth", "quadratic", "parabolic"),
        [((0.25, 0.0), 0.021, 0.11), ((0.5, 0.5), 0.021, 0.295), ((0.1, -0.4), 0.017, 0.152)],
    )
    def test_the_quadratic_beats_the_paired_parabolas_on_a_smooth_field(
        self, fields, truth, quadratic, parabolic
    ):
        moved = shifted(fields["smooth"], *truth)
        fine = find_shift(fields["smooth"], moved, 4)
        assert error(fine[:2], truth) == pytest.approx(quadratic, abs=1e-3)
        crude = find_shift(fields["smooth"], moved, 4, "parabolic")
        assert error(crude[:2], truth) == pytest.approx(parabolic, abs=1e-3)

    @pytest.mark.parametrize(
        ("truth", "quadratic", "parabolic"),
        [((-1.3, 2.7), 0.025, 0.084), ((0.25, 0.0), 0.009, 0.066)],
    )
    def test_texture_helps_both(self, fields, truth, quadratic, parabolic):
        moved = shifted(fields["textured"], *truth)
        assert error(find_shift(fields["textured"], moved, 4)[:2], truth) == pytest.approx(
            quadratic, abs=1e-3
        )
        assert error(
            find_shift(fields["textured"], moved, 4, "parabolic")[:2], truth
        ) == pytest.approx(parabolic, abs=1e-3)

    def test_no_pull_toward_whole_cells(self, fields):
        for k in range(11):
            f = k / 10
            fine = find_shift(fields["textured"], shifted(fields["textured"], 0.0, f), 3)
            assert abs(fine[1] - f) < 0.02


class TestNoise:
    @pytest.mark.parametrize(
        ("name", "sigma", "mean", "worst"),
        [
            ("smooth", 0.2, 0.0824, 0.143),
            ("smooth", 1.0, 0.8554, 1.2413),
            ("textured", 1.0, 0.0898, 0.1867),
        ],
    )
    def test_noise_hurts_the_smooth_field_ten_times_more(
        self, fields, name, sigma, mean, worst
    ):
        errors = []
        for k in range(5):
            moved = noisy(shifted(fields[name], 0.3, -1.6), sigma, random.Random(510 + k))
            errors.append(error(find_shift(fields[name], moved, 4)[:2], (0.3, -1.6)))
        assert sum(errors) / 5 == pytest.approx(mean, abs=1e-3)
        assert max(errors) == pytest.approx(worst, abs=1e-3)


class TestPieces:
    def test_sampling_peaks_and_refusals(self, fields):
        grid = [[0.0, 1.0], [2.0, 3.0]]
        assert sample(grid, 0.5, 0.5) == 1.5
        assert sample(grid, -3.0, 9.0) == 1.0
        surface = correlation_surface(fields["smooth"], fields["smooth"], 2)
        assert integer_peak(surface) == (0, 0)
        assert surface[(0, 0)] == pytest.approx(1.0)
        assert quadratic_peak(surface) == pytest.approx((0.0, 0.0), abs=0.02)
        assert parabolic_peak(surface) == pytest.approx((0.0, 0.0), abs=0.02)
        with pytest.raises(Invalid):
            correlation_surface(fields["smooth"], grid, 2)
        with pytest.raises(Invalid):
            correlation_surface(grid, grid, 1)
        with pytest.raises(Invalid):
            find_shift(fields["smooth"], fields["smooth"], 2, "spline")
