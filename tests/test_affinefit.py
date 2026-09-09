from __future__ import annotations

import random

import pytest

from atlas.affinefit import (
    apply,
    control_points,
    expected_rms,
    fit_affine,
    fit_similarity,
    jitter,
    leave_one_out,
    residuals,
    rms,
    rotation_of,
    scale_of,
    true_transform,
)
from atlas.errors import Invalid

TRUTH = true_transform(2.0, 30.0, (500.0, -200.0), 0.05)


def _draws(n, count=20):
    for k in range(count):
        rng = random.Random(530 + k)
        source = control_points(n, rng)
        yield source, jitter([apply(TRUTH, p) for p in source], 1.0, rng)


class TestResiduals:
    @pytest.mark.parametrize(
        ("n", "fit", "law", "loo", "similarity"),
        [
            (3, 0.0, 0.0, None, 3.974),
            (4, 0.719, 0.707, 16.513, 6.478),
            (10, 1.156, 1.183, 1.679, 9.157),
            (50, 1.391, 1.371, 1.48, 10.121),
        ],
    )
    def test_the_fit_law_and_the_leave_one_out_blowup(self, n, fit, law, loo, similarity):
        fits, loos, sims = [], [], []
        for source, target in _draws(n):
            fits.append(rms(residuals(fit_affine(source, target), source, target)))
            sims.append(rms(residuals(fit_similarity(source, target), source, target)))
            if n > 3:
                loos.append(rms(leave_one_out(source, target)))
        assert sum(fits) / 20 == pytest.approx(fit, abs=1e-3)
        assert expected_rms(1.0, n) == pytest.approx(law, abs=1e-3)
        assert sum(sims) / 20 == pytest.approx(similarity, abs=1e-3)
        if loo is not None:
            assert sum(loos) / 20 == pytest.approx(loo, abs=1e-3)

    def test_recovery_and_exactness(self):
        rng = random.Random(531)
        source = control_points(10, rng)
        target = jitter([apply(TRUTH, p) for p in source], 1.0, rng)
        t = fit_affine(source, target)
        assert scale_of(t) == pytest.approx(1.9862, abs=1e-4)
        assert rotation_of(t) == pytest.approx(29.961, abs=1e-3)
        assert scale_of(TRUTH) == pytest.approx(1.9875, abs=1e-4)
        exact = fit_affine(source[:3], target[:3])
        assert rms(residuals(exact, source[:3], target[:3])) < 1e-9


class TestShear:
    @pytest.mark.parametrize(("shear", "floor"), [(0.0, 0.0), (0.05, 10.192), (0.2, 40.767)])
    def test_a_similarity_cannot_absorb_shear(self, shear, floor):
        truth = true_transform(2.0, 30.0, (500.0, -200.0), shear)
        source = control_points(50, random.Random(550))
        target = [apply(truth, p) for p in source]
        assert rms(residuals(fit_similarity(source, target), source, target)) == pytest.approx(
            floor, abs=1e-3
        )
        assert rms(residuals(fit_affine(source, target), source, target)) < 1e-9


class TestRefusals:
    def test_too_few_collinear_and_mismatched_points(self):
        with pytest.raises(Invalid):
            fit_affine([(0, 0), (1, 0)], [(0, 0), (1, 0)])
        with pytest.raises(Invalid):
            fit_affine([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)])
        with pytest.raises(Invalid):
            fit_affine([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0)])
        with pytest.raises(Invalid):
            fit_similarity([(0, 0)], [(0, 0)])
        with pytest.raises(Invalid):
            leave_one_out([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])
        with pytest.raises(Invalid):
            rms([])
