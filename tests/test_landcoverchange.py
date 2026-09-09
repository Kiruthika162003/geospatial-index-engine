from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.landcoverchange import (
    change_fraction,
    confusion,
    edge_fraction,
    kappa,
    majority_filter,
    patches,
    real_change,
    shifted,
    spurious_change_law,
)


def _scene(patch):
    return patches(128, patch, 4, random.Random(571))


class TestMisregistration:
    @pytest.mark.parametrize(
        ("patch", "fractions", "kappas"),
        [
            (4, (0.188, 0.376, 0.752), (0.749, 0.498, -0.0039)),
            (8, (0.0908, 0.1816, 0.3633), (0.8785, 0.7571, 0.5142)),
            (32, (0.0234, 0.0469, 0.0938), (0.9684, 0.9368, 0.8737)),
        ],
    )
    def test_a_shift_fakes_change_on_three_quarters_of_shift_over_patch(
        self, patch, fractions, kappas
    ):
        scene = _scene(patch)
        for shift, fraction, k in zip((1, 2, 4), fractions, kappas, strict=True):
            moved = shifted(scene, shift, 0)
            assert change_fraction(scene, moved) == pytest.approx(fraction, abs=1e-4)
            assert kappa(confusion(scene, moved)) == pytest.approx(k, abs=1e-4)
            assert abs(fraction / spurious_change_law(patch, shift, 4) - 1) < 0.035
        diagonal = change_fraction(scene, shifted(scene, 1, 1)) / fractions[0]
        assert 1.7 < diagonal < 2.0

    def test_edge_fractions(self):
        assert edge_fraction(_scene(4)) == pytest.approx(0.3399, abs=1e-4)
        assert edge_fraction(_scene(32)) == pytest.approx(0.0463, abs=1e-4)


class TestRealChange:
    @pytest.mark.parametrize(
        ("fraction", "aligned", "misread", "filtered", "k_aligned", "k_misread"),
        [
            (0.02, 0.018, 0.1065, 0.0909, 0.9759, 0.8576),
            (0.05, 0.0482, 0.134, 0.092, 0.9356, 0.8209),
            (0.1, 0.0974, 0.1769, 0.0948, 0.8698, 0.7635),
        ],
    )
    def test_the_filter_keeps_the_seams_and_loses_the_change(
        self, fraction, aligned, misread, filtered, k_aligned, k_misread
    ):
        scene = _scene(8)
        changed = real_change(scene, fraction, 4, random.Random(572))
        moved = shifted(changed, 1, 0)
        assert change_fraction(scene, changed) == pytest.approx(aligned, abs=1e-4)
        assert change_fraction(scene, moved) == pytest.approx(misread, abs=1e-4)
        cleaned = change_fraction(majority_filter(scene), majority_filter(moved))
        assert cleaned == pytest.approx(filtered, abs=1e-4)
        assert kappa(confusion(scene, changed)) == pytest.approx(k_aligned, abs=1e-4)
        assert kappa(confusion(scene, moved)) == pytest.approx(k_misread, abs=1e-4)

    def test_kappa_ends(self):
        scene = _scene(8)
        assert kappa(confusion(scene, scene)) == 1.0
        assert kappa(confusion(scene, patches(128, 8, 4, random.Random(999)))) == pytest.approx(
            -0.0194, abs=1e-4
        )


class TestRefusals:
    def test_bad_shapes_patches_fractions_and_tables(self):
        with pytest.raises(Invalid):
            confusion([], [])
        with pytest.raises(Invalid):
            confusion([[0]], [[0, 1]])
        with pytest.raises(Invalid):
            patches(8, 0, 4, random.Random(1))
        with pytest.raises(Invalid):
            real_change([[0]], 2.0, 4, random.Random(1))
        with pytest.raises(Invalid):
            kappa({})
        with pytest.raises(Invalid):
            spurious_change_law(4, 1, 1)
