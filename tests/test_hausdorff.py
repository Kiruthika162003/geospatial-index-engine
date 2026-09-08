from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.hausdorff import directed, hausdorff


class TestAsymmetry:
    def test_a_short_curve_along_a_long_one_is_near_it_but_not_vice_versa(self):
        long_curve = [(x, 0) for x in range(0, 101, 5)]
        short_curve = [(x, 0.5) for x in range(0, 21, 5)]
        # the long curve's far end (100, 0) is nearest to (20, 0.5): hypot(80, 0.5)
        far = (80**2 + 0.5**2) ** 0.5
        assert directed(short_curve, long_curve) == pytest.approx(0.5)
        assert directed(long_curve, short_curve) == pytest.approx(far)
        assert hausdorff(short_curve, long_curve) == pytest.approx(far)

    def test_a_subset_is_at_directed_distance_zero_from_its_superset(self):
        rng = random.Random(99)
        for _ in range(500):
            big = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(30)]
            sub = big[:5]
            assert directed(sub, big) == 0.0
            assert directed(big, sub) > 0.0


class TestOutlierDominance:
    def test_one_displaced_point_sets_the_whole_value(self):
        base = [(x, 0) for x in range(100)]
        near = [(x, 0.1) for x in range(100)]
        assert hausdorff(base, near) == pytest.approx(0.1)
        spoiled = [*near[:-1], (99, 50)]
        assert hausdorff(base, spoiled) == pytest.approx(50.0)


class TestMetricProperties:
    def test_symmetry_identity_and_the_triangle_inequality(self):
        rng = random.Random(99)
        for _ in range(2000):
            a = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(rng.randint(1, 12))]
            b = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(rng.randint(1, 12))]
            c = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(rng.randint(1, 12))]
            assert hausdorff(a, b) == pytest.approx(hausdorff(b, a))
            assert hausdorff(a, a) == 0.0
            assert hausdorff(a, c) <= hausdorff(a, b) + hausdorff(b, c) + 1e-9


class TestRefusals:
    def test_an_empty_set_is_refused(self):
        with pytest.raises(Invalid):
            directed([], [(0, 0)])

    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            hausdorff(None, [(0, 0)])
