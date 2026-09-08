from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.frechet import frechet
from atlas.hausdorff import hausdorff

PATH = [(x, 0) for x in range(0, 101, 10)]


class TestOrderMatters:
    def test_a_reversed_path_has_zero_hausdorff_but_full_length_frechet(self):
        assert hausdorff(PATH, PATH[::-1]) == 0.0
        assert frechet(PATH, PATH[::-1]) == pytest.approx(100.0)

    def test_a_doubled_back_path_shares_its_point_set_but_not_its_frechet(self):
        straight = [(x, 0) for x in range(11)]
        doubled = (
            [(x, 0) for x in range(11)]
            + [(x, 0) for x in range(9, -1, -1)]
            + [(x, 0) for x in range(1, 11)]
        )
        assert hausdorff(straight, doubled) == 0.0
        assert frechet(straight, doubled) == pytest.approx(5.0)


class TestAgreementWhenAligned:
    def test_a_parallel_shift_gives_the_same_value_as_hausdorff(self):
        shifted = [(x, 0.3) for x in range(0, 101, 10)]
        assert frechet(PATH, shifted) == pytest.approx(0.3)
        assert hausdorff(PATH, shifted) == pytest.approx(0.3)

    def test_frechet_is_never_below_hausdorff_and_close_on_jittered_copies(self):
        rng = random.Random(101)
        gaps = []
        for _ in range(2000):
            n = rng.randint(2, 15)
            p = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)]
            q = [(x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5)) for x, y in p]
            f, h = frechet(p, q), hausdorff(p, q)
            assert f >= h - 1e-9
            gaps.append(f - h)
        assert sum(gaps) / len(gaps) < 0.05  # measured 0.0015


class TestBasics:
    def test_symmetric_and_zero_on_itself(self):
        shifted = [(x, 0.3) for x in range(0, 101, 10)]
        assert frechet(PATH, shifted) == pytest.approx(frechet(shifted, PATH))
        assert frechet(PATH, PATH) == 0.0

    def test_single_points(self):
        assert frechet([(0, 0)], [(3, 4)]) == pytest.approx(5.0)


class TestRefusals:
    def test_an_empty_path_is_refused(self):
        with pytest.raises(Invalid):
            frechet([], [(0, 0)])
