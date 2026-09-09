from __future__ import annotations

import math
import random

import pytest

from atlas.distancetransform import brute_force, chamfer, exact, relative_errors
from atlas.errors import Invalid

SIZE = 41
SINGLE = [[r == 20 and c == 20 for c in range(SIZE)] for r in range(SIZE)]


def _worst_deviation(a, b) -> float:
    rows = zip(a, b, strict=True)
    return max(abs(x - y) for ra, rb in rows for x, y in zip(ra, rb, strict=True))


class TestTheExactTransform:
    def test_it_matches_brute_force_round_a_single_feature(self):
        assert _worst_deviation(exact(SINGLE), brute_force(SINGLE)) == 0.0

    def test_it_matches_brute_force_on_random_grids(self):
        rng = random.Random(195)
        checked = 0
        for _ in range(12):
            grid = [[rng.random() < 0.02 for _ in range(40)] for _ in range(30)]
            if not any(any(row) for row in grid):
                continue
            assert _worst_deviation(exact(grid), brute_force(grid)) == 0.0
            checked += 1
        assert checked >= 10

    def test_features_read_zero_and_axes_read_the_offset(self):
        e = exact(SINGLE)
        assert e[20][20] == 0.0
        assert e[20][30] == 10.0
        assert e[30][30] == pytest.approx(10 * math.sqrt(2))


class TestTheChamferTransform:
    def test_three_four_is_exact_on_axes_and_low_on_diagonals(self):
        ch = chamfer(SINGLE)
        assert ch[20][30] == 10.0
        assert ch[30][30] == pytest.approx(40 / 3)
        errors = relative_errors(ch, brute_force(SINGLE))
        assert min(errors) == pytest.approx(-0.0572, abs=1e-3)
        assert max(errors) == pytest.approx(0.0541, abs=1e-3)
        assert sum(errors) / len(errors) == pytest.approx(0.0186, abs=1e-3)

    def test_the_worst_overestimate_sits_18_degrees_off_an_axis(self):
        ch, truth = chamfer(SINGLE), brute_force(SINGLE)
        worst = max(
            (ch[r][c] / truth[r][c] - 1, r, c)
            for r in range(SIZE)
            for c in range(SIZE)
            if (r, c) != (20, 20)
        )
        _, r, c = worst
        off_axis = min(math.degrees(math.atan2(abs(r - 20), abs(c - 20))) % 90, 90)
        assert min(off_axis, 90 - off_axis) == pytest.approx(18.4, abs=0.1)

    def test_random_grids_average_one_percent_high(self):
        rng = random.Random(195)
        means, worsts = [], []
        for _ in range(20):
            grid = [[rng.random() < 0.02 for _ in range(40)] for _ in range(30)]
            if not any(any(row) for row in grid):
                continue
            errors = relative_errors(chamfer(grid), brute_force(grid))
            means.append(sum(errors) / len(errors))
            worsts.append(max(errors))
        assert sum(means) / len(means) == pytest.approx(0.0128, abs=1e-3)
        assert max(worsts) == pytest.approx(0.0541, abs=1e-3)

    def test_cruder_weights(self):
        truth = brute_force(SINGLE)
        two_three = max(relative_errors(chamfer(SINGLE, 2, 3), truth))
        assert two_three == pytest.approx(0.118, abs=1e-3)
        chessboard = min(relative_errors(chamfer(SINGLE, 1, 1), truth))
        assert chessboard == pytest.approx(math.sqrt(2) / 2 - 1, abs=1e-3)


class TestRefusals:
    def test_empty_and_featureless_grids_are_refused(self):
        with pytest.raises(Invalid):
            exact([[False, False]])
        with pytest.raises(Invalid):
            exact([])
        with pytest.raises(Invalid):
            chamfer([[False]])
