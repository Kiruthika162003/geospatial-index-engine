from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.morani import (
    blocks,
    boundary_fraction,
    checkerboard,
    expected_i,
    gradient,
    morans_i,
    rook_pairs,
    shuffle_scores,
    z_score,
)


class TestExtremes:
    @pytest.mark.parametrize("size", [4, 5, 7, 10, 20])
    def test_a_checkerboard_reads_exactly_minus_one(self, size):
        assert morans_i(checkerboard(size)) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        ("size", "reading"), [(4, 0.666667), (10, 0.888889), (20, 0.947368)]
    )
    def test_blocks_fall_short_of_one_by_twice_the_boundary_fraction(self, size, reading):
        assert morans_i(blocks(size)) == pytest.approx(reading, abs=1e-6)
        assert morans_i(blocks(size)) == pytest.approx(1 - 2 * boundary_fraction(size))

    @pytest.mark.parametrize("size", [4, 10, 20])
    def test_a_gradient_reads_identically_to_the_blocks(self, size):
        assert morans_i(gradient(size)) == pytest.approx(morans_i(blocks(size)))

    def test_rook_pairs_and_the_boundary(self):
        assert len(rook_pairs(10, 10)) == 180
        assert boundary_fraction(10) == pytest.approx(10 / 180)


class TestTheBaseline:
    def test_the_expectation_is_minus_one_over_n_minus_one(self):
        assert expected_i(100) == pytest.approx(-1 / 99)
        assert expected_i(16) == pytest.approx(-1 / 15)

    def test_shuffles_land_on_the_expectation_within_their_standard_error(self):
        mean, sd = shuffle_scores(gradient(10), 500, random.Random(200))
        assert mean == pytest.approx(-0.0069, abs=1e-3)
        assert sd == pytest.approx(0.0697, abs=2e-3)
        assert abs(mean - expected_i(100)) < 2 * sd / 500**0.5


class TestZScores:
    def test_patterns_score_far_from_random_and_a_shuffle_scores_within_two(self):
        assert z_score(blocks(10), 300, random.Random(201)) == pytest.approx(11.42, abs=0.05)
        assert z_score(gradient(10), 300, random.Random(201)) == pytest.approx(12.82, abs=0.05)
        checker = z_score(checkerboard(10), 300, random.Random(201))
        assert checker == pytest.approx(-13.44, abs=0.05)
        values = [v for row in gradient(10) for v in row]
        random.Random(202).shuffle(values)
        shuffled = [values[r * 10 : (r + 1) * 10] for r in range(10)]
        assert morans_i(shuffled) == pytest.approx(-0.0552, abs=1e-3)
        assert abs(z_score(shuffled, 300, random.Random(203))) < 2


class TestRefusals:
    def test_constant_empty_and_single_grids_are_refused(self):
        with pytest.raises(Invalid):
            morans_i([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(Invalid):
            morans_i([[1.0]])
        with pytest.raises(Invalid):
            morans_i([])
        with pytest.raises(Invalid):
            expected_i(1)
        with pytest.raises(Invalid):
            shuffle_scores(gradient(4), 1, random.Random(0))
