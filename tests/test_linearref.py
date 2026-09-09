from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.linearref import brute_project, cumulative, cut, hairpin, length, locate, project

SQUARE_PATH = [(0, 0), (3, 0), (3, 4), (0, 4)]


class TestIdentities:
    def test_locate_then_project_returns_the_measure_with_zero_offset(self):
        rng = random.Random(246)
        for _ in range(60):
            count = rng.randint(2, 8)
            line = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(count)]
            total = length(line)
            for _ in range(20):
                m = rng.uniform(0, total)
                back, offset, _ = project(line, locate(line, m))
                assert abs(back - m) < 1e-9
                assert abs(offset) < 1e-9
            q = (rng.uniform(0, 100), rng.uniform(0, 100))
            assert abs(project(line, q)[0] - brute_project(line, q, 4000)) / total < 2.6e-4
            a, b = sorted((rng.uniform(0, total), rng.uniform(0, total)))
            if b - a > 1e-6:
                assert length(cut(line, a, b)) == pytest.approx(b - a, abs=1e-9)

    def test_marks_cuts_and_signed_offsets_on_a_square_path(self):
        assert cumulative(SQUARE_PATH) == [0.0, 3.0, 7.0, 10.0]
        assert cut(SQUARE_PATH, 3.0, 7.0) == [(3.0, 0.0), (3.0, 4.0)]
        assert project(SQUARE_PATH, (1.5, 1.0)) == (1.5, 1.0, 0)
        assert project(SQUARE_PATH, (1.5, -1.0)) == (1.5, -1.0, 0)
        assert locate(SQUARE_PATH, 8.5) == (1.5, 4.0)


class TestTheHairpin:
    @pytest.mark.parametrize(("x", "apart"), [(2.0, 17.0), (5.0, 11.0), (9.0, 3.0)])
    def test_two_measures_claim_a_point_between_the_legs(self, x, apart):
        line = hairpin(10.0, 1.0)
        measure, offset, segment = project(line, (x, 0.5))
        assert (measure, segment) == (x, 0)
        assert abs(offset) == pytest.approx(0.5)
        assert abs((21.0 - x) - measure) == pytest.approx(apart)

    def test_outside_the_bend_the_crossbar_wins(self):
        assert project(hairpin(10.0, 1.0), (11.0, 0.5)) == (10.5, -1.0, 1)


class TestRefusals:
    def test_measures_off_the_line_and_backward_cuts_are_refused(self):
        with pytest.raises(Outside):
            locate(SQUARE_PATH, -1)
        with pytest.raises(Outside):
            locate(SQUARE_PATH, 11)
        with pytest.raises(Outside):
            cut(SQUARE_PATH, 5, 2)
        with pytest.raises(Invalid):
            cumulative([(0, 0)])
