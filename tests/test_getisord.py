from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.getisord import (
    false_positive_rate,
    hot_cells,
    labeled_fraction,
    nominal_rate,
    scores,
)
from atlas.morani import blocks


class TestTheBlocks:
    def test_interiors_boundary_and_corners_score_as_measured(self):
        z = scores(blocks(10))
        assert z[5][2] == pytest.approx(2.2827, abs=1e-4)
        assert z[5][7] == pytest.approx(-2.2827, abs=1e-4)
        assert z[5][4] == pytest.approx(1.3696, abs=1e-4)
        assert z[5][5] == pytest.approx(-1.3696, abs=1e-4)
        assert z[0][0] == pytest.approx(1.7498, abs=1e-4)
        hot, cold = labeled_fraction(blocks(10))
        assert (hot, cold) == (0.38, 0.38)


class TestFalsePositives:
    def test_binary_shuffles_run_1_45_times_nominal_and_uniform_values_run_nominal(self):
        rng = random.Random(250)
        rate = false_positive_rate(blocks(10), 300, rng)
        assert rate == pytest.approx(0.0723, abs=2e-3)
        assert nominal_rate() == pytest.approx(0.05, abs=1e-4)
        assert rate / nominal_rate() == pytest.approx(1.445, abs=0.05)
        uniform = [[rng.random() for _ in range(30)] for _ in range(30)]
        assert false_positive_rate(uniform, 100, rng) == pytest.approx(0.0484, abs=5e-3)


class TestResolution:
    def test_a_single_hot_cell_lights_exactly_its_neighbourhood(self):
        flat = [[0.0] * 10 for _ in range(10)]
        flat[5][5] = 10.0
        assert sorted(hot_cells(flat)) == [(4, 5), (5, 4), (5, 5), (5, 6), (6, 5)]
        z = scores(flat)
        assert z[5][5] == pytest.approx(4.359, abs=1e-3)
        assert z[5][6] == pytest.approx(z[5][5])
        assert z[0][0] == pytest.approx(-0.176, abs=1e-3)


class TestRefusals:
    def test_constant_tiny_and_empty_grids_are_refused(self):
        with pytest.raises(Invalid):
            scores([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(Invalid):
            scores([[1.0, 2.0]])
        with pytest.raises(Invalid):
            scores([])
        with pytest.raises(Invalid):
            false_positive_rate(blocks(4), 0, random.Random(0))
