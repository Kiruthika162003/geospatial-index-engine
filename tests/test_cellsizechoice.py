from __future__ import annotations

import random

import pytest

from atlas.cellsizechoice import (
    cell_for_empty_share,
    clustered,
    coefficient_of_variation,
    counts,
    empty_share,
    poisson_empty_law,
    rule_of_thumb,
    sweep,
    uniform,
)
from atlas.errors import Invalid

FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]


class TestUniform:
    def test_the_rule_leaves_most_cells_empty(self):
        res = sweep(uniform(2000, random.Random(860), 1000.0), 1000.0, FACTORS)
        assert [round(res[f]["empty"], 4) for f in FACTORS] == [
            0.9392,
            0.7828,
            0.3773,
            0.0359,
            0.0139,
        ]
        assert [round(res[f]["law"], 4) for f in FACTORS] == [
            0.9394,
            0.7788,
            0.3679,
            0.0183,
            0.0,
        ]
        assert [round(res[f]["cv"], 3) for f in FACTORS] == [3.988, 2.03, 1.014, 0.542, 0.436]
        assert res[0.5]["cell"] == pytest.approx(11.18, abs=0.01)
        assert res[0.5]["per_cell"] == pytest.approx(0.25)
        assert rule_of_thumb(1000.0, 2000) == pytest.approx(11.18, abs=0.01)
        assert cell_for_empty_share(2000, 1000.0, 0.05) == pytest.approx(38.702, abs=1e-3)
        assert cell_for_empty_share(2000, 1000.0, 0.01) == pytest.approx(47.985, abs=1e-3)
        assert poisson_empty_law(1.0) == pytest.approx(0.3679, abs=1e-4)


class TestClustered:
    def test_the_law_fails_on_clusters(self):
        res = sweep(clustered(2000, random.Random(861), 1000.0), 1000.0, FACTORS)
        assert [round(res[f]["empty"], 4) for f in FACTORS] == [
            0.9537,
            0.8926,
            0.8247,
            0.7543,
            0.6736,
        ]
        assert [round(res[f]["cv"], 3) for f in FACTORS] == [5.597, 3.872, 3.282, 2.972, 2.737]
        assert all(res[f]["empty"] > res[f]["law"] for f in FACTORS)


class TestRefusals:
    def test_bad_counts_cells_and_targets(self):
        with pytest.raises(Invalid):
            rule_of_thumb(1000.0, 0)
        with pytest.raises(Invalid):
            counts([(0.0, 0.0)], 10.0, 0.0)
        with pytest.raises(Invalid):
            coefficient_of_variation([[0, 0]])
        with pytest.raises(Invalid):
            cell_for_empty_share(10, 100.0, 1.0)
        assert empty_share(counts([(5.0, 5.0)], 10.0, 5.0)) == 0.75
