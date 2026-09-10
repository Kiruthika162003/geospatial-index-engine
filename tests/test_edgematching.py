from __future__ import annotations

import random

import pytest

from atlas.edgematching import gap_law, match, miss_law, score, sheet_edges, stranger_law, sweep
from atlas.errors import Invalid

TOLERANCES = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


class TestSparseEdge:
    def test_a_floor_of_misses_never_clears(self):
        res = sweep(40, 10, 0.5, TOLERANCES, 770)
        assert [round(res[t]["true"], 1) for t in TOLERANCES] == [
            11.5,
            19.4,
            31.9,
            37.7,
            38.0,
            38.0,
            38.0,
        ]
        assert [round(res[t]["false"], 2) for t in TOLERANCES] == [
            0.6,
            1.0,
            1.7,
            2.15,
            2.75,
            3.2,
            4.55,
        ]
        assert res[20.0]["missed"] == pytest.approx(2.05, abs=1e-6)
        assert gap_law(0.5) == pytest.approx(0.7071, abs=1e-4)
        for t in (0.25, 0.5):
            assert res[t]["missed"] == pytest.approx(40 * miss_law(0.5, t), abs=2.0)

    def test_more_jitter_raises_the_floor(self):
        res = sweep(40, 10, 2.0, TOLERANCES, 770)
        assert res[20.0]["true"] == pytest.approx(30.8, abs=1e-6)
        assert res[20.0]["missed"] == pytest.approx(9.2, abs=1e-6)
        assert res[20.0]["false"] == pytest.approx(11.8, abs=1e-6)


class TestDenseEdge:
    @pytest.mark.parametrize(
        ("jitter", "best", "lost", "false_at_20"),
        [(0.5, 143.6, 56.4, 87.95), (2.0, 75.2, 124.8, 156.3)],
    )
    def test_density_steals_pairs(self, jitter, best, lost, false_at_20):
        res = sweep(200, 50, jitter, [2.0, 20.0], 770)
        assert res[20.0]["true"] == pytest.approx(best, abs=1e-6)
        assert res[20.0]["missed"] == pytest.approx(lost, abs=1e-6)
        assert res[20.0]["false"] == pytest.approx(false_at_20, abs=1e-6)
        assert stranger_law(0.25, 2.0) == pytest.approx(0.6321, abs=1e-4)
        assert stranger_law(0.05, 2.0) == pytest.approx(0.1813, abs=1e-4)


class TestPieces:
    def test_a_clean_edge_pairs_exactly_and_refusals(self):
        left, right = sheet_edges(5, 0, 0.1, random.Random(1))
        pairs = match(left, right, 1.0)
        assert score(pairs, left, right) == {"true": 5, "false": 0, "missed": 0}
        assert miss_law(0.0, 1.0) == 0.0
        with pytest.raises(Invalid):
            match(left, right, -1.0)
        with pytest.raises(Invalid):
            sheet_edges(-1, 0, 0.1, random.Random(1))
