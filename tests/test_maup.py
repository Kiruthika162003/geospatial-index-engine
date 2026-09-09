from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.maup import (
    aggregate,
    cell_correlation,
    cell_slope,
    correlation,
    slope,
    synthetic,
    weighted_mean_identity,
)


def _records():
    return synthetic(5000, 1.5, random.Random(268))


class TestTheScaleFace:
    def test_the_correlation_climbs_from_0_42_toward_one_with_the_cell(self):
        records = _records()
        first = [r[2] for r in records]
        second = [r[3] for r in records]
        assert correlation(first, second) == pytest.approx(0.4219, abs=1e-3)
        expected = {2.0: 0.5591, 5.0: 0.8886, 10.0: 0.9697, 20.0: 0.9898, 50.0: 0.9972}
        for cell, value in expected.items():
            assert cell_correlation(records, cell) == pytest.approx(value, abs=1e-3)
        assert len(aggregate(records, 10.0)) == 100

    def test_the_noisy_predictors_slope_is_attenuated_then_restored(self):
        records = _records()
        point_slope = slope([r[2] for r in records], [r[3] for r in records])
        assert point_slope == pytest.approx(0.5862, abs=1e-3)
        assert cell_slope(records, 5.0) == pytest.approx(1.6816, abs=1e-3)
        assert cell_slope(records, 10.0) == pytest.approx(1.9155, abs=1e-3)
        assert cell_slope(records, 20.0) == pytest.approx(1.9804, abs=1e-3)

    def test_the_accounting_holds(self):
        records = _records()
        for cell in (2.0, 10.0, 50.0):
            assert max(weighted_mean_identity(records, cell)) < 1e-12


class TestTheZoningFace:
    def test_shifting_the_grid_moves_the_correlation(self):
        records = _records()
        for cell, low, high in ((10.0, 0.8538, 0.9697), (20.0, 0.9505, 0.9898)):
            readings = [cell_correlation(records, cell, cell * k / 10) for k in range(10)]
            assert (min(readings), max(readings)) == pytest.approx((low, high), abs=1e-3)


class TestRefusals:
    def test_bad_sequences_and_cells_are_refused(self):
        with pytest.raises(Invalid):
            correlation([1, 2], [1, 2])
        with pytest.raises(Invalid):
            correlation([1, 1, 1], [1, 2, 3])
        with pytest.raises(Invalid):
            aggregate(_records(), 0)
        with pytest.raises(Invalid):
            slope([1, 1], [1, 2])
