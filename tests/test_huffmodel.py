from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.huffmodel import (
    breakeven_distance,
    entropy_of_choice,
    market_shares,
    nearest_shares,
    probabilities,
    share_drift,
    size_shares,
    trade_area,
    uniform_customers,
)

STORES = [(20.0, 20.0), (80.0, 80.0), (50.0, 50.0), (85.0, 15.0)]
SIZES = [1.0, 4.0, 1.0, 1.0]


@pytest.fixture(scope="module")
def customers():
    return uniform_customers(3000, random.Random(620))


class TestShares:
    @pytest.mark.parametrize(
        ("beta", "big", "centre", "area", "entropy"),
        [
            (0.0, 0.5714, 0.1429, 3000, 1.1537),
            (1.0, 0.5123, 0.1769, 1390, 1.0634),
            (2.0, 0.4284, 0.2085, 1160, 0.8128),
            (5.0, 0.3219, 0.266, 934, 0.3918),
        ],
    )
    def test_the_exponent_moves_the_big_stores_share(
        self, customers, beta, big, centre, area, entropy
    ):
        read = market_shares(customers, STORES, SIZES, beta)
        assert read[1] == pytest.approx(big, abs=1e-4)
        assert read[2] == pytest.approx(centre, abs=1e-4)
        assert sum(read) == pytest.approx(1.0)
        assert trade_area(customers, STORES, SIZES, beta, 1, 0.5) == area
        mean = sum(entropy_of_choice(p, STORES, SIZES, beta) for p in customers[:500]) / 500
        assert mean == pytest.approx(entropy, abs=1e-4)

    def test_the_ends(self, customers):
        assert market_shares(customers, STORES, SIZES, 0.0) == pytest.approx(
            size_shares(SIZES), abs=1e-9
        )
        near = nearest_shares(customers, STORES)
        assert near == pytest.approx([0.2343, 0.2273, 0.3573, 0.181], abs=1e-4)
        assert market_shares(customers, STORES, SIZES, 5.0)[2] < near[2]
        assert share_drift(customers, STORES, SIZES, [1.0, 1.5, 2.0, 3.0]) == pytest.approx(
            0.138, abs=1e-3
        )
        assert math.log(4) == pytest.approx(1.3863, abs=1e-4)


class TestReilly:
    @pytest.mark.parametrize(("beta", "point"), [(1.0, 80.0), (2.0, 66.667), (5.0, 56.887)])
    def test_the_breakeven_closes_on_the_midpoint(self, beta, point):
        assert breakeven_distance(4.0, 1.0, 100.0, beta) == pytest.approx(point, abs=1e-3)
        assert breakeven_distance(1.0, 1.0, 100.0, beta) == pytest.approx(50.0)

    def test_refusals(self):
        with pytest.raises(Invalid):
            probabilities((0, 0), STORES, SIZES[:3], 1.0)
        with pytest.raises(Invalid):
            probabilities((0, 0), STORES, SIZES, -1.0)
        with pytest.raises(Invalid):
            probabilities((0, 0), STORES, [1.0, 0.0, 1.0, 1.0], 1.0)
        with pytest.raises(Invalid):
            market_shares([], STORES, SIZES, 1.0)
        with pytest.raises(Invalid):
            breakeven_distance(1.0, 1.0, 10.0, 0.0)
