from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.kdrangecost import (
    brute,
    build,
    fitted_constant,
    mean_cost,
    overhead_law,
    range_query,
    square_queries,
    strip_queries,
    uniform,
)


@pytest.fixture(scope="module")
def trees():
    return {n: build(uniform(n, random.Random(670))) for n in (1000, 4000, 16000)}


class TestTheLaw:
    @pytest.mark.parametrize(
        ("n", "overhead", "constant"),
        [(1000, 61.6, 1.947), (4000, 126.3, 1.998), (16000, 252.6, 1.997)],
    )
    def test_a_strip_pays_two_root_n(self, trees, n, overhead, constant):
        _visits, _answers, extra = mean_cost(
            trees[n], strip_queries(200, 0.01, random.Random(672))
        )
        assert extra == pytest.approx(overhead, abs=0.05)
        assert fitted_constant(n, extra) == pytest.approx(constant, abs=1e-3)
        assert overhead_law(n) == pytest.approx(2 * math.sqrt(n))

    @pytest.mark.parametrize(
        ("n", "readings"),
        [
            (1000, ((0.01, 11.3, 0.357), (0.05, 17.1, 0.542), (0.2, 37.9, 1.199))),
            (16000, ((0.01, 19.9, 0.157), (0.05, 44.5, 0.352), (0.2, 137.7, 1.089))),
        ],
    )
    def test_a_square_pays_far_less(self, trees, n, readings):
        for side, overhead, constant in readings:
            _visits, _answers, extra = mean_cost(
                trees[n], square_queries(200, side, random.Random(671))
            )
            assert extra == pytest.approx(overhead, abs=0.05)
            assert fitted_constant(n, extra) == pytest.approx(constant, abs=1e-3)
            assert constant < 2.0


class TestCorrectness:
    def test_matches_brute_force(self):
        points = uniform(4000, random.Random(670))
        root = build(points)
        for box in square_queries(300, 0.05, random.Random(673)):
            assert sorted(range_query(root, box)[0]) == sorted(brute(points, box))

    def test_refusals(self):
        with pytest.raises(Invalid):
            range_query(build([(0.5, 0.5)]), (1.0, 0.0, 0.0, 1.0))
        with pytest.raises(Invalid):
            mean_cost(build([(0.5, 0.5)]), [])
        assert range_query(None, (0.0, 0.0, 1.0, 1.0)) == ([], 0)
