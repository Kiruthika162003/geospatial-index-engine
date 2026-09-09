from __future__ import annotations

import math
import random

import pytest

from atlas.distancematrix import (
    asymmetry,
    diagonal,
    equirectangular,
    great_circle,
    is_metric,
    manhattan,
    matrix,
    nearest_neighbours,
    summary,
    uniform_sphere,
    worst_triangle,
)
from atlas.errors import Invalid


def _global():
    return uniform_sphere(120, random.Random(233))


class TestGreatCircle:
    def test_the_table_is_a_metric_to_rounding(self):
        table = matrix(_global(), great_circle)
        assert asymmetry(table) == 0.0
        assert diagonal(table) == 0.0
        assert worst_triangle(table) < 1e-9
        assert is_metric(table)
        mean, longest = summary(table)
        assert mean == pytest.approx(10054.8, abs=0.5)
        assert abs(mean - math.pi * 6371.0088 / 2) / mean < 0.01
        assert longest == pytest.approx(19961.2, abs=0.5)


class TestEquirectangular:
    def test_the_global_table_is_not_a_metric_and_holds_impossible_entries(self):
        table = matrix(_global(), equirectangular)
        assert asymmetry(table) == 0.0
        assert not is_metric(table)
        assert worst_triangle(table) == pytest.approx(0.6986, abs=1e-3)
        assert summary(table)[1] == pytest.approx(41008.4, abs=0.5)
        assert summary(table)[1] > 2 * math.pi * 6371.0088 / 2

    def test_nearest_neighbours_mostly_survive(self):
        places = _global()
        agree = sum(
            a == b
            for a, b in zip(
                nearest_neighbours(matrix(places, great_circle)),
                nearest_neighbours(matrix(places, equirectangular)),
                strict=True,
            )
        )
        assert agree == 116

    def test_a_local_table_is_safe(self):
        rng = random.Random(233)
        uniform_sphere(120, rng)
        local = []
        for _ in range(120):
            local.append((51.5 + rng.uniform(-0.3, 0.3), -0.1 + rng.uniform(-0.5, 0.5)))
        exact, flat = matrix(local, great_circle), matrix(local, equirectangular)
        assert summary(exact)[1] < 100
        assert worst_triangle(flat) < 2e-5
        pairs = [(i, j) for i in range(120) for j in range(120) if i != j]
        drift = max(abs(flat[i][j] - exact[i][j]) / exact[i][j] for i, j in pairs)
        assert drift < 2e-5


class TestManhattan:
    def test_a_metric_longer_than_the_line_by_at_most_root_two(self):
        rng = random.Random(234)
        flat = [(rng.random(), rng.random()) for _ in range(120)]
        table = matrix(flat, manhattan)
        assert is_metric(table)
        euclid = matrix(flat, math.dist)
        pairs = [(i, j) for i in range(120) for j in range(120) if i != j]
        ratio = max(table[i][j] / euclid[i][j] for i, j in pairs)
        assert ratio <= math.sqrt(2) + 1e-9
        assert ratio > 1.4


class TestRefusals:
    def test_a_single_place_is_refused(self):
        with pytest.raises(Invalid):
            matrix([(0, 0)], great_circle)
