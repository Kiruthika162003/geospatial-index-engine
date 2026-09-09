from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.fastmarching import (
    dijkstra,
    fast_marching,
    manhattan_law,
    octile_law,
    ring_errors,
    wall_with_gap,
    worst_and_mean,
)

SIZE = 161
SOURCE = (80, 80)


@pytest.fixture(scope="module")
def fields():
    return {
        4: dijkstra(SIZE, SOURCE, 4),
        8: dijkstra(SIZE, SOURCE, 8),
        16: dijkstra(SIZE, SOURCE, 16),
        "fmm": fast_marching(SIZE, SOURCE),
    }


class TestMoveSets:
    @pytest.mark.parametrize(
        ("moves", "worst", "mean", "law"),
        [(4, 41.421, 27.904, 41.421), (8, 8.236, 5.335, 8.239), (16, 2.747, 1.412, None)],
    )
    def test_each_metric_has_its_own_error_that_distance_does_not_cure(
        self, fields, moves, worst, mean, law
    ):
        read_worst, read_mean, _ = worst_and_mean(ring_errors(fields[moves], SOURCE, 40.0))
        assert 100 * read_worst == pytest.approx(worst, abs=1e-3)
        assert 100 * read_mean == pytest.approx(mean, abs=1e-3)
        far_worst = worst_and_mean(ring_errors(fields[moves], SOURCE, 75.0))[0]
        assert 100 * far_worst == pytest.approx(worst, abs=0.05)
        if law is not None:
            assert 100 * read_worst <= law + 1e-3

    def test_the_laws(self):
        assert 100 * manhattan_law(45) == pytest.approx(41.421, abs=1e-3)
        assert 100 * octile_law(22.5) == pytest.approx(8.239, abs=1e-3)
        assert octile_law(45) == pytest.approx(0.0, abs=1e-12)
        assert octile_law(0) == pytest.approx(0.0, abs=1e-12)


class TestFastMarching:
    @pytest.mark.parametrize(
        ("radius", "worst", "mean"),
        [(5.0, 10.6, 6.373), (10.0, 7.305, 4.276), (40.0, 2.851, 1.777), (75.0, 1.78, 1.073)],
    )
    def test_the_error_falls_with_distance_and_sits_on_diagonals(
        self, fields, radius, worst, mean
    ):
        errors = ring_errors(fields["fmm"], SOURCE, radius)
        read_worst, read_mean, angle = worst_and_mean(errors)
        assert 100 * read_worst == pytest.approx(worst, abs=1e-2)
        assert 100 * read_mean == pytest.approx(mean, abs=1e-3)
        if radius >= 10:
            assert angle % 90 == pytest.approx(45.0, abs=1.0)

    def test_sixteen_moves_win_out_to_forty(self, fields):
        def worst(key, radius):
            return worst_and_mean(ring_errors(fields[key], SOURCE, radius))[0]

        for radius in (10.0, 40.0):
            assert worst(16, radius) < worst("fmm", radius)
        assert worst(16, 75.0) > worst("fmm", 75.0)
        assert worst(8, 10.0) > worst("fmm", 10.0)

    def test_the_wall_with_a_gap(self):
        wall = wall_with_gap(SIZE, 120, 80)
        truth = 40.0 + math.hypot(60, 30)
        assert dijkstra(SIZE, SOURCE, 8, wall)[20][150] == pytest.approx(112.426, abs=1e-3)
        assert dijkstra(SIZE, SOURCE, 16, wall)[20][150] == pytest.approx(truth, abs=1e-9)
        assert fast_marching(SIZE, SOURCE, wall)[20][150] == pytest.approx(108.619, abs=1e-3)
        assert dijkstra(SIZE, SOURCE, 8, wall)[80][150] == 70.0


class TestRefusals:
    def test_bad_grids_sources_and_moves(self):
        with pytest.raises(Invalid):
            dijkstra(0, (0, 0))
        with pytest.raises(Invalid):
            dijkstra(5, (9, 0))
        with pytest.raises(Invalid):
            dijkstra(5, (0, 0), 6)
        with pytest.raises(Invalid):
            fast_marching(5, (5, 5))
        with pytest.raises(Invalid):
            worst_and_mean([])
