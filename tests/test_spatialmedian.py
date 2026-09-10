from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.spatialmedian import (
    breakdown_count,
    centroid,
    centroid_shift_law,
    gaussian_cloud,
    line_cloud,
    medoid,
    shift,
    summed_distance,
    summed_squares,
    weiszfeld,
    with_outlier,
    with_outliers,
)


@pytest.fixture(scope="module")
def cloud():
    points = gaussian_cloud(200, random.Random(720))
    return points, centroid(points), weiszfeld(points)[0]


class TestClean:
    def test_each_centre_wins_its_own_objective(self, cloud):
        points, c0, m0 = cloud
        assert weiszfeld(points)[1] == 27
        assert shift(c0, m0) < 0.15
        assert summed_distance(points, m0) == pytest.approx(230.984, abs=1e-3)
        assert summed_distance(points, c0) == pytest.approx(231.774, abs=1e-3)
        assert summed_squares(points, c0) == pytest.approx(341.815, abs=1e-3)
        assert summed_squares(points, m0) == pytest.approx(343.781, abs=1e-3)
        assert summed_distance(points, m0) < summed_distance(points, c0)
        assert summed_squares(points, c0) < summed_squares(points, m0)


class TestOutliers:
    @pytest.mark.parametrize("distance", [10.0, 100.0, 1000.0])
    def test_one_outlier(self, cloud, distance):
        points, c0, m0 = cloud
        dirty = with_outlier(points, distance)
        assert shift(c0, centroid(dirty)) == pytest.approx(
            centroid_shift_law(distance, 200), abs=1e-6
        )
        assert shift(m0, weiszfeld(dirty)[0]) == pytest.approx(0.0063, abs=1e-4)
        assert shift(medoid(points), medoid(dirty)) == 0.0

    @pytest.mark.parametrize(
        ("count", "median", "mean"),
        [(10, 0.0135, 1.079), (100, 0.0617, 3.877), (150, 0.0471, 2.198)],
    )
    def test_random_directions_cancel(self, cloud, count, median, mean):
        points, c0, m0 = cloud
        dirty = with_outliers(points, count, 100.0, random.Random(721))
        assert shift(m0, weiszfeld(dirty)[0]) == pytest.approx(median, abs=1e-4)
        assert shift(c0, centroid(dirty)) == pytest.approx(mean, abs=1e-3)

    @pytest.mark.parametrize(
        ("count", "moved"), [(50, 0.316), (150, 1.351), (199, 9.495), (201, 99.364)]
    )
    def test_the_breakdown_point_is_one_half(self, cloud, count, moved):
        points, _, m0 = cloud
        stacked = [*points, *([(100.0, 0.0)] * count)]
        assert shift(m0, weiszfeld(stacked)[0]) == pytest.approx(moved, abs=1e-3)

    def test_random_outliers_never_break_it(self, cloud):
        points, _, _ = cloud
        assert breakdown_count(points, 100.0, random.Random(723), 5.0, cap=60) == 60


class TestConvergence:
    def test_steps_and_the_line(self):
        assert [
            weiszfeld(gaussian_cloud(n, random.Random(724)))[1] for n in (50, 200, 1000)
        ] == [30, 26, 28]
        line = line_cloud(21)
        assert weiszfeld(line) == ((10.0, 0.0), 1)
        assert centroid(line) == (10.0, 0.0)

    def test_refusals(self):
        with pytest.raises(Invalid):
            centroid([])
        with pytest.raises(Invalid):
            weiszfeld([(0.0, 0.0)], tolerance=0)
        with pytest.raises(Invalid):
            medoid([])
