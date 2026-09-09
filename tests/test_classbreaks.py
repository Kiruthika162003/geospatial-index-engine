from __future__ import annotations

import random

import pytest

from atlas.classbreaks import (
    assign,
    class_counts,
    equal_interval,
    goodness_of_variance_fit,
    jenks,
    lognormal,
    quantile,
    stability,
    two_clusters,
    uniform,
)
from atlas.errors import Invalid


def _gvf(values, classes):
    return tuple(
        round(goodness_of_variance_fit(values, rule(values, classes)), 4)
        for rule in (jenks, quantile, equal_interval)
    )


class TestSkewedData:
    @pytest.mark.parametrize(
        ("classes", "fits", "jenks_counts", "equal_counts"),
        [
            (3, (0.8313, 0.4018, 0.6719), [332, 61, 7], [387, 9, 4]),
            (5, (0.937, 0.5443, 0.8198), [219, 111, 47, 16, 7], [370, 21, 5, 3, 1]),
        ],
    )
    def test_jenks_beats_quantiles_on_a_lognormal(
        self, classes, fits, jenks_counts, equal_counts
    ):
        values = lognormal(400, random.Random(580))
        assert _gvf(values, classes) == fits
        assert class_counts(values, jenks(values, classes)) == jenks_counts
        assert class_counts(values, equal_interval(values, classes)) == equal_counts

    @pytest.mark.parametrize(
        ("classes", "fits", "equal_counts"),
        [
            (3, (0.9947, 0.7413, 0.9877), [208, 0, 192]),
            (5, (0.9976, 0.8596, 0.9947), [208, 0, 0, 77, 115]),
        ],
    )
    def test_two_clusters_leave_equal_intervals_empty(self, classes, fits, equal_counts):
        values = two_clusters(400, random.Random(580))
        assert _gvf(values, classes) == fits
        assert class_counts(values, equal_interval(values, classes)) == equal_counts
        assert 0 not in class_counts(values, jenks(values, classes))


class TestUniformData:
    @pytest.mark.parametrize(
        ("classes", "fits"), [(3, (0.8881, 0.8876, 0.8878)), (5, (0.9608, 0.9591, 0.9604))]
    )
    def test_the_three_rules_tie(self, classes, fits):
        values = uniform(400, random.Random(580))
        read = _gvf(values, classes)
        assert read == fits
        assert max(read) - min(read) < 0.002

    def test_stability(self):
        assert stability(lambda r: lognormal(400, r), 5, 10, 590) == pytest.approx(
            12.855, abs=1e-3
        )
        assert stability(lambda r: uniform(400, r), 5, 10, 590) == pytest.approx(
            7.715, abs=1e-3
        )


class TestPieces:
    def test_assignment_and_refusals(self):
        assert assign([1, 5, 10], [4.0, 8.0]) == [0, 1, 2]
        assert jenks([1, 2, 3, 10, 11, 12], 2) == [10]
        assert goodness_of_variance_fit([1, 2, 3, 10, 11, 12], [10]) == pytest.approx(
            0.9681, abs=1e-4
        )
        assert goodness_of_variance_fit([4, 4, 4], [4]) == 1.0
        with pytest.raises(Invalid):
            jenks([1, 2], 3)
        with pytest.raises(Invalid):
            quantile([1, 2], 0)
        with pytest.raises(Invalid):
            goodness_of_variance_fit([], [])
