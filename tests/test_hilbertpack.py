from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.hilbertpack import (
    METHODS,
    clustered_boxes,
    compare,
    diagonal_boxes,
    hilbert_leaves,
    ideal_touches,
    leaves_by,
    overlap,
    pack_by_key,
    square_queries,
    street_boxes,
    touches,
    uniform_boxes,
)


@pytest.fixture(scope="module")
def sets():
    rng = random.Random(270)
    return {
        "uniform": uniform_boxes(2000, rng),
        "sized": uniform_boxes(2000, rng, size=10.0),
        "clustered": clustered_boxes(2000, 10, rng),
        "diagonal": diagonal_boxes(2000),
        "streets": street_boxes(2000, 8, rng),
    }


def _queries():
    return square_queries(300, random.Random(1), 50.0)


class TestUniformPoints:
    def test_zero_overlap_strips_touch_eleven_leaves_and_str_beats_the_tile_model(self, sets):
        res = compare(sets["uniform"], _queries(), 8)
        assert all(s["leaves"] == 250 for s in res.values())
        assert res["str"]["overlap"] == 0.0
        assert res["x_sorted"]["overlap"] == 0.0
        assert res["str"]["touches"] == pytest.approx(2.71, abs=0.01)
        assert res["x_sorted"]["touches"] == pytest.approx(11.32, abs=0.01)
        assert res["x_sorted"]["margin"] / res["str"]["margin"] == pytest.approx(7.32, abs=0.02)
        assert ideal_touches(2000, 8, 50.0) == pytest.approx(3.206, abs=1e-3)
        assert res["hilbert"]["overlap"] == pytest.approx(104791, abs=1)
        assert res["hilbert"]["touches"] == pytest.approx(3.043, abs=1e-3)
        ratio = res["morton"]["overlap"] / res["hilbert"]["overlap"]
        assert ratio == pytest.approx(8.81, abs=0.01)
        assert res["morton"]["touches"] == pytest.approx(4.44, abs=0.01)

    def test_sized_boxes_cost_str_its_zero(self, sets):
        res = compare(sets["sized"], _queries(), 8)
        assert res["str"]["overlap"] == pytest.approx(80728, abs=1)
        expected = {"str": 3.107, "hilbert": 3.547, "morton": 5.15, "x_sorted": 13.69}
        for method, value in expected.items():
            assert res[method]["touches"] == pytest.approx(value, abs=1e-3)


class TestNonUniformPoints:
    def test_hilbert_beats_str_on_clusters_and_streets(self, sets):
        clustered = compare(sets["clustered"], _queries(), 8)
        assert clustered["str"]["overlap"] == 0.0
        assert clustered["hilbert"]["overlap"] == pytest.approx(43361, abs=1)
        assert clustered["hilbert"]["touches"] < clustered["str"]["touches"]
        assert clustered["hilbert"]["area"] < clustered["str"]["area"]
        streets = compare(sets["streets"], _queries(), 8)
        assert streets["str"]["touches"] == pytest.approx(2.597, abs=1e-3)
        assert streets["hilbert"]["touches"] == pytest.approx(2.133, abs=1e-3)
        assert streets["morton"]["touches"] < streets["str"]["touches"]

    def test_a_diagonal_is_the_same_under_every_order(self, sets):
        res = compare(sets["diagonal"], _queries(), 8)
        readings = {tuple(sorted(s.items())) for s in res.values()}
        assert len(readings) == 1
        assert res["str"]["touches"] == pytest.approx(0.923, abs=1e-3)


class TestCapacityAndOrder:
    def test_capacity_sweep(self):
        boxes = uniform_boxes(2000, random.Random(271))
        queries = square_queries(300, random.Random(2), 50.0)
        expected = {
            4: (3.18, 3.42, 4.83, 17.81),
            32: (1.75, 2.24, 3.54, 4.08),
        }
        for capacity, values in expected.items():
            res = compare(boxes, queries, capacity)
            got = tuple(res[m]["touches"] for m in METHODS)
            assert got == pytest.approx(values, abs=0.01)
            assert res["str"]["touches"] < ideal_touches(2000, capacity, 50.0)

    def test_the_curve_order_saturates_at_eight_bits(self):
        boxes = uniform_boxes(2000, random.Random(271))
        queries = square_queries(300, random.Random(2), 50.0)
        readings = {}
        for order in (4, 8, 10):
            leaves = hilbert_leaves(boxes, 8, order)
            readings[order] = (round(overlap(leaves)), round(touches(leaves, queries), 2))
        assert readings[4] == (464067, 3.67)
        assert readings[8] == (104204, 2.95)
        assert readings[10] == (108713, 2.95)


class TestRefusals:
    def test_bad_inputs(self):
        with pytest.raises(Invalid):
            pack_by_key([], [1], 8)
        with pytest.raises(Invalid):
            pack_by_key(diagonal_boxes(3), [1, 2, 3], 0)
        with pytest.raises(Invalid):
            leaves_by("random", diagonal_boxes(3))
        with pytest.raises(Invalid):
            compare([], _queries())
        with pytest.raises(Invalid):
            touches(diagonal_boxes(3), [])
        assert leaves_by("hilbert", []) == []
        assert leaves_by("x_sorted", []) == []
