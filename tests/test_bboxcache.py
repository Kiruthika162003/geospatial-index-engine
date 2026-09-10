from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.bboxcache import (
    FatBoxIndex,
    false_candidate_law,
    simulate,
    update_law_random,
    update_law_straight,
)
from atlas.errors import Invalid


class TestUpdates:
    @pytest.mark.parametrize(
        ("speed", "readings", "laws"),
        [
            (1.0, (0.158, 0.0833, 0.042, 0.0154, 0.0061), (0.2, 0.1, 0.05, 0.02, 0.01)),
            (3.0, (0.4508, 0.2355, 0.126, 0.0517, 0.0258), (0.6, 0.3, 0.15, 0.06, 0.03)),
        ],
    )
    def test_straight_motion_leaves_at_four_fifths_of_v_over_m(self, speed, readings, laws):
        for margin, reading, law in zip(
            (5.0, 10.0, 20.0, 50.0, 100.0), readings, laws, strict=True
        ):
            res = simulate(500, margin, speed, 200, random.Random(800), True)
            assert res["updates_per_object_step"] == pytest.approx(reading, abs=1e-4)
            assert update_law_straight(margin, speed) == pytest.approx(law)
            if margin <= 50.0 or speed > 1.0:
                assert 0.75 < reading / law < 0.9

    @pytest.mark.parametrize(
        ("speed", "readings"),
        [(1.0, (0.0286, 0.0066, 0.0007)), (3.0, (0.2056, 0.0603, 0.0159))],
    )
    def test_a_random_walk_leaves_with_the_square_of_the_margin(self, speed, readings):
        for margin, reading in zip((5.0, 10.0, 20.0), readings, strict=True):
            res = simulate(500, margin, speed, 200, random.Random(800), False)
            assert res["updates_per_object_step"] == pytest.approx(reading, abs=1e-4)
            assert reading < update_law_random(margin, speed)
            assert reading < update_law_straight(margin, speed)
        assert (
            simulate(500, 0.0, speed, 20, random.Random(800), False)["updates_per_object_step"]
            == 1.0
        )


class TestFalseCandidates:
    def test_the_band_law(self):
        readings = []
        for margin in (5.0, 20.0, 100.0):
            res = simulate(500, margin, 1.0, 200, random.Random(800), True)
            readings.append(round(res["false_candidates_per_query"], 2))
            assert (
                res["false_candidates_per_query"]
                >= false_candidate_law(500, margin, 200.0) * 0.9
            )
        assert readings == [2.29, 10.59, 65.75]
        assert false_candidate_law(500, 100.0, 200.0) == pytest.approx(60.0)

    def test_the_index_itself_and_refusals(self):
        index = FatBoxIndex([(10.0, 10.0)], 5.0)
        assert not index.move(0, (12.0, 12.0))
        assert index.move(0, (16.0, 10.0))
        assert index.updates == 1
        hits, false = index.query(BBox(0.0, 0.0, 30.0, 30.0), [(16.0, 10.0)])
        assert (hits, false) == ([0], 0)
        hits, false = index.query(BBox(20.0, 0.0, 30.0, 30.0), [(16.0, 10.0)])
        assert (hits, false) == ([], 1)
        with pytest.raises(Invalid):
            FatBoxIndex([], -1.0)
        with pytest.raises(Invalid):
            update_law_straight(5.0, 0.0)
        with pytest.raises(Invalid):
            update_law_random(5.0, 0.0)
