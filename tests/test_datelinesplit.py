from __future__ import annotations

import random

import pytest

from atlas.datelinesplit import (
    crosses_dateline,
    great_circle_gap_km,
    piece_span,
    planar_span,
    short_delta,
    split,
)
from atlas.errors import Outside

EASTWARD = ((10.0, 170.0), (20.0, -170.0))


class TestTheCut:
    def test_an_eastward_crossing_is_cut_at_the_seam(self):
        assert split(EASTWARD) == [
            ((10.0, 170.0), (15.0, 180.0)),
            ((15.0, -180.0), (20.0, -170.0)),
        ]

    def test_a_westward_crossing_is_cut_at_the_other_edge(self):
        assert split(((5.0, -170.0), (-5.0, 170.0))) == [
            ((5.0, -170.0), (0.0, -180.0)),
            ((0.0, 180.0), (-5.0, 170.0)),
        ]

    def test_the_pieces_span_the_short_way(self):
        assert planar_span(EASTWARD) == 20.0
        assert piece_span(split(EASTWARD)) == 20.0


class TestNoCut:
    def test_a_segment_away_from_the_seam_comes_back_whole(self):
        assert split(((0, 10), (0, 20))) == [((0, 10), (0, 20))]

    def test_a_segment_ending_on_the_seam_comes_back_whole(self):
        assert split(((0, 170), (0, 180))) == [((0, 170), (0, 180))]

    def test_the_decision_goes_the_short_way_round(self):
        assert not crosses_dateline(10, 20)
        assert crosses_dateline(170, -170)
        assert crosses_dateline(-170, 170)
        assert short_delta(170, -170) == 20.0
        assert short_delta(-170, 170) == -20.0


class TestTheLengthIdentity:
    def test_planar_spans_sum_exactly_over_random_segments(self):
        rng = random.Random(174)
        for _ in range(5000):
            a = (rng.uniform(-80, 80), rng.uniform(-180, 180))
            b = (rng.uniform(-80, 80), rng.uniform(-180, 180))
            assert abs(planar_span((a, b)) - piece_span(split((a, b)))) < 1e-9

    def test_the_geodesic_gap_grows_with_the_span(self):
        assert great_circle_gap_km(EASTWARD) == pytest.approx(0.9186, abs=1e-3)
        rng = random.Random(175)
        worsts = []
        for span in (5, 10, 20, 40):
            worst = 0.0
            for _ in range(200):
                lat1 = rng.uniform(-60, 60)
                lat2 = lat1 + rng.uniform(-10, 10)
                lon1 = 180 - rng.uniform(0, span)
                lon2 = lon1 + span
                if lon2 > 180:
                    lon2 -= 360
                worst = max(worst, abs(great_circle_gap_km(((lat1, lon1), (lat2, lon2)))))
            worsts.append(worst)
        assert worsts == sorted(worsts)
        assert worsts[0] < 1.0  # measured 0.57 km at five degrees
        assert worsts[2] > 5.0  # measured 10.3 km at twenty degrees, not a few meters


class TestRefusals:
    def test_a_longitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            split(((0, 181), (0, 0)))
