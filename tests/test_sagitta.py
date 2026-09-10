from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.sagitta import (
    chord_point,
    cross_track_km,
    east_west_leg,
    great_circle_distance_km,
    great_circle_point,
    sagitta_km,
    segments_for_tolerance,
    sine_guess,
    tangent_law,
)


class TestBows:
    @pytest.mark.parametrize(
        ("lat", "readings", "legs", "ratios"),
        [
            (30.0, (10.53, 96.46, 410.32), (962.7, 2880.5, 5706.3), (1.158, 1.185, 1.285)),
            (45.0, (12.14, 110.41, 456.63), (785.8, 2345.2, 4604.5), (1.418, 1.447, 1.552)),
            (60.0, (10.51, 94.8, 381.95), (555.4, 1653.6, 3219.7), (2.004, 2.041, 2.168)),
        ],
    )
    def test_the_bow_carries_the_tangent(self, lat, readings, legs, ratios):
        for span, bow, leg, ratio in zip(
            (10.0, 30.0, 60.0), readings, legs, ratios, strict=True
        ):
            a, b = east_west_leg(lat, span)
            d = great_circle_distance_km(a, b)
            assert d == pytest.approx(leg, abs=0.1)
            assert sagitta_km(a, b) == pytest.approx(bow, abs=0.01)
            assert sagitta_km(a, b) / sine_guess(d, lat) == pytest.approx(ratio, abs=1e-3)
            if span == 10.0:
                assert sagitta_km(a, b) == pytest.approx(tangent_law(d, lat), rel=0.003)
        assert math.cos(math.radians(lat)) * ratios[0] == pytest.approx(1.0, abs=0.01)

    def test_the_equator_the_meridian_and_the_atlantic(self):
        assert sagitta_km(*east_west_leg(0.0, 60.0)) == pytest.approx(0.0, abs=1e-6)
        assert sagitta_km((0.0, 10.0), (60.0, 10.0)) == pytest.approx(0.0, abs=1e-6)
        a, b = (40.0, -74.0), (51.5, 0.0)
        assert great_circle_distance_km(a, b) == pytest.approx(5628.4, abs=0.1)
        assert sagitta_km(a, b) == pytest.approx(723.44, abs=0.01)
        assert segments_for_tolerance(a, b, 1.0) == 32
        assert segments_for_tolerance(a, b, 5.0) == 16

    @pytest.mark.parametrize(
        ("span", "fine", "coarse"), [(10.0, 4, 2), (30.0, 16, 4), (60.0, 32, 8)]
    )
    def test_pieces_for_a_tolerance(self, span, fine, coarse):
        for lat in (30.0, 60.0):
            a, b = east_west_leg(lat, span)
            assert segments_for_tolerance(a, b, 1.0) == fine
            assert segments_for_tolerance(a, b, 10.0) == coarse


class TestPieces:
    def test_points_and_refusals(self):
        a, b = east_west_leg(45.0, 60.0)
        assert great_circle_point(a, b, 0.5) == pytest.approx((49.107, 0.0), abs=1e-3)
        assert chord_point(a, b, 0.5) == pytest.approx((45.0, 0.0), abs=1e-9)
        assert cross_track_km(a, b, great_circle_point(a, b, 0.3)) == pytest.approx(
            0.0, abs=1e-6
        )
        with pytest.raises(Invalid):
            great_circle_point(a, b, 1.5)
        with pytest.raises(Invalid):
            sagitta_km(a, b, 1)
        with pytest.raises(Invalid):
            segments_for_tolerance(a, b, 0.0)
        with pytest.raises(Invalid):
            cross_track_km(a, a, b)
