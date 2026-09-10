from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.geohashcompress import (
    city,
    coded_bytes,
    common_prefix,
    decode_error_m,
    front_code,
    front_decode,
    hashes,
    log32_law,
    mean_shared,
    ratio,
    uniform_world,
)


class TestWorld:
    @pytest.mark.parametrize(
        ("precision", "coded", "raw", "unsorted", "error"),
        [
            (6, 5.787, 7.0, 7.967, 611.441),
            (8, 7.787, 9.0, 9.967, 19.088),
            (12, 11.787, 13.0, 13.967, 0.019),
        ],
    )
    def test_a_character_saved_and_unsorted_worse_than_raw(
        self, precision, coded, raw, unsorted, error
    ):
        points = uniform_world(20000, random.Random(680))
        assert ratio(points, precision) == pytest.approx((coded, raw), abs=1e-3)
        assert ratio(points, precision, presorted=False)[0] == pytest.approx(unsorted, abs=1e-3)
        assert ratio(points, precision, presorted=False)[0] > raw
        assert decode_error_m(points[:2000], precision) == pytest.approx(error, abs=1e-3)

    @pytest.mark.parametrize(
        ("n", "shared", "law"),
        [(1000, 1.346, 1.993), (20000, 2.21, 2.858), (80000, 2.65, 3.258)],
    )
    def test_neighbours_share_two_thirds_of_a_character_under_the_law(self, n, shared, law):
        sorted_hashes = sorted(hashes(uniform_world(n, random.Random(683)), 12))
        assert mean_shared(sorted_hashes) == pytest.approx(shared, abs=1e-3)
        assert log32_law(n) == pytest.approx(law, abs=1e-3)
        assert 0.55 < law - shared < 0.75


class TestCities:
    @pytest.mark.parametrize(
        ("spread", "precision", "coded", "shared"),
        [
            (0.05, 6, 2.068, 5.933),
            (0.05, 12, 7.667, 6.333),
            (0.005, 8, 2.373, 7.627),
            (0.005, 12, 6.316, 7.685),
        ],
    )
    def test_a_city_codes_to_a_few_bytes(self, spread, precision, coded, shared):
        seed = 681 if spread == 0.05 else 682
        points = city(20000, random.Random(seed), spread=spread)
        assert ratio(points, precision)[0] == pytest.approx(coded, abs=1e-3)
        assert mean_shared(sorted(hashes(points, precision))) == pytest.approx(shared, abs=1e-3)

    def test_round_trip_and_collapse(self):
        points = city(20000, random.Random(681))
        sorted_hashes = sorted(hashes(points, 10))
        coded = front_code(sorted_hashes)
        assert front_decode(coded) == sorted_hashes
        assert coded_bytes(coded) == 20000 * 2 + sum(len(s) for _, s in coded)
        assert len(set(hashes(points, 6))) == 1272
        assert common_prefix("gcpvj", "gcpvn") == 4


class TestRefusals:
    def test_bad_precision_codes_and_laws(self):
        with pytest.raises(Invalid):
            hashes([(0.0, 0.0)], 0)
        with pytest.raises(Invalid):
            front_decode([(3, "abc")])
        with pytest.raises(Invalid):
            mean_shared(["abc"])
        with pytest.raises(Invalid):
            log32_law(1)
