from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.geohash import bounds, decode, encode
from atlas.haversine import haversine


def _shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class TestEncodeDecode:
    def test_the_wikipedia_example(self):
        assert encode(57.64911, 10.40744, 11) == "u4pruydqqvj"

    def test_decode_returns_near_the_original(self):
        gh = encode(48.8566, 2.3522, 9)
        lat, lon = decode(gh)
        assert haversine(48.8566, 2.3522, lat, lon) * 1000 < 5  # within a few meters

    def test_longer_precision_is_more_exact(self):
        coarse = decode(encode(48.8566, 2.3522, 4))
        fine = decode(encode(48.8566, 2.3522, 10))
        d_coarse = haversine(48.8566, 2.3522, *coarse)
        d_fine = haversine(48.8566, 2.3522, *fine)
        assert d_fine < d_coarse


class TestPrefixProperty:
    def test_every_truncation_encloses_the_point(self):
        lat, lon = 48.8566, 2.3522
        gh = encode(lat, lon, 12)
        for p in range(1, 13):
            min_lat, min_lon, max_lat, max_lon = bounds(gh[:p])
            assert min_lat <= lat <= max_lat
            assert min_lon <= lon <= max_lon

    def test_shared_prefix_tracks_proximity(self):
        rng = random.Random(3)
        close, far = [], []
        for _ in range(2000):
            la, lo = rng.uniform(-60, 60), rng.uniform(-170, 170)
            close.append(_shared_prefix(encode(la, lo, 12), encode(la + 5e-4, lo + 5e-4, 12)))
            far.append(_shared_prefix(
                encode(la, lo, 12),
                encode(rng.uniform(-60, 60), rng.uniform(-170, 170), 12),
            ))
        assert sum(close) / len(close) > 5  # measured ~6.24
        assert sum(far) / len(far) < 0.5  # measured ~0.04

    def test_a_seam_makes_adjacent_points_share_no_prefix(self):
        # two points 3 meters apart across the origin split get disjoint hashes
        a = encode(1e-5, 1e-5, 8)
        b = encode(-1e-5, -1e-5, 8)
        assert a == "s0000000"
        assert b == "7zzzzzzz"
        assert _shared_prefix(a, b) == 0
        assert haversine(1e-5, 1e-5, -1e-5, -1e-5) * 1000 < 5


class TestRefusals:
    def test_a_latitude_off_the_globe_is_refused(self):
        with pytest.raises(Outside):
            encode(91, 0)

    def test_zero_precision_is_refused(self):
        with pytest.raises(Invalid):
            encode(0, 0, 0)

    def test_an_empty_geohash_is_refused(self):
        with pytest.raises(Invalid):
            bounds("")

    def test_a_bad_character_is_refused(self):
        with pytest.raises(Invalid):
            bounds("abcia")  # 'i' and 'a' are not in the alphabet
