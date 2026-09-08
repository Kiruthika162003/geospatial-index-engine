from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.slippytile import (
    from_quadkey,
    ground_resolution,
    lonlat_to_tile,
    tile_bounds,
    tile_count,
    to_quadkey,
)


class TestTileAddressing:
    def test_known_london_tile(self):
        assert lonlat_to_tile(-0.1278, 51.5074, 12) == (2046, 1362)

    def test_zoom_zero_is_one_world_tile(self):
        assert lonlat_to_tile(0, 0, 0) == (0, 0)
        assert tile_count(0) == 1

    def test_the_tile_bounds_contain_their_point(self):
        lon, lat = -0.1278, 51.5074
        tx, ty = lonlat_to_tile(lon, lat, 12)
        south, west, north, east = tile_bounds(tx, ty, 12)
        assert south <= lat <= north
        assert west <= lon <= east


class TestTheDoublingLaws:
    def test_tile_count_is_four_to_the_zoom(self):
        assert [tile_count(z) for z in range(6)] == [1, 4, 16, 64, 256, 1024]

    def test_ground_resolution_halves_each_zoom(self):
        base = ground_resolution(0, 0)
        assert base == pytest.approx(156543.03, abs=0.1)
        for z in range(1, 12):
            assert ground_resolution(0, z) == pytest.approx(base / (1 << z))


class TestQuadkey:
    def test_roundtrip_and_length(self):
        rng = random.Random(37)
        for _ in range(50000):
            z = rng.randint(1, 20)
            n = 1 << z
            x, y = rng.randint(0, n - 1), rng.randint(0, n - 1)
            qk = to_quadkey(x, y, z)
            assert len(qk) == z
            assert from_quadkey(qk) == (x, y, z)

    def test_a_parent_quadkey_is_a_prefix_of_its_child(self):
        x, y, z = 5, 9, 8
        assert to_quadkey(x, y, z).startswith(to_quadkey(x >> 1, y >> 1, z - 1))


class TestRefusals:
    def test_a_negative_zoom_is_refused(self):
        with pytest.raises(Invalid):
            lonlat_to_tile(0, 0, -1)

    def test_a_latitude_past_the_limit_is_refused(self):
        with pytest.raises(Outside):
            lonlat_to_tile(0, 86, 5)

    def test_a_bad_quadkey_is_refused(self):
        with pytest.raises(Invalid):
            from_quadkey("012x")
