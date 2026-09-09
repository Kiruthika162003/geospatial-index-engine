from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.tilecache import (
    LRUCache,
    belady,
    compulsory_floor,
    distinct,
    pan_session,
    replay,
    viewport_tiles,
    zipf_session,
    zoom_session,
)


@pytest.fixture(scope="module")
def pan():
    return pan_session(12, 400, random.Random(380))


class TestPanning:
    def test_the_scene(self, pan):
        assert len(viewport_tiles(10, 100.5, 100.5, 6, 4)) == 35
        assert len(pan) == 14000
        assert distinct(pan) == 997
        assert compulsory_floor(pan) == pytest.approx(0.9288, abs=1e-4)

    @pytest.mark.parametrize(
        ("capacity", "lru", "optimal"),
        [
            (10, 0.0, 0.2628),
            (20, 0.0, 0.5439),
            (35, 0.6499, 0.8903),
            (50, 0.8922, 0.9015),
            (400, 0.9173, 0.9288),
        ],
    )
    def test_lru_gets_nothing_below_a_viewport(self, pan, capacity, lru, optimal):
        assert replay(pan, capacity) == pytest.approx(lru, abs=1e-4)
        assert belady(pan, capacity) == pytest.approx(optimal, abs=1e-4)
        assert replay(pan, capacity) <= belady(pan, capacity) + 1e-9

    @pytest.mark.parametrize(
        ("stride", "seen", "floor", "small", "large"),
        [
            (0.25, 533, 0.9619, 0.8685, 0.9559),
            (1.0, 2539, 0.8186, 0.5653, 0.7992),
            (4.0, 9963, 0.2884, 0.1678, 0.2792),
        ],
    )
    def test_the_stride_sets_the_floor(self, stride, seen, floor, small, large):
        session = pan_session(12, 400, random.Random(383), stride=stride)
        assert distinct(session) == seen
        assert compulsory_floor(session) == pytest.approx(floor, abs=1e-4)
        assert replay(session, 35) == pytest.approx(small, abs=1e-4)
        assert replay(session, 100) == pytest.approx(large, abs=1e-4)
        assert floor - large < 0.02


class TestOtherHabits:
    def test_zooming_over_one_spot(self):
        session = zoom_session(8, 12, 400, random.Random(381))
        assert distinct(session) == 175
        assert replay(session, 35) == pytest.approx(0.495, abs=1e-4)
        assert replay(session, 100) == pytest.approx(0.7975, abs=1e-4)
        assert replay(session, 200) == pytest.approx(compulsory_floor(session), abs=1e-9)

    @pytest.mark.parametrize(
        ("capacity", "lru", "optimal"),
        [(35, 0.0469, 0.2265), (200, 0.2464, 0.5081), (800, 0.5966, 0.7664)],
    )
    def test_zipf_hotspots(self, capacity, lru, optimal):
        session = zipf_session(12, 400, random.Random(382))
        assert distinct(session) == 2333
        assert replay(session, capacity) == pytest.approx(lru, abs=1e-4)
        assert belady(session, capacity) == pytest.approx(optimal, abs=1e-4)


class TestPiecesAndRefusals:
    def test_the_cache_itself(self):
        cache = LRUCache(2)
        assert not cache.fetch((0, 0, 0))
        assert not cache.fetch((0, 1, 0))
        assert cache.fetch((0, 0, 0))
        assert not cache.fetch((0, 2, 0))
        assert not cache.fetch((0, 1, 0))
        assert cache.hit_rate() == pytest.approx(0.2)

    def test_refusals(self):
        with pytest.raises(Invalid):
            LRUCache(0)
        with pytest.raises(Invalid):
            LRUCache(1).hit_rate()
        with pytest.raises(Invalid):
            viewport_tiles(-1, 0, 0, 6, 4)
        with pytest.raises(Invalid):
            zoom_session(5, 3, 10, random.Random(1))
        with pytest.raises(Invalid):
            compulsory_floor([])
        with pytest.raises(Invalid):
            belady([(0, 0, 0)], 0)
