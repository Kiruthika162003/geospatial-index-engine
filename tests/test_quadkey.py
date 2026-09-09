from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.quadkey import (
    children,
    common_prefix,
    covering,
    decode,
    encode,
    expected_prefix_histogram,
    key_count_by_zoom,
    neighbours,
    parent,
    prefix_histogram,
    shared_with_neighbours,
)


class TestKeys:
    def test_round_trips_and_family(self):
        assert encode(3, 3, 5) == "213"
        assert decode("213") == (3, 3, 5)
        for zoom in range(6):
            for x in range(1 << zoom):
                for y in range(1 << zoom):
                    assert decode(encode(zoom, x, y)) == (zoom, x, y)
        assert parent("213") == "21"
        assert children("21") == ["210", "211", "212", "213"]
        assert key_count_by_zoom(4) == 256
        assert encode(0, 0, 0) == ""

    def test_the_central_seam_shares_nothing(self):
        half = 1 << 9
        west, east = encode(10, half - 1, half), encode(10, half, half)
        assert (west, east) == ("2111111111", "3000000000")
        assert common_prefix(west, east) == 0
        assert shared_with_neighbours(10, half - 1, half) == [0, 9, 9, 0]
        assert shared_with_neighbours(10, 0, 0) == [9, 0, 9]
        assert shared_with_neighbours(10, 1, 1) == [8, 9, 8, 9]
        assert len(neighbours(10, 0, 0)) == 3


class TestTheHalvingLaw:
    def test_the_prefix_histogram_halves_and_averages_zoom_minus_two(self):
        hist = prefix_histogram(10, random.Random(400), 20000)
        counts = [20, 49, 76, 151, 303, 609, 1227, 2568, 5043, 9939]
        assert hist == dict(enumerate(counts))
        total = sum(hist.values())
        law = expected_prefix_histogram(10)
        for shared in range(4, 10):
            assert hist[shared] / total == pytest.approx(law[shared], rel=0.05)
        mean = sum(k * v for k, v in hist.items()) / total
        assert mean == pytest.approx(8.0082, abs=1e-4)

    @pytest.mark.parametrize(
        ("zoom", "mean", "zero"), [(5, 3.1701, 0.0299), (18, 16.0067, 0.0001)]
    )
    def test_other_zooms(self, zoom, mean, zero):
        hist = prefix_histogram(zoom, random.Random(401), 20000)
        total = sum(hist.values())
        assert sum(k * v for k, v in hist.items()) / total == pytest.approx(mean, abs=1e-4)
        assert hist.get(0, 0) / total == pytest.approx(zero, abs=1e-4)


class TestCovering:
    @pytest.mark.parametrize(
        ("box", "keys", "lengths"),
        [
            ((0, 0, 3, 3), 1, [2]),
            ((0, 0, 7, 7), 1, [1]),
            ((0, 0, 15, 15), 1, [0]),
            ((1, 1, 6, 6), 24, [3, 4]),
            ((3, 3, 12, 12), 40, [2, 4]),
            ((0, 0, 1, 15), 8, [3]),
        ],
    )
    def test_boxes_merge_only_at_whole_quadrants(self, box, keys, lengths):
        cover = covering(4, *box)
        assert len(cover) == keys
        assert sorted({len(k) for k in cover}) == lengths
        xs, ys = range(box[0], box[2] + 1), range(box[1], box[3] + 1)
        tiles = {encode(4, x, y) for x in xs for y in ys}
        covered = {t for t in tiles if any(t.startswith(k) for k in cover)}
        assert covered == tiles

    def test_refusals(self):
        with pytest.raises(Invalid):
            encode(-1, 0, 0)
        with pytest.raises(Invalid):
            encode(2, 4, 0)
        with pytest.raises(Invalid):
            decode("21x")
        with pytest.raises(Invalid):
            parent("")
        with pytest.raises(Invalid):
            covering(4, 3, 3, 1, 1)
        with pytest.raises(Invalid):
            prefix_histogram(4, random.Random(1), 0)
