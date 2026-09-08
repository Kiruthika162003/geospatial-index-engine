from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.tilecover import brute_cover, count, cover

LONDON = (-0.5, 51.3, 0.3, 51.7)


class TestCover:
    def test_it_matches_an_exhaustive_scan_at_zoom_twelve(self):
        assert set(cover(*LONDON, 12)) == set(brute_cover(*LONDON, 12))
        assert count(*LONDON, 12) == 80

    def test_the_count_is_the_product_of_the_two_spans(self):
        tiles = cover(*LONDON, 12)
        columns = {x for x, _ in tiles}
        rows = {y for _, y in tiles}
        assert len(tiles) == len(columns) * len(rows)

    def test_random_boxes_match_the_brute_scan(self):
        rng = random.Random(107)
        for _ in range(400):
            z = rng.randint(1, 6)
            w, e = sorted((rng.uniform(-179, 179), rng.uniform(-179, 179)))
            s, n = sorted((rng.uniform(-80, 80), rng.uniform(-80, 80)))
            assert set(cover(w, s, e, n, z)) == set(brute_cover(w, s, e, n, z))


class TestZoomGrowth:
    def test_the_cover_quadruples_only_once_the_box_spans_many_tiles(self):
        counts = [count(*LONDON, z) for z in range(8, 14)]
        assert counts == [4, 4, 9, 20, 80, 304]
        # a tile-sized sliver barely grows; a many-tile box grows about fourfold
        assert counts[1] / counts[0] == pytest.approx(1.0)
        assert counts[4] / counts[3] == pytest.approx(4.0)
        assert counts[5] / counts[4] == pytest.approx(3.8)


class TestAntimeridian:
    def test_a_straddling_box_wraps_and_matches_the_brute_scan(self):
        tiles = cover(170, -10, -170, 10, 4)
        assert set(tiles) == set(brute_cover(170, -10, -170, 10, 4))
        assert sorted({x for x, _ in tiles}) == [0, 15]


class TestRefusals:
    def test_a_negative_zoom_is_refused(self):
        with pytest.raises(Invalid):
            cover(*LONDON, -1)

    def test_south_above_north_is_refused(self):
        with pytest.raises(Invalid):
            cover(0, 10, 1, 5, 3)

    def test_a_latitude_past_the_mercator_limit_is_refused(self):
        with pytest.raises(Outside):
            cover(0, 0, 1, 89, 3)
