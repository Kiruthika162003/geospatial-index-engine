from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.rasterbits import (
    PackedMask,
    intersection_count,
    naive_window_count,
    random_mask,
    timings,
)


class TestBytes:
    @pytest.mark.parametrize(("size", "count"), [(64, 1185), (256, 19773), (1024, 314496)])
    def test_an_eighth_of_the_bytes_and_a_round_trip(self, size, count):
        rng = random.Random(780)
        for s in (64, 256, 1024):
            mask = random_mask(s, 0.3, rng)
            if s == size:
                packed = PackedMask(mask)
                assert packed.bytes_bools() / packed.bytes_packed() == 8.0
                assert packed.bytes_packed() == size * size // 8
                assert packed.count() == count
                assert packed.unpack() == mask


class TestWindows:
    def test_popcounts_agree_and_win_only_at_wide_windows(self):
        mask = random_mask(256, 0.3, random.Random(781))
        packed = PackedMask(mask)
        q = random.Random(782)
        queries = [(q.randrange(256), q.randrange(256)) for _ in range(300)]
        assert all(
            packed.window_count(r, c, 7) == naive_window_count(mask, r, c, 7)
            for r, c in queries
        )
        fast_small, slow_small = timings(mask, 1, queries)
        fast_wide, slow_wide = timings(mask, 31, queries)
        assert slow_wide / fast_wide > 3.0
        assert slow_wide / fast_wide > slow_small / fast_small

    def test_set_get_rows_and_combinations(self):
        p = PackedMask([[True, False], [False, True]])
        p.set(0, 1, True)
        assert p.get(0, 1) and p.count() == 3 and p.row_count(0, 0, 1) == 2
        p.set(0, 1, False)
        assert not p.get(0, 1) and p.count() == 2
        a = random_mask(256, 0.3, random.Random(781))
        b = random_mask(256, 0.3, random.Random(783))
        assert intersection_count(a, b) == 5923
        assert PackedMask(a).union(PackedMask(b)).count() == 33395


class TestRefusals:
    def test_bad_masks_cells_spans_and_shapes(self):
        with pytest.raises(Invalid):
            PackedMask([])
        with pytest.raises(Invalid):
            PackedMask([[True], [True, False]])
        p = PackedMask([[True, False]])
        with pytest.raises(Invalid):
            p.get(1, 0)
        with pytest.raises(Invalid):
            p.set(0, 5, True)
        with pytest.raises(Invalid):
            p.row_count(0, 1, 0)
        with pytest.raises(Invalid):
            p.intersect(PackedMask([[True]]))
