from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.rasterrle import (
    blobs,
    boundary_length,
    bytes_needed,
    decode,
    disc,
    encode,
    expected_noise_runs,
    hilbert_order,
    noise,
    raw_bytes,
    row_order,
    run_count,
    stripes,
)

SIZE = 64
ROW = row_order(SIZE, SIZE)
HIL = hilbert_order(SIZE, SIZE)


class TestGrain:
    def test_stripes_swap_the_winner_with_their_direction(self):
        vertical, horizontal = stripes(SIZE, 8, True), stripes(SIZE, 8, False)
        assert (run_count(vertical, ROW), run_count(vertical, HIL)) == (1024, 128)
        assert (run_count(horizontal, ROW), run_count(horizontal, HIL)) == (16, 129)
        assert boundary_length(vertical) == boundary_length(horizontal) == 1088
        assert bytes_needed(encode(vertical, ROW)) == 2176
        assert bytes_needed(encode(horizontal, ROW)) == 34
        assert raw_bytes(SIZE, SIZE) == 512

    @pytest.mark.parametrize(
        ("radius", "rows", "curve"), [(20, 81, 67), (30, 121, 111)]
    )
    def test_discs_in_the_64_grid(self, radius, rows, curve):
        mask = disc(SIZE, radius)
        assert (run_count(mask, ROW), run_count(mask, HIL)) == (rows, curve)
        assert boundary_length(mask) == 8 * radius
        assert decode(encode(mask, HIL), HIL, SIZE, SIZE) == mask

    @pytest.mark.parametrize(
        ("side", "rows", "curve"),
        [(16, 25, 27), (32, 53, 59), (128, 205, 223), (256, 409, 395)],
    )
    def test_the_hilbert_walk_is_no_better_on_discs(self, side, rows, curve):
        mask = disc(side, side * 0.4)
        assert run_count(mask, row_order(side, side)) == rows
        assert run_count(mask, hilbert_order(side, side)) == curve
        assert rows / boundary_length(mask) == pytest.approx(0.5, abs=0.025)

    def test_blobs(self):
        mask = blobs(SIZE, 12, 6.0, random.Random(430))
        assert (run_count(mask, ROW), run_count(mask, HIL)) == (211, 205)


class TestNoise:
    @pytest.mark.parametrize(
        ("density", "rows", "curve", "law"),
        [(0.01, 89, 89, 82.1), (0.05, 369, 371, 390.0), (0.5, 2060, 2042, 2048.5)],
    )
    def test_noise_defeats_both_walks_alike(self, density, rows, curve, law):
        mask = noise(SIZE, density, random.Random(431))
        assert (run_count(mask, ROW), run_count(mask, HIL)) == (rows, curve)
        assert expected_noise_runs(SIZE, density) == pytest.approx(law, abs=0.1)
        assert bytes_needed(encode(mask, ROW)) == {0.01: 190, 0.05: 785, 0.5: 4378}[density]


class TestRefusals:
    def test_bad_masks_walks_and_runs(self):
        with pytest.raises(Invalid):
            encode([], ROW)
        with pytest.raises(Invalid):
            encode(disc(4, 1), ROW)
        with pytest.raises(Invalid):
            hilbert_order(6, 6)
        with pytest.raises(Invalid):
            hilbert_order(4, 8)
        with pytest.raises(Invalid):
            decode([(True, 3)], row_order(2, 2), 2, 2)
        with pytest.raises(Invalid):
            stripes(8, 1, True)
