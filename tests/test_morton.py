from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.morton import decode, encode


class TestEncodeDecode:
    def test_known_codes(self):
        assert encode(0, 0) == 0
        assert encode(1, 0) == 1
        assert encode(0, 1) == 2
        assert encode(1, 1) == 3
        assert encode(2, 0) == 4
        assert encode(3, 3) == 15

    def test_decode_inverts_encode(self):
        rng = random.Random(9)
        for _ in range(200000):
            x, y = rng.randint(0, 65535), rng.randint(0, 65535)
            assert decode(encode(x, y)) == (x, y)


class TestLocality:
    def test_z_order_is_far_more_local_than_random_but_has_seams(self):
        pts = [(x, y) for x in range(16) for y in range(16)]
        zorder = sorted(pts, key=lambda p: encode(*p))
        jumps = [
            abs(zorder[i][0] - zorder[i + 1][0]) + abs(zorder[i][1] - zorder[i + 1][1])
            for i in range(len(zorder) - 1)
        ]
        mean_jump = sum(jumps) / len(jumps)
        # measured: mean 1.88, far below a shuffled baseline near 10.6
        assert mean_jump == pytest.approx(1.882, abs=0.05)
        # but the seams make the worst jump large: 16 across the grid
        assert max(jumps) == 16

    def test_the_seam_jumps_are_a_real_fraction(self):
        pts = [(x, y) for x in range(16) for y in range(16)]
        zorder = sorted(pts, key=lambda p: encode(*p))
        jumps = [
            abs(zorder[i][0] - zorder[i + 1][0]) + abs(zorder[i][1] - zorder[i + 1][1])
            for i in range(len(zorder) - 1)
        ]
        assert sum(1 for j in jumps if j > 1) == 127


class TestRefusals:
    def test_a_negative_coordinate_is_refused(self):
        with pytest.raises(Invalid):
            encode(-1, 0)

    def test_an_oversized_coordinate_is_refused(self):
        with pytest.raises(Invalid):
            encode(1 << 32, 0)

    def test_a_negative_code_is_refused(self):
        with pytest.raises(Invalid):
            decode(-1)
