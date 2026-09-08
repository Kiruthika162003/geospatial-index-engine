from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.viewshed import viewshed, visible_count, visible_from

N = 21
RIDGE = [[10.0 if c == 10 else 0.0 for c in range(N)] for r in range(N)]


class TestFlatGround:
    def test_everything_is_visible_with_a_raised_eye(self):
        flat = [[0.0] * N for _ in range(N)]
        assert visible_count(viewshed(flat, (10, 10), 1.0)) == N * N

    def test_the_viewer_sees_its_own_cell(self):
        flat = [[0.0] * N for _ in range(N)]
        assert visible_from(flat, (3, 3), 0.0, (3, 3))


class TestRidge:
    def test_the_near_side_is_seen_and_the_far_side_hidden(self):
        shed = viewshed(RIDGE, (10, 2), 1.5)
        near = sum(1 for r in range(N) for c in range(N) if c < 10 and shed[r][c])
        far = sum(1 for r in range(N) for c in range(N) if c > 10 and shed[r][c])
        assert near == 10 * N
        assert far == 0

    def test_far_side_verdicts_match_the_exact_geometry(self):
        shed = viewshed(RIDGE, (10, 2), 1.5)
        for r in range(N):
            for c in range(11, N):
                t = (10 - 2) / (c - 2)
                line_height_at_crest = 1.5 + (0.0 - 1.5) * t
                assert shed[r][c] == (line_height_at_crest >= 10.0)

    def test_a_low_eye_on_the_crest_misses_the_flanks_and_a_tall_one_sees_all(self):
        # the refined guess: 369 of 441 at 1.5 m, the 72 hidden all within three columns
        low = viewshed(RIDGE, (10, 10), 1.5)
        hidden = [(r, c) for r in range(N) for c in range(N) if not low[r][c]]
        assert len(hidden) == 72
        assert all(abs(c - 10) <= 3 for _, c in hidden)
        assert visible_count(viewshed(RIDGE, (10, 10), 5.0)) == 409
        assert visible_count(viewshed(RIDGE, (10, 10), 20.0)) == N * N


class TestEyeHeight:
    def test_the_visible_count_never_falls_as_the_eye_rises(self):
        rng = random.Random(153)
        rough = [
            [5 * math.sin(r / 3) + 5 * math.cos(c / 4) + rng.uniform(0, 1) for c in range(N)]
            for r in range(N)
        ]
        heights = (0.0, 0.5, 1.0, 2.0, 5.0, 20.0)
        counts = [visible_count(viewshed(rough, (10, 10), h)) for h in heights]
        assert counts == [199, 226, 256, 286, 321, 430]
        assert counts == sorted(counts)


class TestRefusals:
    def test_an_off_grid_target_is_refused(self):
        with pytest.raises(Invalid):
            visible_from(RIDGE, (0, 0), 1.0, (50, 50))

    def test_a_negative_eye_height_is_refused(self):
        with pytest.raises(Invalid):
            visible_from(RIDGE, (0, 0), -1.0, (1, 1))

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(Invalid):
            visible_from([], (0, 0), 1.0, (0, 0))
