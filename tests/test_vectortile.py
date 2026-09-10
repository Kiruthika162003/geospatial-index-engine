from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.vectortile import (
    clip,
    raw_vertices,
    simplify,
    summary,
    tile_box,
    tiles_at,
    wiggly_line,
)


@pytest.fixture(scope="module")
def lines():
    rng = random.Random(760)
    return [wiggly_line(2000, rng) for _ in range(20)]


class TestTotals:
    @pytest.mark.parametrize(
        ("tolerance", "totals", "largest"),
        [
            (0.0, (40000, 40202, 40822, 43300, 53042, 92268), (40000, 9155, 2049, 466, 72, 22)),
            (0.5, (2467, 6947, 19209, 35421, 49933, 90165), (2467, 1597, 952, 385, 67, 22)),
            (2.0, (640, 2610, 7393, 20883, 43142, 85938), (640, 612, 360, 223, 61, 22)),
        ],
    )
    def test_totals_grow_with_zoom(self, lines, tolerance, totals, largest):
        assert raw_vertices(lines) == 40000
        for zoom, total, big, tiles in zip(
            (0, 2, 4, 6, 8, 10), totals, largest, (1, 14, 77, 483, 3521, 21629), strict=True
        ):
            read = summary(tiles_at(zoom, lines, tolerance))
            assert read["vertices"] == total
            assert read["largest"] == big
            assert read["tiles"] == tiles
        assert totals[-1] > raw_vertices(lines)
        assert totals[0] <= totals[-1]


class TestPieces:
    def test_simplify_and_clip(self, lines):
        line = lines[0]
        assert [len(simplify(line, t)) for t in (0.0, 1e-4, 1e-3, 1e-2)] == [
            2000,
            1022,
            207,
            19,
        ]
        assert simplify(line[:2], 1.0) == line[:2]
        whole = clip(line, (0.0, 0.0, 1.0, 1.0))
        assert len(whole) == 1 and len(whole[0]) == 2000
        mx, my = line[1000]
        box = (mx - 0.05, my - 0.05, mx + 0.05, my + 0.05)
        pieces = clip(line, box)
        assert pieces
        assert all(
            box[0] - 1e-9 <= x <= box[2] + 1e-9 and box[1] - 1e-9 <= y <= box[3] + 1e-9
            for p in pieces
            for x, y in p
        )
        assert tile_box(1, 1, 0) == (0.5, 0.0, 1.0, 0.5)

    def test_refusals(self, lines):
        with pytest.raises(Invalid):
            simplify(lines[0], -1.0)
        with pytest.raises(Invalid):
            tiles_at(-1, lines, 0.5)
        with pytest.raises(Invalid):
            summary({})
