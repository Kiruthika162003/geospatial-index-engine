from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.pipindex import (
    PolygonGrid,
    boundary_fraction_law,
    circle,
    query_batch,
    random_points,
    ray_cast,
    star,
)


@pytest.fixture(scope="module")
def points():
    return random_points(20000, random.Random(440))


class TestTheCircle:
    @pytest.mark.parametrize(
        ("cells", "crossed", "rays", "law"),
        [(4, 12, 12408, 0.7854), (16, 60, 3890, 0.1963), (64, 252, 1030, 0.0491)],
    )
    def test_crossed_cells_and_rays_cast(self, points, cells, crossed, rays, law):
        grid = PolygonGrid(circle(64), cells)
        inside, outside, boundary = grid.counts()
        assert boundary == crossed
        assert inside + outside + boundary == cells * cells
        read = boundary_fraction_law(cells, 2 * math.pi * 100, 200.0)
        assert read == pytest.approx(law, abs=1e-4)
        assert (boundary / cells**2 > law) == (cells >= 8)
        found, cast = query_batch(grid, points)
        assert (found, cast) == (12996, rays)
        if cells >= 8:
            assert abs(cast / 20000 - law) < 0.03

    def test_every_answer_matches_the_ray_cast(self, points):
        for ring, total in ((circle(64), 12996), (circle(512), 13013), (star(8), 5126)):
            grid = PolygonGrid(ring, 16)
            assert query_batch(grid, points)[0] == total
            assert all(grid.contains(x, y)[0] == ray_cast(ring, x, y) for x, y in points[:3000])
            assert grid.contains(500.0, 500.0) == (False, False)


class TestTheStar:
    @pytest.mark.parametrize(
        ("cells", "fraction", "rays"), [(4, 1.0, 16588), (8, 0.625, 10301), (64, 0.1035, 1679)]
    )
    def test_a_star_is_crossed_everywhere_at_four_cells(self, points, cells, fraction, rays):
        grid = PolygonGrid(star(8), cells)
        assert grid.counts()[2] / cells**2 == pytest.approx(fraction, abs=1e-4)
        assert query_batch(grid, points)[1] == rays


class TestRefusals:
    def test_bad_rings_grids_and_batches(self):
        with pytest.raises(Invalid):
            PolygonGrid([(0, 0), (1, 1)], 4)
        with pytest.raises(Invalid):
            PolygonGrid(circle(8), 0)
        with pytest.raises(Invalid):
            query_batch(PolygonGrid(circle(8), 4), [])
