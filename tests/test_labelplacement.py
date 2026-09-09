from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.labelplacement import (
    CollisionIndex,
    brute_collides,
    by_size,
    candidate,
    coverage,
    label_sizes,
    overlaps,
    place,
    placed_fraction,
    uniform_points,
)


def _scene(n):
    rng = random.Random(460)
    return uniform_points(n, rng), label_sizes(n, rng)


class TestDensity:
    @pytest.mark.parametrize(
        ("n", "cover", "one", "four", "eight"),
        [
            (100, 0.0668, 0.81, 0.98, 1.0),
            (300, 0.1988, 0.6767, 0.9267, 0.94),
            (1000, 0.66, 0.362, 0.593, 0.653),
            (2000, 1.3155, 0.196, 0.3305, 0.37),
        ],
    )
    def test_slots_and_coverage(self, n, cover, one, four, eight):
        points, sizes = _scene(n)
        assert coverage(sizes) == pytest.approx(cover, abs=1e-4)
        for slots, expected in ((1, one), (4, four), (8, eight)):
            placed, _ = place(points, sizes, slots)
            assert placed_fraction(placed) == pytest.approx(expected, abs=1e-4)
            assert overlaps(placed) == 0

    @pytest.mark.parametrize(
        ("n", "largest", "smallest", "free"),
        [(300, 0.9267, 0.9167, 0.9), (1000, 0.584, 0.59, 0.603), (2000, 0.306, 0.3305, 0.3815)],
    )
    def test_order_barely_matters_and_covering_points_helps_only_when_crowded(
        self, n, largest, smallest, free
    ):
        points, sizes = _scene(n)
        assert placed_fraction(place(points, sizes, 4, by_size(sizes))[0]) == pytest.approx(
            largest, abs=1e-4
        )
        small_first = place(points, sizes, 4, by_size(sizes, False))[0]
        assert placed_fraction(small_first) == pytest.approx(smallest, abs=1e-4)
        assert placed_fraction(place(points, sizes, 4, avoid_points=False)[0]) == pytest.approx(
            free, abs=1e-4
        )


class TestTheIndex:
    def test_checks_and_agreement_with_brute_force(self):
        rng = random.Random(461)
        points, sizes = uniform_points(2000, rng), label_sizes(2000, rng)
        placed, checks = place(points, sizes, 4)
        assert checks == 92120
        boxes = [b for b in placed if b is not None]
        index = CollisionIndex(80.0)
        for box in boxes:
            index.add(box)
        q = random.Random(462)
        probes = [
            candidate(p, s, (1, 1))
            for p, s in zip(uniform_points(500, q), label_sizes(500, q), strict=True)
        ]
        assert all(index.collides(pr) == brute_collides(pr, boxes) for pr in probes)

    def test_candidates_sit_beside_the_point(self):
        box = candidate((10.0, 10.0), (20.0, 5.0), (1, 1))
        assert (box.min_x, box.min_y, box.max_x, box.max_y) == (11.0, 11.0, 31.0, 16.0)
        box = candidate((10.0, 10.0), (20.0, 5.0), (-1, -1))
        assert (box.max_x, box.max_y) == (9.0, 9.0)
        box = candidate((10.0, 10.0), (20.0, 5.0), (0, 1))
        assert (box.min_x, box.max_x) == (0.0, 20.0)


class TestRefusals:
    def test_bad_sizes_slots_orders_and_cells(self):
        points, sizes = _scene(5)
        with pytest.raises(Invalid):
            place(points, sizes[:3])
        with pytest.raises(Invalid):
            place(points, sizes, 0)
        with pytest.raises(Invalid):
            place(points, sizes, 4, [0, 0, 1, 2, 3])
        with pytest.raises(Invalid):
            CollisionIndex(0)
        with pytest.raises(Invalid):
            placed_fraction([])
