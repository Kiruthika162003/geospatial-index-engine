from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.kdinsertion import (
    InsertionTree,
    balanced_height,
    brute_nearest,
    build,
    mean_visits,
    sorted_by_x,
    sorted_diagonal,
    sorted_snake,
    uniform,
)


@pytest.fixture(scope="module")
def clouds():
    rng = random.Random(370)
    return {n: uniform(n, rng) for n in (1000, 4000)}


QUERIES = uniform(200, random.Random(371))


class TestOrders:
    @pytest.mark.parametrize(
        ("n", "shuffled", "by_x", "diagonal", "snake"),
        [
            (1000, (21, 11.99, 20.6), (39, 20.54, 41.5), (79, 34.48, 48.2), (40, 21.65, 39.1)),
            (4000, (31, 14.14, 23.0), (52, 26.12, 70.1), (159, 59.69, 87.3), (59, 31.04, 62.0)),
        ],
    )
    def test_heights_depths_and_visits(self, clouds, n, shuffled, by_x, diagonal, snake):
        points = clouds[n]
        orders = {
            "shuffled": points,
            "by_x": sorted_by_x(points),
            "diagonal": sorted_diagonal(points),
            "snake": sorted_snake(points, 16),
        }
        expected = {"shuffled": shuffled, "by_x": by_x, "diagonal": diagonal, "snake": snake}
        for name, order in orders.items():
            tree = build(order)
            assert tree.size == n
            visits = round(mean_visits(tree, QUERIES), 1)
            read = (tree.height(), round(tree.mean_depth(), 2), visits)
            assert read == expected[name]
        assert balanced_height(n) == {1000: 10, 4000: 12}[n]
        assert expected["by_x"][0] < 2 * expected["shuffled"][0]
        assert expected["diagonal"][0] > 2 * expected["by_x"][0]


class TestRebuild:
    def test_median_splits_restore_the_balanced_height(self):
        points = uniform(4000, random.Random(372))
        tree = build(sorted_by_x(points))
        assert tree.height() == 50
        tree.rebuild()
        assert tree.height() == 12
        assert tree.mean_depth() == pytest.approx(10.98, abs=0.01)
        queries = uniform(200, random.Random(373))
        assert mean_visits(tree, queries) == pytest.approx(18.1, abs=0.05)
        assert all(tree.nearest(q)[0] == brute_nearest(points, q) for q in queries)

    def test_the_insertion_tree_finds_the_true_nearest(self):
        points = uniform(500, random.Random(374))
        tree = build(sorted_diagonal(points))
        for q in uniform(100, random.Random(375)):
            assert tree.nearest(q)[0] == brute_nearest(points, q)


class TestRefusals:
    def test_empty_trees_and_bad_arguments(self):
        tree = InsertionTree()
        assert tree.height() == 0
        with pytest.raises(Invalid):
            tree.mean_depth()
        with pytest.raises(Invalid):
            tree.nearest((0.0, 0.0))
        with pytest.raises(Invalid):
            balanced_height(-1)
        with pytest.raises(Invalid):
            sorted_snake([(0.0, 0.0)], 0)
        with pytest.raises(Invalid):
            mean_visits(build([(0.0, 0.0)]), [])
        assert tree.insert((0.5, 0.5)) == 1
        assert tree.insert((0.2, 0.9)) == 2
