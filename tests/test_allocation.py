from __future__ import annotations

import random

import pytest

from atlas.allocation import (
    clustered,
    displaced,
    fair_capacities,
    greedy,
    improve,
    loads,
    mean_distance,
    nearest,
    uniform,
)
from atlas.errors import Invalid

FACILITIES = uniform(8, random.Random(491))


@pytest.fixture(scope="module")
def demand():
    return {
        "uniform": uniform(2000, random.Random(492)),
        "clustered": clustered(2000, random.Random(493)),
    }


class TestFreeAllocation:
    def test_nearest_loads_are_lopsided(self, demand):
        first = nearest(demand["uniform"], FACILITIES)
        assert loads(first, 8) == [531, 529, 191, 123, 161, 41, 36, 388]
        assert mean_distance(demand["uniform"], FACILITIES, first) == pytest.approx(
            22.552, abs=1e-3
        )
        first = nearest(demand["clustered"], FACILITIES)
        assert max(loads(first, 8)) == 872
        assert mean_distance(demand["clustered"], FACILITIES, first) == pytest.approx(
            21.068, abs=1e-3
        )


class TestCapacity:
    @pytest.mark.parametrize(
        ("slack", "regret", "near", "order", "moved"),
        [
            (1.0, 36.213, 37.121, 37.881, 946),
            (1.25, 29.786, 31.197, 31.228, 630),
            (2.0, 22.596, 22.747, 22.853, 62),
        ],
    )
    def test_uniform_demand_favours_regret_first(
        self, demand, slack, regret, near, order, moved
    ):
        points = demand["uniform"]
        caps = fair_capacities(2000, 8, slack)
        readings = {
            p: greedy(points, FACILITIES, caps, p) for p in ("regret", "nearest", "input")
        }
        assert mean_distance(points, FACILITIES, readings["regret"]) == pytest.approx(
            regret, abs=1e-3
        )
        assert mean_distance(points, FACILITIES, readings["nearest"]) == pytest.approx(
            near, abs=1e-3
        )
        assert mean_distance(points, FACILITIES, readings["input"]) == pytest.approx(
            order, abs=1e-3
        )
        assert displaced(points, FACILITIES, readings["regret"]) == moved
        assert max(loads(readings["regret"], 8)) <= caps[0]

    @pytest.mark.parametrize(
        ("slack", "regret", "near"),
        [(1.0, 40.879, 41.28), (1.5, 32.078, 30.734), (2.0, 26.687, 26.074)],
    )
    def test_clustered_demand_turns_the_tables_from_slack_1_25(
        self, demand, slack, regret, near
    ):
        points = demand["clustered"]
        caps = fair_capacities(2000, 8, slack)
        assert mean_distance(
            points, FACILITIES, greedy(points, FACILITIES, caps)
        ) == pytest.approx(regret, abs=1e-3)
        assert mean_distance(
            points, FACILITIES, greedy(points, FACILITIES, caps, "nearest")
        ) == pytest.approx(near, abs=1e-3)
        assert (near < regret) == (slack >= 1.25)


class TestSwaps:
    @pytest.mark.parametrize(
        ("slack", "before", "after", "swaps"),
        [(1.0, 34.901, 30.58, 422), (1.5, 24.845, 23.238, 156)],
    )
    def test_swaps_lower_the_trip_and_keep_every_load(self, slack, before, after, swaps):
        points = uniform(500, random.Random(494))
        caps = fair_capacities(500, 8, slack)
        first = greedy(points, FACILITIES, caps)
        assert mean_distance(points, FACILITIES, first) == pytest.approx(before, abs=1e-3)
        better, count = improve(points, FACILITIES, first, 3)
        assert mean_distance(points, FACILITIES, better) == pytest.approx(after, abs=1e-3)
        assert count == swaps
        assert loads(better, 8) == loads(first, 8)


class TestRefusals:
    def test_bad_capacities_priorities_and_shapes(self):
        points = uniform(10, random.Random(1))
        with pytest.raises(Invalid):
            nearest(points, [])
        with pytest.raises(Invalid):
            greedy(points, FACILITIES, [1] * 7)
        with pytest.raises(Invalid):
            greedy(points, FACILITIES, [1] * 8)
        with pytest.raises(Invalid):
            greedy(points, FACILITIES, [2] * 8, "random")
        with pytest.raises(Invalid):
            mean_distance(points, FACILITIES, [0] * 9)
        with pytest.raises(Invalid):
            fair_capacities(10, 2, 0.5)
