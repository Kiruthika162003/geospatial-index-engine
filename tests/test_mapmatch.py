from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.mapmatch import RoadNetwork, flips, wrong_fraction

PARALLEL = {"north": [(0, 4), (100, 4)], "south": [(0, 0), (100, 0)]}


def _jittered_drive(seed=161):
    rng = random.Random(seed)
    return [(x, 4 + rng.gauss(0, 2.5)) for x in range(0, 101, 2)]


class TestParallelRoads:
    def test_per_fix_snapping_flips_between_the_roads(self):
        trace = _jittered_drive()
        snapped = RoadNetwork(PARALLEL, set()).nearest_snap(trace)
        truth = ["north"] * len(trace)
        assert wrong_fraction(snapped, truth) == pytest.approx(0.294, abs=0.01)
        assert flips(snapped) == 25

    def test_the_viterbi_match_stays_on_the_driven_road(self):
        trace = _jittered_drive()
        matched = RoadNetwork(PARALLEL, set()).match(trace, sigma=2.5, jump_penalty=20.0)
        assert wrong_fraction(matched, ["north"] * len(trace)) == 0.0
        assert flips(matched) == 0

    def test_a_zero_jump_penalty_degrades_to_per_fix_snapping(self):
        trace = _jittered_drive()
        net = RoadNetwork(PARALLEL, set())
        unpenalized = net.match(trace, sigma=2.5, jump_penalty=0.0)
        assert flips(unpenalized) == flips(net.nearest_snap(trace))


class TestARealTurn:
    def test_a_connected_turn_is_followed_with_one_change_of_road(self):
        rng = random.Random(161)
        roads = {"east": [(0, 0), (50, 0)], "northbound": [(50, 0), (50, 50)]}
        net = RoadNetwork(roads, {("east", "northbound")})
        turn = [(x, rng.gauss(0, 0.5)) for x in range(0, 51, 5)]
        turn += [(50 + rng.gauss(0, 0.5), y) for y in range(5, 51, 5)]
        matched = net.match(turn, sigma=1.0, jump_penalty=20.0)
        assert flips(matched) == 1
        assert wrong_fraction(matched, ["east"] * 11 + ["northbound"] * 10) < 0.1


class TestRefusals:
    def test_no_roads_is_refused(self):
        with pytest.raises(Invalid):
            RoadNetwork({}, set())

    def test_a_one_vertex_road_is_refused(self):
        with pytest.raises(Invalid):
            RoadNetwork({"stub": [(0, 0)]}, set())

    def test_an_empty_trace_is_refused(self):
        with pytest.raises(Invalid):
            RoadNetwork(PARALLEL, set()).match([], 1.0, 1.0)

    def test_bad_parameters_are_refused(self):
        net = RoadNetwork(PARALLEL, set())
        with pytest.raises(Invalid):
            net.match([(0, 0)], sigma=0.0, jump_penalty=1.0)
        with pytest.raises(Invalid):
            net.match([(0, 0)], sigma=1.0, jump_penalty=-1.0)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(Invalid):
            wrong_fraction(["a"], ["a", "b"])
