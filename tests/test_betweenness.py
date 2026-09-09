from __future__ import annotations

import pytest

from atlas.betweenness import (
    brandes,
    bridge_ends,
    bridged_town,
    crossing_pairs,
    grid,
    normalised,
    summary,
    top,
    town,
)
from atlas.errors import Invalid


class TestTowns:
    @pytest.mark.parametrize(
        ("n", "busiest", "mean", "idle"),
        [(100, 0.167, 0.0433, 0.07), (300, 0.1059, 0.026, 0.0367)],
    )
    def test_the_busiest_node_and_the_idle_share(self, n, busiest, mean, idle):
        _, graph = town(n, 520)
        score = normalised(brandes(graph))
        read = summary(score)
        assert read["max"] == pytest.approx(busiest, abs=1e-4)
        assert read["mean"] == pytest.approx(mean, abs=1e-4)
        assert read["zero_fraction"] == pytest.approx(idle, abs=1e-4)
        assert top(score, 1)[0][1] == pytest.approx(busiest, abs=1e-4)

    @pytest.mark.parametrize(
        ("bridge_y", "ends", "approach"),
        [(50.0, (0.5027, 0.5027), 0.2226), (10.0, (0.5044, 0.5014), 0.3993)],
    )
    def test_the_bridge_ends_carry_the_crossing_share(self, bridge_y, ends, approach):
        points, graph = bridged_town(300, 520, bridge_y)
        score = normalised(brandes(graph))
        a, b = bridge_ends(points, graph)
        assert (score[a], score[b]) == pytest.approx(ends, abs=1e-4)
        assert crossing_pairs(points) / (299 * 298 / 2) == pytest.approx(0.5048, abs=1e-4)
        assert top(score, 3)[2][1] == pytest.approx(approach, abs=1e-4)


class TestGrids:
    @pytest.mark.parametrize(
        ("side", "centre", "corner", "mean"),
        [(5, 0.2382, 0.0097, 0.1014), (7, 0.1777, 0.0032, 0.078), (11, 0.1192, 0.0007, 0.0532)],
    )
    def test_the_centre_carries_over_one_over_the_side(self, side, centre, corner, mean):
        _, graph = grid(side)
        score = normalised(brandes(graph))
        middle = (side // 2) * side + side // 2
        assert score[middle] == pytest.approx(centre, abs=1e-4)
        assert score[middle] == max(score.values())
        assert score[0] == pytest.approx(corner, abs=1e-4)
        read = summary(score)
        assert read["mean"] == pytest.approx(mean, abs=1e-4)
        assert read["zero_fraction"] == 0.0
        assert 1.15 < centre * side < 1.4


class TestRefusals:
    def test_empty_graphs_and_tiny_normalisations(self):
        with pytest.raises(Invalid):
            brandes({})
        with pytest.raises(Invalid):
            normalised({0: 0.0, 1: 0.0})
        with pytest.raises(Invalid):
            bridge_ends([(10.0, 0.0), (20.0, 0.0)], {0: {1: 10.0}, 1: {0: 10.0}})
        score = brandes({0: {1: 1.0}, 1: {0: 1.0, 2: 1.0}, 2: {1: 1.0}})
        assert score == {0: 0.0, 1: 1.0, 2: 0.0}
