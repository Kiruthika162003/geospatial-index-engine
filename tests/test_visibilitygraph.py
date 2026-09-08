from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from atlas.errors import Invalid
from atlas.visibilitygraph import VisibilityGraph


class TestKnownRoutes:
    def test_one_square_in_the_way_bends_the_path_at_its_corners(self):
        vg = VisibilityGraph([[(4, -1), (6, -1), (6, 1), (4, 1)]])
        path, length = vg.shortest_path((0, 0), (10, 0))
        assert path == [(0, 0), (4, -1), (6, -1), (10, 0)]
        assert length == pytest.approx(10.2462, abs=1e-4)
        assert all(p in vg.corners for p in path[1:-1])

    def test_a_clear_line_is_the_straight_segment(self):
        vg = VisibilityGraph([[(4, -1), (6, -1), (6, 1), (4, 1)]])
        path, length = vg.shortest_path((0, 5), (10, 5))
        assert path == [(0, 5), (10, 5)]
        assert length == pytest.approx(10.0)

    def test_a_walled_off_goal_is_refused(self):
        vg = VisibilityGraph([[(4, 4), (6, 4), (6, 6), (4, 6)]])
        with pytest.raises(Invalid):
            vg.shortest_path((0, 0), (5, 5))


class TestRandomFields:
    def test_clearance_the_straight_line_bound_and_corner_turns(self):
        rng = random.Random(147)
        solved = 0
        excess = []
        for _ in range(60):
            obstacles = []
            for _ in range(rng.randint(2, 8)):
                cx, cy, s = rng.uniform(10, 90), rng.uniform(10, 90), rng.uniform(2, 8)
                obstacles.append(
                    [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
                )
            vg = VisibilityGraph(obstacles)
            start, goal = (0, rng.uniform(0, 100)), (100, rng.uniform(0, 100))
            try:
                path, length = vg.shortest_path(start, goal)
            except Invalid:
                continue
            solved += 1
            for a, b in pairwise(path):
                assert vg.visible(a, b)
            straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
            assert length >= straight - 1e-9
            if vg.visible(start, goal):
                assert length == pytest.approx(straight)
            assert all(p in vg.corners for p in path[1:-1])
            excess.append(length / straight - 1)
        assert solved > 50
        assert sum(excess) / len(excess) < 0.02  # measured 0.33 percent
        assert max(excess) < 0.1  # measured 3.06 percent


class TestRefusals:
    def test_none_obstacles_is_refused(self):
        with pytest.raises(Invalid):
            VisibilityGraph(None)

    def test_an_obstacle_with_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            VisibilityGraph([[(0, 0), (1, 1)]])
