from __future__ import annotations

import math
import random

import pytest

from atlas.douglaspeucker import max_error, simplify
from atlas.errors import Invalid


class TestSimplify:
    def test_a_straight_line_collapses_to_its_endpoints(self):
        line = [(i, 0) for i in range(100)]
        assert simplify(line, 0.5) == [(0, 0), (99, 0)]

    def test_a_spike_is_preserved(self):
        spike = [(0, 0), (1, 0), (2, 0), (3, 5), (4, 0), (5, 0)]
        assert simplify(spike, 1.0) == [(0, 0), (2, 0), (3, 5), (5, 0)]

    def test_short_inputs_pass_through(self):
        assert simplify([], 1) == []
        assert simplify([(1, 1)], 1) == [(1, 1)]
        assert simplify([(1, 1), (2, 2)], 1) == [(1, 1), (2, 2)]

    def test_larger_tolerance_keeps_fewer_points(self):
        rng = random.Random(29)
        path = [(i * 0.1, math.sin(i * 0.1) + rng.uniform(-0.05, 0.05)) for i in range(500)]
        counts = [len(simplify(path, t)) for t in (0.01, 0.05, 0.1, 0.5)]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] < 30  # heavy tolerance leaves a sparse line


class TestErrorBound:
    def test_no_dropped_point_exceeds_the_tolerance(self):
        rng = random.Random(29)
        for _ in range(2000):
            p = [(i, rng.uniform(0, 10)) for i in range(rng.randint(3, 40))]
            tol = rng.uniform(0, 3)
            assert max_error(p, tol) <= tol + 1e-9

    def test_the_bound_is_tight_near_the_tolerance(self):
        rng = random.Random(29)
        path = [(i * 0.1, math.sin(i * 0.1) + rng.uniform(-0.05, 0.05)) for i in range(500)]
        # at tol 0.1 the worst dropped point sits just under the ceiling
        assert 0.05 < max_error(path, 0.1) <= 0.1 + 1e-9


class TestRefusals:
    def test_none_points_is_refused(self):
        with pytest.raises(Invalid):
            simplify(None, 1)

    def test_a_negative_tolerance_is_refused(self):
        with pytest.raises(Invalid):
            simplify([(0, 0), (1, 1), (2, 0)], -1)
