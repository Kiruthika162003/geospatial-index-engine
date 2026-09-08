from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.shoelace import area
from atlas.sutherlandhodgman import clip

WINDOW = [(0, 0), (1, 0), (1, 1), (0, 1)]  # CCW unit square


class TestClipping:
    def test_a_big_triangle_fills_the_window(self):
        clipped = clip([(-1, -1), (3, 0), (0, 3)], WINDOW)
        assert area(clipped) == pytest.approx(1.0)

    def test_a_contained_polygon_is_unchanged_in_area(self):
        small = [(0.2, 0.2), (0.8, 0.3), (0.5, 0.7)]
        assert area(clip(small, WINDOW)) == pytest.approx(area(small))

    def test_a_disjoint_polygon_clips_to_nothing(self):
        far = [(5, 5), (6, 5), (6, 6)]
        assert clip(far, WINDOW) == []


class TestMonotoneRestriction:
    def test_clipping_never_enlarges_the_polygon(self):
        rng = random.Random(43)
        for _ in range(20000):
            k = rng.randint(3, 8)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            cx, cy, r = rng.uniform(-1, 2), rng.uniform(-1, 2), rng.uniform(0.2, 3)
            subject = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            clipped = clip(subject, WINDOW)
            clipped_area = area(clipped) if len(clipped) >= 3 else 0.0
            assert clipped_area <= area(subject) + 1e-9
            assert clipped_area <= 1.0 + 1e-9  # never exceeds the window

    def test_a_window_containing_the_subject_preserves_it(self):
        small = [(0.2, 0.2), (0.8, 0.3), (0.5, 0.7)]
        big = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
        assert area(clip(small, big)) == pytest.approx(area(small))


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            clip(None, WINDOW)

    def test_too_few_window_vertices_is_refused(self):
        with pytest.raises(Invalid):
            clip([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 1)])
