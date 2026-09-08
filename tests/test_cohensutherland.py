from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.cohensutherland import clip, is_trivial
from atlas.errors import Invalid

BOX = BBox(0, 0, 10, 10)


def _liang(a, b, box):
    x0, y0 = a
    x1, y1 = b
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - box.min_x, box.max_x - x0, y0 - box.min_y, box.max_y - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q, strict=True):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u0 = max(u0, t)
            else:
                u1 = min(u1, t)
    if u0 > u1:
        return None
    return ((x0 + u0 * dx, y0 + u0 * dy), (x1 - (1 - u1) * dx, y1 - (1 - u1) * dy))


class TestClipping:
    def test_inside_stays(self):
        assert clip((2, 2), (8, 8), BOX) == ((2, 2), (8, 8))

    def test_a_crossing_is_clipped_to_the_border(self):
        assert clip((-5, 5), (15, 5), BOX) == ((0, 5.0), (10, 5.0))

    def test_fully_outside_is_rejected(self):
        assert clip((-5, -5), (-1, -1), BOX) is None


class TestAgainstLiangBarsky:
    def test_it_matches_a_parametric_clipper(self):
        rng = random.Random(45)
        trivial = total = 0
        for _ in range(200000):
            a = (rng.uniform(-20, 30), rng.uniform(-20, 30))
            b = (rng.uniform(-20, 30), rng.uniform(-20, 30))
            if a == b:
                continue
            total += 1
            cs = clip(a, b, BOX)
            lb = _liang(a, b, BOX)
            assert (cs is None) == (lb is None)
            if cs is not None:
                for (px, py), (qx, qy) in zip(cs, lb, strict=True):
                    assert abs(px - qx) < 1e-6
                    assert abs(py - qy) < 1e-6
            if is_trivial(a, b, BOX):
                trivial += 1
        # a 10x10 box in a 50x50 scene settles about half the segments for free
        assert 0.45 < trivial / total < 0.60


class TestRefusals:
    def test_a_none_box_is_refused(self):
        with pytest.raises(Invalid):
            clip((0, 0), (1, 1), None)
