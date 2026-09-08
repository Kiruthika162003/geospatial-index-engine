from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.lineofsight import Visibility

VIEWER = (50, 50)


def _walls(rng, count, length=8):
    out = []
    for _ in range(count):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        a = rng.uniform(0, 2 * math.pi)
        out.append(((x, y), (x + length * math.cos(a), y + length * math.sin(a))))
    return out


class TestVerdicts:
    def test_a_target_on_a_wall_is_blocked_and_one_short_of_it_is_seen(self):
        v = Visibility([((0, 5), (10, 5))])
        assert not v.visible((5, 0), (5, 5))
        assert v.visible((5, 0), (5, 4))

    def test_no_walls_means_everything_is_visible(self):
        rng = random.Random(1)
        targets = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(50)]
        assert Visibility([]).visible_fraction(VIEWER, targets) == 1.0


class TestDecay:
    def test_the_visible_fraction_falls_with_obstacle_density(self):
        rng = random.Random(143)
        targets = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(300)]
        fractions = [
            Visibility(_walls(rng, n)).visible_fraction(VIEWER, targets)
            for n in (0, 10, 30, 100)
        ]
        assert fractions[0] == 1.0
        assert fractions == pytest.approx([1.0, 0.840, 0.460, 0.090], abs=0.01)
        assert fractions == sorted(fractions, reverse=True)


class TestBoxFilter:
    def test_it_never_changes_a_verdict_and_prunes_most_exact_tests(self):
        rng = random.Random(144)
        targets = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(300)]
        v = Visibility(_walls(rng, 300))
        with_filter = without = 0
        for t in targets:
            a = v.visible(VIEWER, t, use_box_filter=True)
            with_filter += v.tests
            b = v.visible(VIEWER, t, use_box_filter=False)
            without += v.tests
            assert a == b
        # a blocked target stops at its first crossing, so even the unfiltered
        # scan runs well under the 90000 tests a full pass would take
        assert with_filter < without < 300 * 300
        assert with_filter / without < 0.15  # measured about 8 percent

    def test_shorter_walls_prune_more(self):
        rng = random.Random(145)
        targets = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(100)]
        ratios = []
        for length in (2, 8, 30):
            v = Visibility(_walls(rng, 300, length))
            wf = wo = 0
            for t in targets:
                v.visible(VIEWER, t, True)
                wf += v.tests
                v.visible(VIEWER, t, False)
                wo += v.tests
            ratios.append(wf / wo)
        assert ratios == sorted(ratios)  # measured about 5, 8, 15 percent


class TestRefusals:
    def test_none_obstacles_is_refused(self):
        with pytest.raises(Invalid):
            Visibility(None)

    def test_no_targets_is_refused(self):
        with pytest.raises(Invalid):
            Visibility([]).visible_fraction(VIEWER, [])
