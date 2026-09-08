from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.segmentsweep import brute, sweep


def _field(rng, count, length):
    segs = []
    for _ in range(count):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        a = rng.uniform(0, 2 * math.pi)
        segs.append(((x, y), (x + length * math.cos(a), y + length * math.sin(a))))
    return segs


class TestLossless:
    def test_the_sweep_finds_exactly_the_brute_crossings(self):
        rng = random.Random(137)
        for length in (2, 5, 10, 25, 100):
            for _ in range(20):
                segs = _field(rng, 80, length)
                assert sweep(segs).crossings == brute(segs).crossings

    def test_a_known_crossing_pair(self):
        segs = [((0, 0), (4, 4)), ((0, 4), (4, 0)), ((10, 10), (11, 11))]
        assert sweep(segs).crossings == {(0, 1)}


class TestTheFilter:
    def test_pairs_tested_rise_with_segment_length(self):
        rng = random.Random(138)
        fractions = []
        for length in (2, 10, 100):
            tested = total = 0
            for _ in range(20):
                segs = _field(rng, 80, length)
                tested += sweep(segs).pairs_tested
                total += brute(segs).pairs_tested
            fractions.append(tested / total)
        # measured about 2.4%, 12%, 62% for lengths 2, 10, 100 across a 100-wide field
        assert fractions[0] < 0.05
        assert fractions[2] > 0.5
        assert fractions == sorted(fractions)

    def test_full_width_segments_degrade_to_brute_force(self):
        rng = random.Random(139)
        segs = [((0, rng.uniform(0, 100)), (100, rng.uniform(0, 100))) for _ in range(60)]
        s, b = sweep(segs), brute(segs)
        assert s.pairs_tested == b.pairs_tested == 60 * 59 // 2
        assert s.crossings == b.crossings


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            sweep(None)
        with pytest.raises(Invalid):
            brute(None)
