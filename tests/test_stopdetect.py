from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.stopdetect import detect_stops, stop_centers


def _itinerary(seed=165, jitter=0.3):
    rng = random.Random(seed)
    fixes = []
    state = {"x": 0.0, "y": 0.0, "t": 0.0}
    plants = []

    def leg(n):
        for _ in range(n):
            state["x"] += 1.0
            state["t"] += 1.0
            jx, jy = rng.gauss(0, jitter), rng.gauss(0, jitter)
            fixes.append((state["x"] + jx, state["y"] + jy, state["t"]))

    def stop(n):
        for _ in range(n):
            state["t"] += 1.0
            jx, jy = rng.gauss(0, jitter), rng.gauss(0, jitter)
            fixes.append((state["x"] + jx, state["y"] + jy, state["t"]))
        plants.append((state["x"], state["y"], n))

    leg(30)
    stop(20)
    leg(40)
    stop(12)
    leg(25)
    stop(35)
    leg(30)
    return fixes, plants


class TestExactRecovery:
    def test_every_planted_stop_is_found_and_nothing_else(self):
        fixes, plants = _itinerary()
        stops = detect_stops(fixes, radius=1.5, min_duration=8)
        assert stops == [(28, 50), (89, 102), (126, 162)]
        assert len(stops) == len(plants)
        for (cx, cy), (px, py, _) in zip(stop_centers(fixes, stops), plants, strict=True):
            assert abs(cx - px) < 0.2
            assert abs(cy - py) < 0.2


class TestFailureModes:
    def test_a_radius_below_the_jitter_fragments_every_stop(self):
        fixes, _ = _itinerary()
        assert detect_stops(fixes, radius=0.2, min_duration=8) == []

    def test_a_duration_above_the_shortest_stop_drops_exactly_that_stop(self):
        fixes, _ = _itinerary()
        assert len(detect_stops(fixes, radius=1.5, min_duration=15)) == 2

    def test_a_slow_crawl_reads_as_false_stops(self):
        rng = random.Random(166)
        crawl = [(0.05 * i + rng.gauss(0, 0.05), 0.0, float(i)) for i in range(60)]
        assert len(detect_stops(crawl, radius=1.5, min_duration=8)) == 2


class TestRefusals:
    def test_an_empty_trace_is_refused(self):
        with pytest.raises(Invalid):
            detect_stops([], 1.0, 1.0)

    def test_non_positive_thresholds_are_refused(self):
        with pytest.raises(Invalid):
            detect_stops([(0, 0, 0)], radius=0, min_duration=1)
        with pytest.raises(Invalid):
            detect_stops([(0, 0, 0)], radius=1, min_duration=0)
