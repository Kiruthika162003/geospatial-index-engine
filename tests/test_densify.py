from __future__ import annotations

import random

import pytest

from atlas.crosstrack import cross_track_km
from atlas.densify import (
    densify,
    densify_segment,
    hops_needed,
    length_km,
    map_gap_km,
    pieces_for_gap,
    wrap_longitude,
)
from atlas.errors import Invalid
from atlas.haversine import haversine

NEW_YORK = (40.0, -74.0)
LONDON = (51.5, -0.1)
PACIFIC = ((10.0, 170.0), (20.0, -170.0))


class TestTheGapFallsQuadratically:
    def test_new_york_to_london_quarters_per_doubling(self):
        gaps = [map_gap_km(NEW_YORK, LONDON, n) for n in (1, 2, 4, 8, 16, 32, 64)]
        assert gaps[0] == pytest.approx(753.49, abs=0.01)
        assert gaps[-1] == pytest.approx(0.2054, abs=1e-3)
        ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
        assert ratios[0] == pytest.approx(3.675, abs=1e-2)
        assert ratios[-1] == pytest.approx(4.0, abs=1e-2)
        assert ratios == sorted(ratios)

    def test_pieces_for_a_target_gap(self):
        assert pieces_for_gap(NEW_YORK, LONDON, 1.0) == 30
        assert pieces_for_gap(NEW_YORK, LONDON, 0.01) == 291
        assert pieces_for_gap(*PACIFIC, 1.0) == 7
        assert pieces_for_gap(*PACIFIC, 0.01) == 67

    def test_the_pacific_segment_is_drawn_the_short_way_round(self):
        assert map_gap_km(*PACIFIC, 1) == pytest.approx(33.51, abs=0.01)
        assert map_gap_km(*PACIFIC, 64) == pytest.approx(0.0108, abs=1e-3)


class TestIdentities:
    def test_the_dense_line_keeps_the_geodesic_length_and_stays_on_it(self):
        dense = densify_segment(NEW_YORK, LONDON, 100.0)
        assert hops_needed(*NEW_YORK, *LONDON, 100.0) == 57
        assert len(dense) == 58
        assert dense[0] == NEW_YORK
        assert dense[-1] == LONDON
        assert length_km(dense) == pytest.approx(haversine(*NEW_YORK, *LONDON), abs=1e-9)
        assert max(abs(cross_track_km(*NEW_YORK, *LONDON, *p)) for p in dense) < 1e-9

    def test_random_segments_hold_both_identities(self):
        rng = random.Random(176)
        for _ in range(150):
            p = (rng.uniform(-70, 70), rng.uniform(-180, 180))
            q = (rng.uniform(-70, 70), rng.uniform(-180, 180))
            if haversine(*p, *q) > 15000:
                continue
            dense = densify_segment(p, q, 250.0)
            assert abs(length_km(dense) - haversine(*p, *q)) < 1e-8
            assert max(abs(cross_track_km(*p, *q, *x)) for x in dense) < 1e-8

    def test_a_polyline_densifies_hop_by_hop_without_changing_length(self):
        line = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
        dense = densify(line, 200.0)
        assert len(dense) == 13
        assert length_km(dense) == pytest.approx(length_km(line), abs=1e-9)

    def test_a_short_hop_is_left_alone(self):
        assert densify_segment((0, 0), (0, 0.5), 100.0) == [(0, 0), (0, 0.5)]

    def test_wrap_longitude(self):
        assert wrap_longitude(181.0) == -179.0
        assert wrap_longitude(-181.0) == 179.0
        assert wrap_longitude(20.0) == 20.0


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            hops_needed(0, 0, 1, 1, 0)
        with pytest.raises(Invalid):
            densify([(0, 0)], 10.0)
        with pytest.raises(Invalid):
            map_gap_km(NEW_YORK, LONDON, 0)
        with pytest.raises(Invalid):
            pieces_for_gap(NEW_YORK, LONDON, 0)
        with pytest.raises(Invalid):
            pieces_for_gap(NEW_YORK, LONDON, 1e-9, limit=8)
