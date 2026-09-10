from __future__ import annotations

from atlas.surveys import gridturn, ringwatch, tiltwatch
from atlas.surveys.registry import SURVEYS


class TestTiltwatch:
    def test_the_south_slope_triples_winter_and_the_north_slope_is_dark(self):
        told = tiltwatch.run()
        assert told.holds
        assert told.readings["winter_south_over_flat"] == 3.073
        assert told.readings["summer_south_over_flat"] == 0.796
        assert told.readings["winter_north_30_mj"] == 0.0
        assert told.readings["winter_noon_elevation"] == 21.56
        assert told.readings["summer_north_30_mj"] == 37.06
        assert told.line().startswith("[HOLDS] tiltwatch:")


class TestGridturn:
    def test_turns_inside_a_slice_keep_the_order_at_one(self):
        told = gridturn.run()
        assert told.holds
        assert told.readings["grid_order_at_0_3_17"] == [1.0, 1.0, 1.0]
        assert told.readings["web_order"] == 0.0239
        assert told.readings["web_entropy"] == 3.5571
        assert "0.0239" in told.finding


class TestRingwatch:
    def test_the_first_ring_rule_fails_on_a_third_of_queries(self):
        told = ringwatch.run()
        assert told.holds
        assert told.readings["naive_wrong_share"] == 0.311
        assert told.readings["safe_wrong_share"] == 0.0
        assert told.readings["safe_rings"] == 1.0
        assert told.readings["naive_rings"] == 0.034
        assert "31 percent" in told.finding


class TestTheRoster:
    def test_the_surveys_are_registered(self):
        assert "atlas.surveys.tiltwatch" in SURVEYS
        assert "atlas.surveys.gridturn" in SURVEYS
        assert "atlas.surveys.ringwatch" in SURVEYS
        assert len(SURVEYS) == 10
