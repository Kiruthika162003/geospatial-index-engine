from __future__ import annotations

from atlas.surveys import cellwatch, fatwatch, joinwatch
from atlas.surveys.registry import SURVEYS


class TestJoinwatch:
    def test_the_seam_is_not_the_story(self):
        told = joinwatch.run()
        assert told.holds
        assert told.readings["true_pairs"] == 153
        assert told.readings["recall_at_window_8"] == 0.9477
        assert told.readings["seam_share"] == 0.0131
        assert told.readings["invented_pairs"] == 0
        assert told.readings["comparisons"] == 8000
        assert told.readings["missed_share"] > 3 * told.readings["seam_share"]


class TestCellwatch:
    def test_the_rule_leaves_most_cells_empty(self):
        told = cellwatch.run()
        assert told.holds
        assert told.readings["rule_cell"] == 11.18
        assert told.readings["rule_points_per_cell"] == 0.25
        assert told.readings["rule_empty_share"] == 0.7828
        assert told.readings["rule_poisson_law"] == 0.7788
        assert told.readings["double_cell_empty_share"] == 0.0359


class TestFatwatch:
    def test_the_walk_leaves_an_eighth_as_often(self):
        told = fatwatch.run()
        assert told.holds
        assert told.readings["straight_updates"] == 0.126
        assert told.readings["straight_law"] == 0.15
        assert told.readings["walk_updates"] == 0.0159
        assert told.readings["walk_law"] == 0.0225
        assert told.readings["ratio_straight_over_walk"] == 7.95


class TestTheRoster:
    def test_ten_surveys_are_registered(self):
        for name in ("joinwatch", "cellwatch", "fatwatch"):
            assert f"atlas.surveys.{name}" in SURVEYS
        assert len(SURVEYS) == 10
