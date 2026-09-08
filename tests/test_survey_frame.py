from __future__ import annotations

import pytest

from atlas.cli import main
from atlas.errors import Invalid
from atlas.surveys.registry import all_surveys, report
from atlas.surveys.survey import Survey


class TestSurvey:
    def test_survey_reads_as_a_checkable_sentence(self):
        told = Survey(
            surveyor="nearcheck",
            finding="the tree's nearest neighbor matched the brute scan",
            readings={"mismatches": 0},
            holds=True,
        )
        assert told.line() == (
            "[HOLDS] nearcheck: the tree's nearest neighbor matched the brute scan"
        )
        assert "mismatches = 0" in told.detail()

    def test_unsigned_readings_are_hearsay(self):
        with pytest.raises(Invalid):
            Survey(surveyor=" ", finding="something")

    def test_a_broken_survey_says_so_first(self):
        told = Survey(surveyor="s", finding="f", holds=False)
        assert told.line().startswith("[BROKEN]")


class TestTheRegistry:
    def test_the_registry_counts_its_own_roster(self):
        surveys = all_surveys()
        assert report().endswith(f"{len(surveys)} surveys, 0 broken")
        assert all(s.holds for s in surveys)


class TestTheCli:
    def test_summary_prints_the_one_line(self, capsys):
        assert main(["summary"]) == 0
        out = capsys.readouterr().out.strip()
        assert out == f"{len(all_surveys())} surveys (0 broken)"

    def test_check_holds_on_an_empty_registry(self, capsys):
        assert main(["check"]) == 0
        assert "all surveys hold" in capsys.readouterr().out

    def test_no_command_prints_help(self, capsys):
        assert main([]) == 2
        assert "surveys" in capsys.readouterr().out
