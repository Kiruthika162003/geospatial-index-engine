"""Every survey, one call, one page."""

from __future__ import annotations

import importlib

from atlas.surveys.survey import Survey

SURVEYS: tuple[str, ...] = (
    "atlas.surveys.nearcheck",
    "atlas.surveys.seamwatch",
    "atlas.surveys.arcwatch",
    "atlas.surveys.rulerwalk",
)


def all_surveys() -> list[Survey]:
    surveys = []
    for dotted in SURVEYS:
        module = importlib.import_module(dotted)
        surveys.append(module.run())
    return surveys


def broken() -> list[str]:
    return [survey.surveyor for survey in all_surveys() if not survey.holds]


def report() -> str:
    surveys = all_surveys()
    lines = [survey.line() for survey in surveys]
    failing = sum(1 for survey in surveys if not survey.holds)
    lines.append("")
    lines.append(f"{len(surveys)} surveys, {failing} broken")
    return "\n".join(lines)
