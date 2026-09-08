"""Survey: a surveyor walked the ground, took readings, and reports the numbers.

A survey in this package is a measured drill: it builds a scene
from the real spatial machinery, runs a query or a construction
against it, and reports with numbers rather than adjectives. Each
survey carries the finding in one sentence, the readings that back
it, and a holds flag the registry can gate on, because a survey
whose readings cannot be checked is a traveller's tale, not a
survey. When a drill's first guess was wrong, the docstring of that
survey keeps the wrong guess beside the measured reading; the
correction is the most trustworthy sentence in the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.errors import Invalid


@dataclass(frozen=True)
class Survey:
    surveyor: str
    finding: str
    readings: dict = field(default_factory=dict)
    holds: bool = False

    def __post_init__(self) -> None:
        if not self.surveyor.strip() or not self.finding.strip():
            raise Invalid(
                "a survey needs a surveyor and a finding; "
                "an unsigned reading is hearsay"
            )

    def line(self) -> str:
        state = "HOLDS" if self.holds else "BROKEN"
        return f"[{state}] {self.surveyor}: {self.finding}"

    def detail(self) -> str:
        lines = [self.line()]
        for name in sorted(self.readings):
            lines.append(f"    {name} = {self.readings[name]}")
        return "\n".join(lines)
