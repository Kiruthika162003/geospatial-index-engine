"""The half-root cell rule read for empty cells on 2000 uniform points against the Poisson law.

The drill drops 2000 uniform points on a 1000 square, takes the
common rule for a raster cell, half the root of the area per point,
and counts the cells left empty at that cell and at twice it. The
guess before measuring was that the rule leaves a couple of percent
of cells empty, the sort of loss a raster can carry. The measurement
refutes that: the rule's cell of 11.18 holds a quarter of a point on
average and leaves 78.3 percent of the cells empty, within half a
percent of the Poisson law e to the minus a quarter, and even twice
the cell leaves 3.6 percent empty against the law's 1.8. The survey
keeps the couple-of-percent guess beside the 78.
"""

from __future__ import annotations

import random

from atlas.cellsizechoice import poisson_empty_law, sweep, uniform
from atlas.surveys.survey import Survey


def run() -> Survey:
    points = uniform(2000, random.Random(860), 1000.0)
    read = sweep(points, 1000.0, [0.5, 2.0])
    readings = {
        "rule_cell": round(read[0.5]["cell"], 2),
        "rule_points_per_cell": round(read[0.5]["per_cell"], 3),
        "rule_empty_share": round(read[0.5]["empty"], 4),
        "rule_poisson_law": round(poisson_empty_law(read[0.5]["per_cell"]), 4),
        "double_cell_empty_share": round(read[2.0]["empty"], 4),
        "double_cell_law": round(read[2.0]["law"], 4),
    }
    holds = (
        abs(read[0.5]["empty"] - read[0.5]["law"]) < 0.005
        and read[0.5]["empty"] > 0.75
        and read[2.0]["empty"] < 0.05
    )
    return Survey(
        surveyor="cellwatch",
        finding=(
            "the half-root cell rule left 78.3 percent of cells empty on 2000 uniform "
            "points, within half a percent of the Poisson law, and twice the cell still "
            "left 3.6 percent"
        ),
        readings=readings,
        holds=holds,
    )
