from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.hexvssquare import (
    closed_form_worst,
    directions_within_two_steps,
    hex_penalty,
    neighbour_distances,
    penalties,
    roundness,
    square_penalty,
)


class TestPenalties:
    def test_the_worst_penalties_land_on_the_closed_forms(self):
        measured = penalties(20000, random.Random(258))
        closed = closed_form_worst()
        assert measured["manhattan"] == pytest.approx((0.2738, 0.4142), abs=1e-3)
        assert measured["octile"] == pytest.approx((0.0542, 0.0824), abs=1e-3)
        assert measured["hex"] == pytest.approx((0.1036, 0.1547), abs=1e-3)
        for key in ("manhattan", "octile", "hex"):
            assert measured[key][1] == pytest.approx(closed[key], abs=1e-4)
            assert measured[key][0] < closed[key]

    def test_the_worst_angles(self):
        assert square_penalty((0, 0), (100, 100), False) == pytest.approx(math.sqrt(2) - 1)
        assert square_penalty((0, 0), (100, 41), True) == pytest.approx(0.0824, abs=1e-4)
        assert hex_penalty((0, 0), (100, 100)) == pytest.approx(2 / math.sqrt(3) - 1)


class TestNeighbourhoods:
    def test_distances_roundness_and_directions(self):
        assert neighbour_distances("square") == pytest.approx([1.0, math.sqrt(2)])
        assert neighbour_distances("hex") == pytest.approx([1.0])
        assert roundness("square") / roundness("hex") == pytest.approx(1.2247, abs=1e-4)
        assert directions_within_two_steps("square") == 16
        assert directions_within_two_steps("hex") == 12


class TestRefusals:
    def test_coincident_cells_bad_kinds_and_empty_samples_are_refused(self):
        with pytest.raises(Invalid):
            square_penalty((0, 0), (0, 0), True)
        with pytest.raises(Invalid):
            neighbour_distances("tri")
        with pytest.raises(Invalid):
            penalties(0, random.Random(0))
