from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.hypsometry import (
    bowl,
    cone,
    curve,
    disc_mask,
    dome,
    integral,
    masked,
    plane,
    quantiles,
)


class TestKnownShapes:
    def test_a_plane_reads_one_half_exactly(self):
        assert integral(plane(101)) == pytest.approx(0.5)

    def test_a_cone_reads_a_third_over_its_disc_and_less_over_the_square(self):
        c = cone(321)
        assert integral(masked(c, disc_mask(321))) == pytest.approx(1 / 3, abs=1e-3)
        assert integral(c) == pytest.approx(0.2618, abs=1e-3)

    def test_a_dome_and_a_bowl_both_read_one_half_and_sum_to_one(self):
        mask = disc_mask(321)
        d = integral(masked(dome(321), mask))
        b = integral(masked(bowl(321), mask))
        assert d == pytest.approx(0.5, abs=1e-3)
        assert b == pytest.approx(0.5, abs=1e-3)
        assert d + b == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("size", "cone_reading", "dome_reading"),
        [
            (11, 0.3135, 0.4696),
            (21, 0.3272, 0.4908),
            (41, 0.3318, 0.4977),
            (161, 0.3334, 0.5001),
        ],
    )
    def test_the_reading_converges_with_the_cell(self, size, cone_reading, dome_reading):
        mask = disc_mask(size)
        assert integral(masked(cone(size), mask)) == pytest.approx(cone_reading, abs=1e-3)
        assert integral(masked(dome(size), mask)) == pytest.approx(dome_reading, abs=1e-3)


class TestTheCurve:
    def test_the_cones_curve_and_quantiles_follow_their_closed_forms(self):
        c = masked(cone(321), disc_mask(321))
        readings = dict(curve(c, 4))
        for h in (0.25, 0.5, 0.75):
            assert readings[h] == pytest.approx((1 - h) ** 2, abs=1e-3)
        assert readings[0.0] == 1.0
        assert readings[1.0] < 1e-4  # the apex cell alone sits at the peak
        found = quantiles(c, (0.25, 0.5, 0.75))
        assert found == pytest.approx([1 - (1 - f) ** 0.5 for f in (0.25, 0.5, 0.75)], abs=1e-3)
        assert found[1] == pytest.approx(0.2929, abs=1e-3)


class TestRefusals:
    def test_empty_flat_and_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            integral([])
        with pytest.raises(Invalid):
            integral([[1.0, 1.0]])
        with pytest.raises(Invalid):
            curve(cone(5), 0)
        with pytest.raises(Invalid):
            quantiles(cone(5), (1.5,))
