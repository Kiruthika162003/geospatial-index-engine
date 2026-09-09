from __future__ import annotations

import math

import pytest

from atlas.errors import Invalid
from atlas.hillshade import (
    cone,
    contrast_law,
    flank_cell,
    flank_contrast,
    flat_shade,
    horn_gradient,
    shade,
    shade_cell,
)

STEEP = cone(41, 20.0, 18.0)  # slope 48 degrees
GENTLE = cone(41, 6.552, 18.0)  # slope 20 degrees


class TestFlatGround:
    def test_flat_ground_takes_the_sine_of_the_altitude(self):
        flat = [[5.0] * 9 for _ in range(9)]
        assert shade_cell(flat, 4, 4, 1.0, 315, 45) == pytest.approx(math.sqrt(0.5))
        assert flat_shade(45) == pytest.approx(math.sqrt(0.5))
        assert flat_shade(90) == 1.0


class TestTheCone:
    def test_horns_gradient_reads_the_flank_slope(self):
        assert horn_gradient(GENTLE, 20, 30, 1.0)[0] == pytest.approx(-0.3631, abs=1e-3)
        assert horn_gradient(GENTLE, 20, 30, 1.0)[1] == 0.0
        assert horn_gradient(STEEP, 20, 30, 1.0)[0] == pytest.approx(-1.1083, abs=1e-3)

    def test_the_flank_follows_the_cosine_law(self):
        shaded = shade(GENTLE, 1.0, 315.0, 45.0)
        slope, alt = math.atan(6.552 / 18), math.radians(45)
        worst = 0.0
        for bearing in range(0, 360, 15):
            r, c = flank_cell(41, 10, bearing)
            cos_term = math.cos(math.radians(315 - bearing))
            oblique = math.cos(alt) * math.sin(slope) * cos_term
            predicted = math.sin(alt) * math.cos(slope) + oblique
            worst = max(worst, abs(shaded[r][c] - predicted))
        assert worst < 0.007

    def test_the_faces_toward_and_away_sum_to_twice_the_flat_shade_times_cos_slope(self):
        shaded = shade(GENTLE, 1.0, 315.0, 45.0)
        nw, se = flank_cell(41, 10, 315), flank_cell(41, 10, 135)
        total = shaded[nw[0]][nw[1]] + shaded[se[0]][se[1]]
        assert total == pytest.approx(1.3291, abs=1e-3)
        law = 2 * math.sqrt(0.5) * math.cos(math.atan(6.552 / 18))
        assert total == pytest.approx(law, abs=1e-3)

    def test_a_slope_steeper_than_the_sun_is_black_on_the_far_face(self):
        shaded = shade(STEEP, 1.0, 315.0, 45.0)
        nw, se = flank_cell(41, 10, 315), flank_cell(41, 10, 135)
        assert shaded[nw[0]][nw[1]] == pytest.approx(0.9987, abs=1e-3)
        assert shaded[se[0]][se[1]] == 0.0
        assert flank_contrast(STEEP, 1.0, 315, 45, 10) == math.inf
        assert contrast_law(48, 45) == math.inf


class TestContrast:
    @pytest.mark.parametrize(
        ("altitude", "expected"), [(30, 4.40), (45, 2.14), (60, 1.53), (75, 1.22), (89, 1.013)]
    )
    def test_the_grid_reproduces_the_closed_form(self, altitude, expected):
        measured = flank_contrast(GENTLE, 1.0, 315.0, altitude, 10)
        assert measured == pytest.approx(expected, abs=0.01)
        assert measured / contrast_law(20, altitude) == pytest.approx(1.0, abs=3e-3)

    def test_a_sun_below_the_slope_has_infinite_contrast(self):
        assert flank_contrast(GENTLE, 1.0, 315.0, 15, 10) == math.inf
        assert contrast_law(20, 15) == math.inf


class TestTheMirror:
    def test_a_south_sun_is_the_point_mirror_of_a_north_sun(self):
        north = shade(STEEP, 1.0, 315.0, 45.0)
        south = shade(STEEP, 1.0, 135.0, 45.0)
        for r in range(41):
            for c in range(41):
                assert north[r][c] == pytest.approx(south[40 - r][40 - c], abs=1e-12)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        flat = [[1.0]]
        with pytest.raises(Invalid):
            shade_cell(flat, 0, 0, 1.0, 315, 91)
        with pytest.raises(Invalid):
            shade_cell(flat, 0, 0, 0.0, 315, 45)
        with pytest.raises(Invalid):
            shade([], 1.0)
