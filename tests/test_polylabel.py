from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.pointinpolygon import ray_casting
from atlas.polylabel import _signed_distance, pole_of_inaccessibility
from atlas.shoelace import centroid

C_SHAPE = [(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (6, 4), (6, 6), (0, 6)]


class TestPole:
    def test_the_square_pole_is_its_center(self):
        sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
        px, py = pole_of_inaccessibility(sq, 0.001)
        assert px == pytest.approx(5.0, abs=0.01)
        assert py == pytest.approx(5.0, abs=0.01)
        assert _signed_distance(px, py, sq) == pytest.approx(5.0, abs=0.01)

    def test_the_pole_is_strictly_inside_and_clear_of_edges(self):
        pole = pole_of_inaccessibility(C_SHAPE, 0.01)
        assert ray_casting(pole, C_SHAPE)
        assert _signed_distance(*pole, C_SHAPE) > 0.5


class TestContrastWithCentroid:
    def test_the_centroid_escapes_the_notch_but_the_pole_does_not(self):
        cen = centroid(C_SHAPE)
        pole = pole_of_inaccessibility(C_SHAPE, 0.01)
        assert ray_casting(cen, C_SHAPE) is False  # centroid lands in the hollow
        assert ray_casting(pole, C_SHAPE) is True  # pole stays in the solid part


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            pole_of_inaccessibility([(0, 0), (1, 1)])

    def test_a_non_positive_precision_is_refused(self):
        with pytest.raises(Invalid):
            pole_of_inaccessibility(C_SHAPE, 0)
