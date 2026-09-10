from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.rasterbuffer import (
    buffer_error,
    disc_mask,
    line_buffer_width,
    line_mask,
    mask_area,
    overshoot_guess,
    raster_buffer,
    ring_area,
)


class TestDiscs:
    @pytest.mark.parametrize(
        ("cell", "buffer", "ring"),
        [
            (4.0, (-8.119, -6.735, -2.011), (-31.049, -15.602, -3.234)),
            (2.0, (-3.026, -0.608, -0.942), (-12.244, -1.779, -1.642)),
            (1.0, (-0.713, -0.667, -0.624), (-2.058, -1.233, -0.846)),
            (0.5, (-0.447, -0.444, -0.404), (-1.177, -0.778, -0.528)),
        ],
    )
    def test_the_buffer_undershoots_at_every_cell(self, cell, buffer, ring):
        size = int(160 / cell) + 1
        for distance, b, r in zip((5.0, 10.0, 20.0), buffer, ring, strict=True):
            read = buffer_error(size, cell, 30.0, distance)
            assert 100 * read["buffer_error"] == pytest.approx(b, abs=1e-3)
            assert 100 * read["ring_error"] == pytest.approx(r, abs=1e-3)
            assert read["buffer_error"] < 0
            assert overshoot_guess(30.0, distance, cell) > 0
            assert abs(read["disc_error"]) < 0.0035

    def test_the_line(self):
        assert line_buffer_width(21, 10, 5.0, 4.0) == 12.0
        assert line_buffer_width(21, 10, 10.0, 4.0) == 20.0
        assert line_buffer_width(81, 40, 5.0, 1.0) == 11.0
        assert line_buffer_width(81, 40, 10.0, 1.0) == 21.0
        assert ring_area(30.0, 5.0) == pytest.approx(3.141592653589793 * (35.0**2 - 30.0**2))
        assert mask_area(raster_buffer(line_mask(5, 2), 0.0, 1.0), 1.0) == 5.0


class TestRefusals:
    def test_bad_grids_distances_and_masks(self):
        with pytest.raises(Invalid):
            disc_mask(2, 1.0, 1.0)
        with pytest.raises(Invalid):
            raster_buffer(line_mask(5, 2), -1.0, 1.0)
        with pytest.raises(Invalid):
            raster_buffer([[False, False]], 1.0, 1.0)
        with pytest.raises(Invalid):
            line_mask(5, 9)
