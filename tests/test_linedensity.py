from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.linedensity import (
    cell_lengths,
    density,
    edge_ratio,
    expected_density,
    mean,
    one_line,
    random_segments,
    total_length,
)

EXTENT, SIZE = 100.0, 50
CELL = EXTENT / SIZE


class TestOneLine:
    @pytest.mark.parametrize(
        ("radius", "on_line", "three_off"),
        [(0, 0.5, 0.0), (2, 0.1, 0.0), (5, 0.0455, 0.0455), (10, 0.0238, 0.0238)],
    )
    def test_the_line_reads_one_over_the_window(self, radius, on_line, three_off):
        lengths = cell_lengths(one_line(EXTENT), SIZE, CELL)
        assert sum(map(sum, lengths)) == pytest.approx(100.0, abs=1e-9)
        field = density(lengths, CELL, radius)
        assert field[25][25] == pytest.approx(on_line, abs=1e-4)
        assert field[25][25] == pytest.approx(1 / ((2 * radius + 1) * CELL), abs=1e-9)
        assert field[28][25] == pytest.approx(three_off, abs=1e-4)
        assert mean(field) == pytest.approx(
            expected_density(one_line(EXTENT), EXTENT), abs=1e-9
        )


class TestRandomSegments:
    def test_conservation_and_the_field(self):
        segments = random_segments(200, random.Random(850), EXTENT, 20.0)
        lengths = cell_lengths(segments, SIZE, CELL)
        assert sum(map(sum, lengths)) == pytest.approx(total_length(segments), abs=1e-6)
        assert total_length(segments) == pytest.approx(3631.922, abs=1e-3)
        readings = []
        for radius in (0, 2, 5, 20):
            field = density(lengths, CELL, radius)
            values = [v for row in field for v in row]
            zeros = sum(1 for v in values if v == 0) / len(values)
            readings.append(
                (
                    round(mean(field), 4),
                    round(max(values), 3),
                    round(zeros, 4),
                    round(edge_ratio(field), 3),
                )
            )
        assert readings[0] == (0.3632, 2.476, 0.368, 2.348)
        assert readings[1] == (0.3627, 1.121, 0.0096, 0.661)
        assert readings[2] == (0.363, 0.769, 0.0, 0.566)
        assert readings[3] == (0.3746, 0.46, 0.0, 0.784)


class TestRefusals:
    def test_bad_grids_and_radii(self):
        with pytest.raises(Invalid):
            cell_lengths(one_line(EXTENT), 0, 1.0)
        with pytest.raises(Invalid):
            density([[1.0]], 1.0, -1)
