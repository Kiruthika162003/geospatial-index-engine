"""The worst consecutive jump along the Morton curve against the Hilbert curve.

The drill lays a sixteen by sixteen grid out along the Morton Z-order
and along the Hilbert curve, and measures the worst consecutive
Manhattan jump each ordering makes between neighboring positions in the
sequence. The guess worth recording was that both are space-filling
curves built for locality, so their worst jumps should be comparable.
The measurement separates them starkly: the Hilbert curve's worst
consecutive jump is exactly one, every step lands on a grid-adjacent
cell, while the Morton curve's worst jump is sixteen, the full width of
the grid, taken at the quadrant seams where a high bit flips and the
low bits reset. The survey keeps the both-are-local guess beside the
measured gap, because the difference is structural, not incidental: the
Hilbert curve is constructed by rotating its quadrant sub-curves so
their ends meet, leaving no seam to jump, whereas the Morton curve's
simple bit interleave saves arithmetic at the cost of those seams, so
the sixteen-to-one gap is the price of Morton's simplicity.
"""

from __future__ import annotations

from atlas import hilbert, morton
from atlas.surveys.survey import Survey


def _worst_jump(points: list[tuple[int, int]]) -> int:
    return max(
        abs(points[i][0] - points[i + 1][0]) + abs(points[i][1] - points[i + 1][1])
        for i in range(len(points) - 1)
    )


def run() -> Survey:
    grid = [(x, y) for x in range(16) for y in range(16)]
    z_sorted = sorted(grid, key=lambda p: morton.encode(*p))
    hilbert_sorted = [hilbert.decode(d, 4) for d in range(16 * 16)]
    morton_jump = _worst_jump(z_sorted)
    hilbert_jump = _worst_jump(hilbert_sorted)
    readings = {
        "grid": "16x16",
        "morton_worst_jump": morton_jump,
        "hilbert_worst_jump": hilbert_jump,
    }
    holds = hilbert_jump == 1 and morton_jump > hilbert_jump
    return Survey(
        surveyor="seamwatch",
        finding=(
            "on a 16x16 grid the Hilbert curve's worst consecutive jump was "
            "1 while the Morton curve's was 16, the width of the grid, at the "
            "quadrant seams"
        ),
        readings=readings,
        holds=holds,
    )
