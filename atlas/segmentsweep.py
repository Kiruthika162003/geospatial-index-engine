"""Segment sweep: find every crossing among many segments by testing only x-overlapping pairs.

Finding all the crossings among a pile of segments, road links,
property lines, contour lines, is the pairwise test applied to every
pair, quadratic in the segment count and mostly wasted, because most
pairs are nowhere near each other. A sweep in x cuts the waste. Sort
every segment's left and right x endpoints as events; walk them in
order keeping an active set of the segments whose x extent currently
contains the sweep line; when a segment enters, test it against only
the active segments, since a segment that has already exited cannot
overlap it in x and a segment not yet entered will test it when it
enters. Two segments are tested exactly once, when the later of them
enters, and only if their x extents overlap, so the pair count drops
from all pairs to the pairs that overlap in x, which for short
segments spread over a wide area is a small fraction. This is the
first half of Bentley-Ottmann, without the second half's y-ordering
that brings the bound down to the crossings themselves, and it is
simpler, exact, and already a large saving. The survey measures three
things. The crossings found are exactly the crossings a brute all-
pairs test finds, since the x-overlap filter never excludes a pair
that could cross, two segments crossing must overlap in x. The pairs
actually tested are a fraction of all pairs that falls as the segments
get shorter relative to the spread, measured across several lengths.
And when every segment spans the whole width the filter excludes
nothing and the sweep degrades to the brute count, which is its honest
worst case. The finding worth stating is that x-overlap is a lossless
filter, the sweep's crossings match brute force exactly, and its cost
tracks the x-overlapping pairs rather than all pairs, collapsing to
brute force only when everything overlaps. This module sweeps for all
crossings and counts the pairs tested, and a survey confirms the match
and measures the fraction against segment length.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.segmentintersect import segments_intersect

Point = tuple[float, float]
Segment = tuple[Point, Point]


class SweepResult:
    def __init__(self, crossings: set[tuple[int, int]], pairs_tested: int) -> None:
        self.crossings = crossings
        self.pairs_tested = pairs_tested


def sweep(segments: list[Segment]) -> SweepResult:
    if segments is None:
        raise Invalid("segments must not be None")
    events: list[tuple[float, int, int]] = []  # (x, kind, index) with kind 0 enter, 1 exit
    for i, (a, b) in enumerate(segments):
        lo, hi = min(a[0], b[0]), max(a[0], b[0])
        events.append((lo, 0, i))
        events.append((hi, 1, i))
    events.sort()  # enters sort before exits at equal x, so touching extents still get tested
    active: set[int] = set()
    crossings: set[tuple[int, int]] = set()
    tested = 0
    for _, kind, i in events:
        if kind == 0:
            for j in active:
                tested += 1
                if segments_intersect(*segments[i], *segments[j]):
                    crossings.add((min(i, j), max(i, j)))
            active.add(i)
        else:
            active.discard(i)
    return SweepResult(crossings, tested)


def brute(segments: list[Segment]) -> SweepResult:
    if segments is None:
        raise Invalid("segments must not be None")
    crossings: set[tuple[int, int]] = set()
    tested = 0
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            tested += 1
            if segments_intersect(*segments[i], *segments[j]):
                crossings.add((i, j))
    return SweepResult(crossings, tested)
