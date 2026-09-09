"""Track compression: dropping GPS fixes without losing where the vehicle was when.

A vehicle track is a list of fixes with times, and most of them
say nothing new: a car on a straight road at steady speed reports
the same line every second. Compressing the track keeps the fixes
that matter, and there are two ideas of mattering. The
Douglas-Peucker rule keeps a fix when it lies far from the
straight line between the fixes already kept, a purely spatial
test that preserves the track's shape but not its timing, so a
car that stops for ten minutes on a straight road is compressed
to a line that says nothing of the stop. The time-ratio rule,
the synchronized Euclidean distance, keeps a fix when it lies far
from where the vehicle would have been at that time had it moved
uniformly between the kept fixes, so a stop on a straight road is
kept, since during the stop the uniform-motion position runs
ahead of the real one. The survey measures the two on tracks with
known events. On a straight steady track of sixty fixes both keep
only the ends. On a straight track at 10 meters a second with a
sixty-second stop in the middle the spatial rule keeps two fixes
and the time rule keeps the stop's ends as well, four, and the
guess about the spatial rule's time error was wrong in an
instructive way: the guess was that the compressed track would
mislocate the vehicle at the stop's midpoint by half the stop's
worth of travel, but a stop in the middle of a symmetric track
puts the uniform-motion position at the stop's midpoint within
2.5 meters of the truth, since the vehicle's average position
over the whole track is the stop; the error lives at the stop's
ends, where uniform motion puts the vehicle 151 meters short of
the stop as it arrives and 151 past it as it leaves, a quarter of
the 600 meters it would have driven, while the time rule's worst
error was 0.0. On a bending track at steady speed the two rules
keep the same five corners. On a 600-fix track with turns and
speed changes every hundred fixes and half a meter of noise the
comparison depends on the tolerance: at 1 meter the time rule
kept 248 fixes with a worst time error of 1.0 against the spatial
rule's 146 fixes and 1.96, at 2 meters 11 against 9, and at 5
meters and beyond it kept 7 with a time error of 2.0 while the
spatial rule kept 7 to 9 with a time error that jumped to 48
meters at a tolerance of 10, where it dropped a speed change the
time rule kept. The finding worth stating is that spatial
compression is blind to stops and speed changes on straight
roads, mislocating the vehicle by a quarter of a stop's travel at
the stop's ends and by 48 meters at a dropped speed change, while
time-ratio compression keeps the fixes those events need and
mislocates by no more than the tolerance, so a track that must
answer when as well as where needs the time rule. This module
compresses tracks by both rules, and a survey measures the kept
counts and the errors on known tracks.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Fix = tuple[float, float, float]  # x, y, time


def _perpendicular(p: Fix, a: Fix, b: Fix) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    return abs(dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / norm


def _synchronized(p: Fix, a: Fix, b: Fix) -> float:
    # distance from the fix to where uniform motion from a to b would put it at the fix's time
    span = b[2] - a[2]
    t = 0.0 if span == 0 else (p[2] - a[2]) / span
    x = a[0] + t * (b[0] - a[0])
    y = a[1] + t * (b[1] - a[1])
    return math.hypot(p[0] - x, p[1] - y)


def _compress(track: Sequence[Fix], tolerance: float, error) -> list[int]:
    if len(track) < 2:
        raise Invalid("a track needs at least two fixes")
    if tolerance < 0:
        raise Invalid("the tolerance cannot be negative")
    keep = {0, len(track) - 1}
    stack = [(0, len(track) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        worst, at = -1.0, i
        for k in range(i + 1, j):
            d = error(track[k], track[i], track[j])
            if d > worst:
                worst, at = d, k
        if worst > tolerance:
            keep.add(at)
            stack.append((i, at))
            stack.append((at, j))
    return sorted(keep)


def spatial(track: Sequence[Fix], tolerance: float) -> list[int]:
    return _compress(track, tolerance, _perpendicular)


def time_ratio(track: Sequence[Fix], tolerance: float) -> list[int]:
    return _compress(track, tolerance, _synchronized)


def position_at(track: Sequence[Fix], kept: Sequence[int], time: float) -> tuple[float, float]:
    # where the compressed track puts the vehicle at a time, by uniform motion between fixes
    for i, j in itertools.pairwise(kept):
        a, b = track[i], track[j]
        if a[2] <= time <= b[2]:
            t = 0.0 if b[2] == a[2] else (time - a[2]) / (b[2] - a[2])
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    raise Invalid("the time lies outside the track")


def worst_errors(track: Sequence[Fix], kept: Sequence[int]) -> tuple[float, float]:
    # the largest perpendicular and the largest time-synchronized error over dropped fixes
    kept_set = set(kept)
    spatial_worst = time_worst = 0.0
    for i, j in itertools.pairwise(kept):
        for k in range(i + 1, j):
            if k in kept_set:
                continue
            spatial_worst = max(spatial_worst, _perpendicular(track[k], track[i], track[j]))
            time_worst = max(time_worst, _synchronized(track[k], track[i], track[j]))
    return spatial_worst, time_worst


def straight_with_stop(speed: float, before: int, stop: int, after: int) -> list[Fix]:
    fixes: list[Fix] = []
    x = 0.0
    t = 0.0
    for _ in range(before):
        fixes.append((x, 0.0, t))
        x += speed
        t += 1
    for _ in range(stop):
        fixes.append((x, 0.0, t))
        t += 1
    for _ in range(after):
        fixes.append((x, 0.0, t))
        x += speed
        t += 1
    return fixes
