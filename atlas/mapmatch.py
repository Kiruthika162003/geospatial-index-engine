"""Map matching: snap a noisy trace to roads by the whole route, not one fix at a time.

A GPS trace is a string of noisy fixes, and matching it to the road
it was driven on is harder than snapping each fix to its nearest
road, because two roads running side by side sit within the noise of
each other, and per-fix nearest snapping flips between them fix by
fix, producing a route that hops across the median a dozen times on a
straight drive. The hidden Markov match fixes this by scoring whole
routes. Each fix has candidate roads, weighted by how near the road
is, the emission, and each pair of consecutive fixes has transition
weights that prefer staying on the same road or moving to a road that
connects, penalizing an unexplained jump between unconnected roads.
The Viterbi algorithm then finds the road sequence with the best
product of emissions and transitions, so a fix that looks slightly
nearer the wrong road is still assigned to the right one when the
route as a whole says so. The survey measures the difference the
route-level view makes. On two parallel roads with a trace driven
down one of them and jittered by noise comparable to their spacing,
per-fix snapping assigns a large share of the fixes to the wrong road
and flips between roads many times, while the Viterbi match assigns
every fix to the driven road with no flips at all, because a flip
costs a transition that the emission gain never repays. When the
trace genuinely changes road at a junction the match follows it, so
the transition penalty discourages hopping without forbidding turns.
The finding worth stating is that route-level matching removes the
flipping that per-fix snapping suffers on parallel roads, taking the
wrong-road fraction from a large share to zero on a jittered trace,
while still following a real turn. This module matches a trace to a
small road set by Viterbi over emissions and transitions, and a survey
counts the flips and wrong-road fixes against per-fix snapping.
"""

from __future__ import annotations

from itertools import pairwise

from atlas.errors import Invalid
from atlas.snaptopath import snap

Point = tuple[float, float]


class RoadNetwork:
    def __init__(self, roads: dict[str, list[Point]], links: set[tuple[str, str]]) -> None:
        if not roads:
            raise Invalid("need at least one road")
        for name, path in roads.items():
            if len(path) < 2:
                raise Invalid(f"road {name} needs at least two vertices")
        self.roads = roads
        self.links = {(a, b) for a, b in links} | {(b, a) for a, b in links}

    def connected(self, a: str, b: str) -> bool:
        return a == b or (a, b) in self.links

    def nearest_snap(self, trace: list[Point]) -> list[str]:
        return [min(self.roads, key=lambda n: snap(p, self.roads[n])[1]) for p in trace]

    def match(self, trace: list[Point], sigma: float, jump_penalty: float) -> list[str]:
        if not trace:
            raise Invalid("the trace must not be empty")
        if sigma <= 0 or jump_penalty < 0:
            raise Invalid("sigma must be positive and the jump penalty non-negative")
        names = list(self.roads)

        def emission(p: Point, name: str) -> float:
            d = snap(p, self.roads[name])[1]
            return -(d * d) / (2 * sigma * sigma)  # log of a Gaussian, constants dropped

        def transition(a: str, b: str) -> float:
            return 0.0 if self.connected(a, b) else -jump_penalty

        score = {n: emission(trace[0], n) for n in names}
        back: list[dict[str, str]] = []
        for p in trace[1:]:
            new_score = {}
            pointers = {}
            for n in names:
                best_prev = max(names, key=lambda m: score[m] + transition(m, n))
                new_score[n] = score[best_prev] + transition(best_prev, n) + emission(p, n)
                pointers[n] = best_prev
            score = new_score
            back.append(pointers)
        last = max(names, key=lambda n: score[n])
        route = [last]
        for pointers in reversed(back):
            route.append(pointers[route[-1]])
        route.reverse()
        return route


def flips(assignment: list[str]) -> int:
    return sum(1 for a, b in pairwise(assignment) if a != b)


def wrong_fraction(assignment: list[str], truth: list[str]) -> float:
    if len(assignment) != len(truth):
        raise Invalid("assignment and truth must have equal length")
    return sum(1 for a, t in zip(assignment, truth, strict=True) if a != t) / len(truth)
