"""Euclidean minimum spanning tree: Kruskal over Delaunay edges finds it without all pairs.

The minimum spanning tree of a point set under straight-line distance
is the cheapest network connecting them all, the skeleton of a road
plan or a cable layout, and computing it over every pair of points is
quadratic in edges before the sort even starts. The saving comes from
a theorem: the Euclidean minimum spanning tree is a subgraph of the
Delaunay triangulation. Any edge of the tree must have an empty lune,
no point closer to both endpoints than they are to each other, or else
that closer point would give a cheaper way to join the two sides;
that lune contains the diameter circle, so the edge is a Gabriel edge,
and every Gabriel edge is a Delaunay edge. So Kruskal's algorithm run
on the roughly three n Delaunay edges, sorted by length and joined
with a union-find that skips any edge closing a cycle, yields the
exact same tree that Kruskal on all n squared pairs would, at a
fraction of the sorting. Three checks pin it and are worth measuring.
The tree has exactly n minus one edges and connects every point,
which the union-find guarantees by construction. Its total length
equals the length of the all-pairs Kruskal tree, edge for edge in
length, which is the theorem confirmed. And it sits at the bottom of
the proximity family, inside the relative-neighborhood graph and
hence inside Gabriel and Delaunay, so the chain runs tree inside RNG
inside Gabriel inside Delaunay, each a superset of the last. The
finding worth stating is that restricting Kruskal to Delaunay edges
loses nothing, the tree lengths match exactly, while the candidate
edge count drops from n squared over two to about three n, so the
theorem is a free speedup. This module builds the tree by Kruskal
over Delaunay edges, and a survey confirms its length against the
all-pairs tree and its place inside the relative-neighborhood graph.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid
from atlas.gabriel import delaunay_edges

Point = tuple[float, float]
Edge = tuple[Point, Point]


def _length(edge: Edge) -> float:
    return math.hypot(edge[0][0] - edge[1][0], edge[0][1] - edge[1][1])


def _kruskal(points: list[Point], candidates: list[Edge]) -> list[Edge]:
    parent: dict[Point, Point] = {p: p for p in points}

    def find(p: Point) -> Point:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    tree: list[Edge] = []
    for edge in sorted(candidates, key=_length):
        ra, rb = find(edge[0]), find(edge[1])
        if ra != rb:
            parent[ra] = rb
            tree.append(edge)
            if len(tree) == len(points) - 1:
                break
    return tree


def minimum_spanning_tree(points: list[Point]) -> list[Edge]:
    if points is None or len(set(points)) < 2:
        raise Invalid("need at least two distinct points")
    pts = list(dict.fromkeys(points))
    if len(pts) == 2:
        return [(pts[0], pts[1])]
    return _kruskal(pts, list(delaunay_edges(pts)))


def all_pairs_tree(points: list[Point]) -> list[Edge]:
    # the brute reference: Kruskal over every pair
    if points is None or len(set(points)) < 2:
        raise Invalid("need at least two distinct points")
    pts = list(dict.fromkeys(points))
    pairs = [(pts[i], pts[j]) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    return _kruskal(pts, pairs)


def total_length(tree: list[Edge]) -> float:
    return sum(_length(e) for e in tree)
