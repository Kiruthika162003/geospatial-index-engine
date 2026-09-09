"""Huff shares: a store four times the size takes 0.57 with no decay, 0.43 at beta 2, 0.32 at 5.

Huff's model sends a customer to each store with a probability
proportional to the store's size over its distance to the power of
a decay exponent beta, and the exponent decides everything. Four
stores over a 100 square, one of them four times the size, and
3000 uniform customers give the big store 0.5714 of the market at
beta 0, its size share, then 0.5522, 0.5123, 0.4675, 0.4284, 0.3742
and 0.3219 at beta 0.5, 1, 1.5, 2, 3 and 5, against 0.2273 by
nearest store. The guess that a high exponent hands each customer to
the nearest store was wrong within any exponent a planner would
use: at beta 5 the central store reads 0.266 against its nearest
share of 0.357, since a customer between stores still splits, and
the mean entropy of choice reads 1.154, 1.063, 0.813 and 0.392 nats
at beta 0, 1, 2 and 5 against a maximum of 1.386. The big store's
trade area at the half-probability level holds all 3000 customers
at beta 0, 1390 at beta 1, 1160 at 2 and 934 at 5, and its share
drifts by 0.138 between beta 1 and 3.

Reilly's breakeven between a store four times bigger and a small
one 100 apart sits 80.0 from the big store at beta 1, 66.7 at beta
2 and 56.9 at beta 5, closing on the midpoint as the exponent
grows.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def probabilities(
    point: Point, stores: list[Point], sizes: list[float], beta: float
) -> list[float]:
    if len(stores) != len(sizes) or not stores:
        raise Invalid("one size per store, and at least one store")
    if beta < 0:
        raise Invalid("the decay exponent must not be negative")
    if any(s <= 0 for s in sizes):
        raise Invalid("store sizes must be positive")
    weights = []
    for store, size in zip(stores, sizes, strict=True):
        d = max(math.dist(point, store), 1e-9)
        weights.append(size / d**beta)
    total = sum(weights)
    return [w / total for w in weights]


def market_shares(
    customers: list[Point], stores: list[Point], sizes: list[float], beta: float
) -> list[float]:
    if not customers:
        raise Invalid("at least one customer is needed")
    totals = [0.0] * len(stores)
    for p in customers:
        for i, prob in enumerate(probabilities(p, stores, sizes, beta)):
            totals[i] += prob
    return [t / len(customers) for t in totals]


def nearest_shares(customers: list[Point], stores: list[Point]) -> list[float]:
    counts = [0] * len(stores)
    for p in customers:
        counts[min(range(len(stores)), key=lambda i: math.dist(p, stores[i]))] += 1
    return [c / len(customers) for c in counts]


def size_shares(sizes: list[float]) -> list[float]:
    total = sum(sizes)
    return [s / total for s in sizes]


def trade_area(
    customers: list[Point],
    stores: list[Point],
    sizes: list[float],
    beta: float,
    store: int,
    level: float,
) -> int:
    # customers whose probability of choosing the store passes the level
    return sum(1 for p in customers if probabilities(p, stores, sizes, beta)[store] >= level)


def breakeven_distance(size_a: float, size_b: float, separation: float, beta: float) -> float:
    # Reilly's point between two stores on the line joining them
    if beta <= 0:
        raise Invalid("the breakeven needs a positive exponent")
    ratio = (size_a / size_b) ** (1 / beta)
    return separation * ratio / (1 + ratio)


def entropy_of_choice(
    point: Point, stores: list[Point], sizes: list[float], beta: float
) -> float:
    return -sum(p * math.log(p) for p in probabilities(point, stores, sizes, beta) if p > 0)


def uniform_customers(n: int, rng: random.Random, side: float = 100.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def share_drift(
    customers: list[Point], stores: list[Point], sizes: list[float], betas: list[float]
) -> float:
    # the largest change in any store's share across the exponents tried
    readings = [market_shares(customers, stores, sizes, b) for b in betas]
    return max(
        max(r[i] for r in readings) - min(r[i] for r in readings) for i in range(len(stores))
    )
