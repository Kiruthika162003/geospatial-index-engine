"""A street grid turned by 3 and 17 degrees read for orientation order, beside a Delaunay web.

The drill builds a 20 by 20 street grid, reads its orientation order,
turns it by 3 and 17 degrees and reads again, then reads a Delaunay
web on 100 uniform points. The guess before measuring was that any
turn would lower the order a little, since the histogram's 36 slices
are fixed to the compass and a turned grid no longer sits on the
cardinal centres. The measurement refutes that for turns that stay
inside a slice: the grid reads 1.0 turned by 0, 3 and 17 degrees to
nine places, because every bearing of a grid lands in one slice as
long as the turn keeps clear of the slice edges at 5 degrees past a
cardinal. The web reads 0.0239, near the uniform floor. The survey
keeps the drop guess beside the three readings of 1.0.
"""

from __future__ import annotations

import random

from atlas.detourindex import grid_network
from atlas.streetorientation import network_order, rotated, uniform_web
from atlas.surveys.survey import Survey


def run() -> Survey:
    points, graph = grid_network(20)
    orders = {}
    for degrees in (0, 3, 17):
        orders[degrees] = network_order(rotated(points, degrees), graph)[1]
    web_points, web_graph = uniform_web(100, random.Random(352))
    web_entropy, web_order = network_order(web_points, web_graph)
    readings = {
        "grid_order_at_0_3_17": [round(orders[d], 9) for d in (0, 3, 17)],
        "web_entropy": round(web_entropy, 4),
        "web_order": round(web_order, 4),
    }
    holds = all(abs(orders[d] - 1.0) < 1e-9 for d in (0, 3, 17)) and web_order < 0.05
    return Survey(
        surveyor="gridturn",
        finding=(
            "a 20 by 20 grid read an orientation order of 1.0 turned by 0, 3 and 17 "
            "degrees, and a Delaunay web on 100 points read 0.0239"
        ),
        readings=readings,
        holds=holds,
    )
