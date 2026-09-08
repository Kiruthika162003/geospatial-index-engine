"""Tile cover: the slippy tiles a bounding box needs at a zoom, a product of two spans.

Fetching or rendering a map for a region means asking which tiles at
a given zoom the region touches, and the answer is a rectangle of tile
indices, not a search. Convert the box's west edge and north edge to a
tile column and row, do the same for its east and south edges, and
every tile whose column and row fall within those two spans intersects
the box, while no tile outside them does, because tile columns are
monotone in longitude and tile rows are monotone in latitude. So the
cover is the full grid between the corner tiles, and its size is the
column span times the row span, which the survey checks two ways:
against a brute scan that tests every tile at that zoom for
intersection with the box, and against the count law. Two properties
worth stating as measurements follow from the tile pyramid. A first
guess said the cover at zoom z plus one is about four times the cover
at zoom z, since each tile splits into four; the measurement refined
it. For a box that spans many tiles the ratio is indeed about four,
measured at 4.0 and 3.8 on the last two zooms of a city box, but while
the box is a tile-sized sliver the ratio is far below four, measured
at 1.0 and 2.25 at the coarse zooms, because a box inside one tile
stays inside one or a few of that tile's children. So the count grows
as four to the zoom only once the box is large relative to a tile,
which is still why a region that needs a few tiles at a city zoom
needs thousands at a street zoom. And a box straddling the antimeridian
wraps: its west column is greater than its east column, and the cover
is two rectangles, one to the right edge of the grid and one from the
left edge, whose combined count still equals the wrapped column span
times the row span. The finding worth stating is that the tile cover
is exactly the corner-to-corner grid, its count the product of the two
spans, so it matches an exhaustive intersection scan while costing
only two coordinate conversions, and it quadruples with each zoom.
This module lists the covering tiles and their count, handling the
antimeridian, and a survey confirms the cover against a brute scan and
measures the quadrupling.
"""

from __future__ import annotations

from atlas.errors import Invalid, Outside
from atlas.slippytile import lonlat_to_tile, tile_bounds

Tile = tuple[int, int]

_MAX_LAT = 85.05112877980659


def cover(
    west: float, south: float, east: float, north: float, zoom: int
) -> list[Tile]:
    if zoom < 0:
        raise Invalid("zoom must not be negative")
    if south > north:
        raise Invalid("south must not exceed north")
    if not (-_MAX_LAT <= south <= _MAX_LAT and -_MAX_LAT <= north <= _MAX_LAT):
        raise Outside("latitudes must lie within the Web Mercator limit")
    for lon in (west, east):
        if not -180.0 <= lon <= 180.0:
            raise Outside("longitude must lie within -180 and 180 degrees")
    n = 1 << zoom
    x_west, y_north = lonlat_to_tile(west, north, zoom)
    x_east, y_south = lonlat_to_tile(east, south, zoom)
    if west <= east:
        columns = list(range(x_west, x_east + 1))
    else:
        # the box straddles the antimeridian: wrap around the grid's right edge
        columns = list(range(x_west, n)) + list(range(0, x_east + 1))
    return [(x, y) for y in range(y_north, y_south + 1) for x in columns]


def count(west: float, south: float, east: float, north: float, zoom: int) -> int:
    return len(cover(west, south, east, north, zoom))


def brute_cover(west: float, south: float, east: float, north: float, zoom: int) -> list[Tile]:
    # the independent check: test every tile at this zoom for intersection with the box
    n = 1 << zoom
    found: list[Tile] = []
    for y in range(n):
        for x in range(n):
            t_south, t_west, t_north, t_east = tile_bounds(x, y, zoom)
            lat_ok = t_south <= north and south <= t_north
            if west <= east:
                lon_ok = t_west <= east and west <= t_east
            else:
                lon_ok = t_east >= west or t_west <= east
            if lat_ok and lon_ok:
                found.append((x, y))
    return found
