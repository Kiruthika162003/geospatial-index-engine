# geospatial-index-engine

`atlas` is a spatial index engine written in plain Python: trees that
carve the plane, curves that flatten it, distances that respect the curve
of the earth, and the map geometry, raster methods and point-pattern
statistics that sit on top of them. Every module states a finding it
measured, and where the guess made before measuring turned out wrong,
the docstring keeps the wrong guess beside the measured truth.

## What is here

The package holds 228 modules and 10 surveys, checked by 1,930 tests,
across 30,076 strict lines of code (docstrings, comments and blank lines
excluded), built over 240 commits, each commit carrying its module's
finding.

Families of modules, with a finding from each:

- Trees and space-filling curves: kd-trees, quadtrees, R-trees, ball and
  VP trees, Morton, Hilbert and geohash orders. A kd-tree grown by
  insertion survives x-sorted input at twice the height, and diagonally
  sorted input wrecks it; STR packing and a plain x sort both reach zero
  sibling overlap, yet the strips touch 11.3 leaves a query to STR's 2.7.
- Grids and hashes: spatial hash cells, ring searches, quantised boxes,
  fat boxes for moving objects, tile caches. Stopping a ring search at the
  first ring holding a point is wrong 31 percent of the time at three
  points a cell; an LRU tile cache smaller than one viewport gets no hits.
- Geodesy and projections: haversine, Vincenty, rhumb lines, transverse
  Mercator, equal-area maps, Tissot's indicatrix, grid convergence, the
  sagitta between a map chord and its great circle. A straight line on a
  Mercator map bows 723 km off the great circle from New York to London.
- Rasters and terrain: distance transforms, sink filling, height above
  drainage, viewsheds and sky view, hillshade and solar beam, pyramids,
  regridding, nodata policies. Noise of a tenth of a wall step triples a
  valley's floodplain by merging rows into rills that filling cannot undo.
- Point patterns and statistics: Ripley's K, Moran's I, Getis-Ord, spatial
  weights, kriging, class breaks, MAUP. Aggregating 5000 points into cells
  lifts a correlation from 0.42 to 0.997 and restores an attenuated slope.
- Routes and networks: cost distance, isochrones, corridors, detour
  indices, betweenness, allocation, gravity models. A bridge's two ends
  each carry exactly the crossing-pair share of every shortest path.
- Encodings, tiles and map production: plus codes, Maidenhead, quadkeys,
  vector tiles, label placement, map sheets, edge matching. Vector tiles
  hold more vertices in total at zoom 10 than the raw lines.

## Surveys

A survey is a measured drill: it builds a scene from the real machinery,
runs a query or a construction against it, and reports the numbers, with
a `holds` flag the registry gates on. The ten surveys run in under two
seconds together:

```bash
python -m atlas.cli surveys
python -m atlas.cli check
python -m atlas.cli summary
```

## Working on it

```bash
python -m pytest tests/ -q
python -m ruff check atlas tests
```

Python 3.11 or later, no third-party dependencies. Tests lock the
measured numbers; where a number came from a random draw the seed is
fixed and the tolerance is the one the measurement earned.
