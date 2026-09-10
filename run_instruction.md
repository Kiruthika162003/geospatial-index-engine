# Running geospatial-index-engine

A spatial index engine, written in Python. This file covers how to build the
project, run its tests, and execute it. Every command below was run from a clean
checkout of this repository before being written down.

## Requirements

Python 3.11 or newer. There are no third party runtime dependencies.

## Setup

Nothing to install. The package has no dependencies, so every command below runs
from the repository root against the source tree as it stands.

## Run the tests

```bash
python -m pytest -q
```

The suite is the primary check. It runs from the repository root with no
arguments and no configuration, and it prints the number of tests it ran. A
non-zero exit status means something is wrong. Read the printed summary rather
than a shell pipeline, because piping the output through another command
replaces the real exit code with that of the last command in the pipe.

## Lint

```bash
python -m ruff check .
```

The lint configuration lives in `pyproject.toml`. It passes with no findings.

## Run the command line tool

```bash
python -m atlas.cli --help
```

The subcommands are `surveys`, `check`, and `summary`.

`summary` prints one line giving the number of surveys and how many are broken:

```bash
python -m atlas.cli summary
```

`surveys` prints every survey in full. `check` exits non-zero if any survey is
broken, which makes it usable as a build gate:

```bash
python -m atlas.cli check
```

At the time of writing that reports `10 surveys (0 broken)` and exits zero.

## Layout

- `atlas/` the package itself, 230 modules
- `atlas/surveys/` the verification organ, a set of measured claims about the
  package as a whole rather than unit tests of one function
- `tests/` the test suite, 231 files

There is no `examples/` directory in this repository. The surveys serve that
purpose: each one exercises the engine end to end and reports a measured number
rather than a printed transcript.

## Notes

Tests lock the measured numbers rather than asserting expected ones. Where a
number came from a random draw the seed is fixed and the tolerance is the one
the measurement earned. Where a guess about behaviour was refuted by
measurement, the wrong guess is kept in the source beside the measured value
rather than deleted.
