"""The command line: three verbs, no ceremony."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("surveys", help="every survey in full")
    commands.add_parser("check", help="exit nonzero if any survey is broken")
    commands.add_parser("summary", help="one line: surveys and broken count")
    arguments = parser.parse_args(argv)
    from atlas.surveys import registry

    if arguments.command == "surveys":
        for survey in registry.all_surveys():
            print(survey.detail())
        return 0
    if arguments.command == "check":
        failing = registry.broken()
        if failing:
            print(f"{len(failing)} broken survey(s): " + ", ".join(failing))
            return 1
        print("all surveys hold")
        return 0
    if arguments.command == "summary":
        surveys = registry.all_surveys()
        failing = sum(1 for survey in surveys if not survey.holds)
        print(f"{len(surveys)} surveys ({failing} broken)")
        return 1 if failing else 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
