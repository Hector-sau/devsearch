"""Shared CLI plumbing.

Every command module (`devsearch.<command>`) builds its own parser with `new_parser` and reports expected failures with
`fail`; `IMPLEMENTED` is the inventory the scaffold tests check against the documented commands. A command that is only
proposed would be added here as a help-only stub that fails closed (parse, report "not implemented", exit 2, touch nothing);
there are none today.

Exit status: 0 success; 1 invalid input, validation or runtime failure; 2 usage errors. `run` and `plan` add documented
campaign statuses (docs/13).
"""

import argparse
import sys

IMPLEMENTED = ("index", "retrieve", "validate_manifest", "prepare_data", "doctor", "plan", "run", "status", "bundle", "rehearsal", "train", "predict", "evaluate", "export", "serve", "prepare")

ALL_COMMANDS = IMPLEMENTED


def new_parser(command: str, description: str, epilog: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=f"python -m devsearch.{command}", description=description, epilog=epilog,
        allow_abbrev=False,
    )


def fail(command: str, error: object) -> int:
    """Report an expected failure without a stack trace."""
    print(f"devsearch.{command}: error: {error}", file=sys.stderr)
    return 1
