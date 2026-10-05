"""Shared CLI plumbing.

Implemented commands (`index`, `retrieve`, `validate_manifest`, `prepare_data`, `doctor`, `plan`, `run`, `status`, `bundle`, `rehearsal`) build their own
parsers. Every other proposed command stays a help-only stub that fails closed:
it parses, reports "not implemented" and returns 2 without reading or writing files.

Exit status: 0 success; 1 invalid input, validation or runtime failure; 2 usage
errors or an operation that is not implemented.
"""

import argparse
import sys
from collections.abc import Sequence

IMPLEMENTED = ("index", "retrieve", "validate_manifest", "prepare_data", "doctor", "plan", "run", "status", "bundle", "rehearsal")

STUB_COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "prepare": ("Acquire/normalize pinned sources and make chunks.", ("config", "output", "offline")),
    "train": ("Run native PyTorch training or complete-epoch recovery.", ("config", "run-dir", "resume")),
    "evaluate": ("Evaluate validation or rank/pool/score a locked final comparison.", ("config", "mode", "index", "checkpoint", "split", "output", "stage", "comparison-plan", "ranked-outputs", "test-manifest")),
    "predict": ("Score supplied question/passage pairs.", ("config", "checkpoint", "pairs", "output")),
    "serve": ("Serve a verified local candidate/evaluated release.", ("config", "manifest", "allow-candidate")),
    "export": ("Export an immutable candidate or evaluated bundle.", ("config", "mode", "index", "checkpoint", "candidate", "final-report", "output")),
}
ALL_COMMANDS = (*IMPLEMENTED, *STUB_COMMANDS)

_FLAGS = frozenset({"offline", "allow-candidate"})
_CHOICES = {
    "mode": ("baseline", "rerank", "compare"),
    "stage": ("rank", "pool", "score"),
    "split": ("development", "train", "validation", "test"),
}


def new_parser(command: str, description: str, epilog: str | None = None) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog=f"python -m devsearch.{command}", description=description, epilog=epilog,
        allow_abbrev=False,
    )


def fail(command: str, error: object) -> int:
    """Report an expected failure without a stack trace."""
    print(f"devsearch.{command}: error: {error}", file=sys.stderr)
    return 1


class _ScaffoldParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"scaffold only; operation not implemented: {message}")


def build_parser(command: str) -> argparse.ArgumentParser:
    description, arguments = STUB_COMMANDS[command]
    parser = _ScaffoldParser(
        prog=f"python -m devsearch.{command}",
        description=f"SCAFFOLD ONLY / 仅架构骨架. Proposed: {description}",
        epilog=(
            "Only --help is implemented. Options document proposed interfaces; "
            "files are not read and no operation is performed. "
            "仅帮助可用；参数为拟定接口，不读取文件或执行操作。"
        ),
        allow_abbrev=False,
    )
    for argument in arguments:
        if argument in _FLAGS:
            parser.add_argument(f"--{argument}", action="store_true", help="Proposed flag (not implemented).")
        else:
            parser.add_argument(
                f"--{argument}", choices=_CHOICES.get(argument),
                help="Proposed value; no validation or IO is performed yet.",
            )
    return parser


def main(command: str, argv: Sequence[str] | None = None) -> int:
    """Stub entry point for commands that are not implemented yet."""
    build_parser(command).parse_args(argv)
    print(
        f"devsearch.{command}: scaffold only; operation not implemented. "
        "No operation was run. / 仅架构骨架，功能尚未实现，未执行操作。",
        file=sys.stderr,
    )
    return 2
