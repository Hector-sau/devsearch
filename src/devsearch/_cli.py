"""Help-only proposed CLI contracts; all operational paths fail closed."""

import argparse
import sys
from collections.abc import Sequence

COMMANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "prepare": ("Acquire/normalize pinned sources and make chunks.", ("config", "output", "offline")),
    "prepare_data": ("Audit labels, prepare splits, or seal final judgments.", ("config", "corpus", "output", "seal-final", "comparison-plan", "reviews")),
    "index": ("Build a versioned BM25 index.", ("config", "corpus", "output")),
    "validate_manifest": ("Validate artifact schema, hashes and containment.", ("manifest", "runtime-root")),
    "train": ("Run native PyTorch training or complete-epoch recovery.", ("config", "run-dir", "resume")),
    "evaluate": ("Evaluate validation or rank/pool/score a locked final comparison.", ("config", "mode", "index", "checkpoint", "split", "output", "stage", "comparison-plan", "ranked-outputs", "test-manifest")),
    "predict": ("Score supplied question/passage pairs.", ("config", "checkpoint", "pairs", "output")),
    "retrieve": ("Search with an explicit baseline or reranking mode.", ("config", "mode", "index", "checkpoint", "query", "output")),
    "serve": ("Serve a verified local candidate/evaluated release.", ("config", "manifest", "allow-candidate")),
    "export": ("Export an immutable candidate or evaluated bundle.", ("config", "mode", "index", "checkpoint", "candidate", "final-report", "output")),
}

_FLAGS = frozenset({"offline", "seal-final", "allow-candidate"})
_CHOICES = {
    "mode": ("baseline", "rerank", "compare"),
    "stage": ("rank", "pool", "score"),
    "split": ("development", "train", "validation", "test"),
}


class _ScaffoldParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(f"scaffold only; operation not implemented: {message}")


def build_parser(command: str) -> argparse.ArgumentParser:
    description, arguments = COMMANDS[command]
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
    build_parser(command).parse_args(argv)
    print(
        f"devsearch.{command}: scaffold only; operation not implemented. "
        "No operation was run. / 仅架构骨架，功能尚未实现，未执行操作。",
        file=sys.stderr,
    )
    return 2
