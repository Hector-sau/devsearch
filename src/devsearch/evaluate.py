"""Help-only evaluate entry point. The operation is not implemented."""

from devsearch._cli import main as _main


def main() -> int:
    return _main("evaluate")


if __name__ == "__main__":
    raise SystemExit(main())
