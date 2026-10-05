"""Help-only train entry point. The operation is not implemented."""

from devsearch._cli import main as _main


def main() -> int:
    return _main("train")


if __name__ == "__main__":
    raise SystemExit(main())
