"""`python -m devsearch.serve`: serve a verified release on loopback (docs/04 "Search API").

    serve --config serve.toml --manifest <release dir | release-manifest.json> [--allow-candidate] [--check-only] [--runtime-root R]

The release's structure and every file hash are verified; a candidate (a model without a final evaluation) needs
`--allow-candidate`; the profile's device must resolve (no CPU fallback); an accelerator profile holds the exclusive accelerator
lease so training and serving never overlap; a model release must pass its parity self-check and a baseline-only start is never
substituted for it; the recorded smoke must replay. Only then does it bind the configured loopback address (the API has no
authentication, so a non-loopback host is refused). `--check-only` runs every check, prints the startup report and exits 0
without binding a port. Search never downloads or trains anything. Exit: 0 clean stop (or checks passed), 1 refused or failed, 2 usage.
"""

import json
import signal
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser


def build_parser():
    parser = new_parser("serve", "Serve a verified release (loopback only).")
    parser.add_argument("--config", type=Path, required=True, help="Serve TOML (see configs/serve.toml).")
    parser.add_argument("--manifest", type=Path, required=True, help="A release directory or its release-manifest.json.")
    parser.add_argument("--allow-candidate", action="store_true", help="Allow a candidate release (a model without a final evaluation).")
    parser.add_argument("--check-only", action="store_true", help="Run every startup check, print the report and exit without serving.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain the release (default: DEVSEARCH_RUNTIME_ROOT, then its parent).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    started, listening = None, False
    previous: dict[int, object] = {}
    try:
        started = _start(args)
        print(json.dumps({"event": "startup_checks_passed", **started.startup}, sort_keys=True), flush=True)
        if args.check_only:
            return 0
        from devsearch.serving.api import serve_forever
        previous = _handle_stop_signals()
        print(json.dumps({"event": "listening", "host": started.settings.host, "port": started.settings.port}), flush=True)
        listening = True
        serve_forever(started.service, started.settings.host, started.settings.port)
        return 0
    except (DevSearchError, OSError, ValueError) as error:
        return fail("serve", error)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)             # type: ignore[arg-type]
        if started is not None:
            started.stop()
            if listening:
                print(json.dumps({"event": "stopped", "release_id": started.startup["release_id"]}), file=sys.stderr, flush=True)


def _handle_stop_signals() -> dict[int, object]:
    """Make SIGTERM/SIGINT end the server *gracefully*.

    uvicorn finishes its shutdown and then re-raises the captured signal with the handler that was installed before it
    started; with the default handler that would kill this process on the spot, skipping the lease release and the stop
    record. A handler that does nothing lets `main` return normally. (A signal in the instant before uvicorn installs its own
    handlers is absorbed too; a second one stops the server.)
    """
    return {number: signal.signal(number, lambda _number, _frame: None) for number in (signal.SIGTERM, signal.SIGINT)}


def _start(args):
    from devsearch.artifacts.paths import resolve_root
    from devsearch.execution.serving import start_service

    manifest = args.manifest.resolve()
    root = resolve_root(args.runtime_root, manifest.parent if manifest.is_file() else manifest.parent)
    return start_service(config=args.config, manifest=args.manifest, allow_candidate=args.allow_candidate, runtime_root=root)


if __name__ == "__main__":
    raise SystemExit(main())
