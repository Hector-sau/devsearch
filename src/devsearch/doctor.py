"""`python -m devsearch.doctor`: preflight and calibration on this host. It opens no test file and downloads nothing.

Exit: 0 every check passed (unverified items are reported, not hidden), 1 the bundle or a manifest is invalid,
2 usage error, 3 a prerequisite failed (device, mount, space, caps, calibration). The report is written in every
case where the host file and bundle could be read, so the failure itself is evidence; an existing report is never
overwritten.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("doctor", "Check this host and calibrate the device profile before planning.")
    parser.add_argument("--profile", required=True, help="apple-mps, nvidia-cuda or cpu-diagnostic (must match the host file).")
    parser.add_argument("--host", type=Path, required=True, help="Absolute host.toml binding paths, volume identity, deadline and caps.")
    parser.add_argument("--bundle", type=Path, help="The prepared bundle (default: the host file's paths.bundle).")
    parser.add_argument("--output", type=Path, required=True, help="doctor-report.json to write (never overwritten).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import ConfigError, DevSearchError
    try:
        from devsearch.artifacts.canonical import pretty_json
        from devsearch.artifacts.files import write_atomic
        from devsearch.execution.doctor import run_doctor
        from devsearch.execution.hostfile import load_host
        host = load_host(args.host)
        if host.profile != args.profile:
            print(f"devsearch.doctor: error: --profile {args.profile} does not match the host file's profile {host.profile}", file=sys.stderr)
            return 2
        bundle = args.bundle or host.bundle
        if Path(bundle).resolve() != host.bundle.resolve():
            raise ConfigError("--bundle must be the bundle path the host file binds (paths.bundle)")
        if args.output.exists():
            raise ConfigError(f"{args.output} already exists; reports are never overwritten")
        report = run_doctor(host, bundle)
        write_atomic(args.output, pretty_json(report).encode("utf-8"), exclusive=True)
    except (DevSearchError, OSError) as error:
        print(f"devsearch.doctor: error: {error}", file=sys.stderr)
        return 1
    failed = [c for c in report["checks"] if c["status"] == "failed"]
    print(json.dumps({"report": str(args.output.resolve()), "report_sha256": report["manifest_sha256"], "gates": {k: v["status"] for k, v in report["gates"].items()},
                      "failed_checks": [c["name"] for c in failed], "diagnostic": report["diagnostic"], "unhealthy_host": report["unhealthy_host"]}, indent=2))
    if any(c["name"] == "bundle" for c in failed):
        return 1
    return 3 if failed or report["unhealthy_host"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
