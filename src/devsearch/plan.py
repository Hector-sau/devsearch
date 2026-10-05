"""`python -m devsearch.plan`: create the immutable execution plan, or add a host-binding revision.

    plan --host host.toml --doctor-report doctor-report.json --output execution-plan.json [--bundle DIR] [--receipts DIR]
    plan --revise-binding --plan execution-plan.json --host host.toml [--authorization TEXT]

A plan is created only after the gates allow it (diagnostic bundles may make a rehearsal plan only). A revision may
change runtime/output/backup paths and the deadline/reserve after fresh mount, hash and access checks; it records its
parent and any authorization for additional expense, and never changes device, model, precision, batch, data or recipe.
Exit: 0 written, 1 invalid input, 2 usage, 3 a prerequisite or gate blocks the plan.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("plan", "Create an immutable execution plan, or revise the host binding.")
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--doctor-report", type=Path)
    parser.add_argument("--output", type=Path, help="execution-plan.json to write (its directory also receives host-binding-0.json).")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--receipts", type=Path, help="Directory of evidence receipts for the implementation and portable-bundle gates.")
    parser.add_argument("--wheel-sha256", help="SHA-256 of the project wheel being run (recorded, not downloaded).")
    parser.add_argument("--revise-binding", action="store_true")
    parser.add_argument("--plan", type=Path, help="With --revise-binding: the existing execution-plan.json.")
    parser.add_argument("--authorization", help="With --revise-binding: evidence that any additional expense was authorized.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError, PrerequisiteError
    try:
        from devsearch.execution.hostfile import load_host
        host = load_host(args.host)
        if args.revise_binding:
            return _revise(args, host)
        if args.doctor_report is None or args.output is None:
            print("devsearch.plan: error: planning needs --doctor-report and --output", file=sys.stderr)
            return 2
        from devsearch.execution.plan import build_plan, write_plan
        from devsearch.execution.records import read_json
        bundle = args.bundle or host.bundle
        if Path(bundle).resolve() != host.bundle.resolve():
            raise DevSearchError("--bundle must be the bundle path the host file binds (paths.bundle)")
        report = read_json(args.doctor_report, str(args.doctor_report))
        plan, binding = build_plan(host, bundle, report, receipts_dir=args.receipts, wheel_sha256=args.wheel_sha256)
        write_plan(args.output, plan, binding)
    except PrerequisiteError as error:
        print(f"devsearch.plan: error: {error}", file=sys.stderr)
        return 3
    except (DevSearchError, OSError) as error:
        print(f"devsearch.plan: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"plan": str(args.output.resolve()), "plan_id": "plan-" + plan["manifest_sha256"][:32], "rehearsal": plan["rehearsal"],
                      "gates": {k: v["status"] for k, v in plan["gates"].items()}, "estimated_seconds": plan["budget"]["estimates"]["total_seconds"],
                      "estimate_label": "estimate"}, indent=2))
    return 0


def _revise(args, host) -> int:
    from devsearch.execution.hostfile import revise_binding, run_checks
    from devsearch.execution.plan import load_plan
    from devsearch.execution.records import binding_chain, write_json
    from devsearch.errors import DevSearchError, PrerequisiteError
    if args.plan is None:
        print("devsearch.plan: error: --revise-binding needs --plan", file=sys.stderr)
        return 2
    plan, _ = load_plan(args.plan)
    directory = args.plan.resolve().parent
    from devsearch.execution.runner import latest_binding
    parent = latest_binding(directory, plan["host_binding"]["sha256"])
    chain = binding_chain(directory, parent["manifest_sha256"])
    revision = revise_binding(chain[-1], host, authorization=args.authorization)
    failed = [c for c in run_checks(revision) if c["status"] == "failed"]
    if failed:
        raise PrerequisiteError("the revised host file fails fresh checks: " + "; ".join(f"{c['name']}: {c['detail']}" for c in failed))
    path = directory / f"host-binding-{revision['revision']}.json"
    if path.exists():
        raise DevSearchError(f"{path} already exists")
    write_json(path, revision, exclusive=True)
    print(json.dumps({"revision": revision["revision"], "binding": str(path), "parent_sha256": revision["parent_sha256"], "deadline": revision["deadline"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
