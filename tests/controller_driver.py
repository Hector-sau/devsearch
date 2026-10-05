"""Runs a tiny fake campaign (no ML) so the controller's recovery paths can be exercised in real processes.

    python controller_driver.py <workdir> [--resume] [--reviews] [--deadline ISO] [--estimate STAGE=SECONDS]

Stages: a -> b -> c -> d. Each appends its id to `<workdir>/ran.log` when it actually executes, so a test can
prove which stages were re-run. Behavior is steered by files in `<workdir>/control/` (so the same command line can
be repeated across a crash): `fail-<stage>` (fatal ExecutionFailure), `disk-<stage>` (ENOSPC), `bug-<stage>`
(unexpected exception), `interrupt-<stage>` (KeyboardInterrupt), `review-<stage>` (awaiting review until --reviews),
`unmount` (runtime verification fails).
"""

import errno
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from devsearch.errors import ExecutionFailure, PrerequisiteError  # noqa: E402
from devsearch.execution.controller import Campaign, Stage, StageOutcome  # noqa: E402
from devsearch.execution.records import binding_manifest, write_json  # noqa: E402


def make_binding(deadline: str):
    return binding_manifest({
        "revision": 0, "parent_sha256": None, "paths": {"runtime_root": "/r", "run_root": "/r/run", "evaluation_root": "/e", "backup_dir": "/b"},
        "mount": {"verified": True}, "deadline": deadline, "reserve_seconds": 60, "limits": {}, "resources": {}, "device": {"profile": "cpu-diagnostic"},
        "authorization": None})


def build(work: Path, *, deadline="2099-01-01T00:00:00Z", estimates=None, reviews=False, plan_marker="plan-v1", clock=None):
    control = work / "control"
    run_root = work / "run"
    plan_dir = work / "plan"
    for directory in (control, run_root, plan_dir):
        directory.mkdir(exist_ok=True)
    binding = make_binding(deadline)
    if not (plan_dir / "host-binding-0.json").exists():
        write_json(plan_dir / "host-binding-0.json", binding)

    def log(stage_id):
        with (work / "ran.log").open("a") as handle:
            handle.write(stage_id + "\n")

    def make(stage_id, needs, tag):
        def run(ctx):
            log(stage_id)
            ctx.crash_point("mid_stage")
            if (control / f"interrupt-{stage_id}").exists():
                raise KeyboardInterrupt
            if (control / f"fail-{stage_id}").exists():
                raise ExecutionFailure("out of memory at epoch 1")
            if (control / f"disk-{stage_id}").exists():
                raise OSError(errno.ENOSPC, "No space left on device")
            if (control / f"bug-{stage_id}").exists():
                raise RuntimeError("unexpected")
            if (control / f"review-{stage_id}").exists() and not ctx.options.get("reviews"):
                (ctx.work_dir / "review-request.json").write_text("{}")
                return StageOutcome(("review-request.json",), {"waiting": True}, status="awaiting_review")
            upstream = "".join((ctx.dependency_dir(n) / "out.txt").read_text() for n in needs)
            (ctx.work_dir / "out.txt").write_text(f"{tag}[{upstream}]")
            return StageOutcome(("out.txt",), {"tag": tag})
        return Stage(stage_id, tuple(needs), run, (estimates or {}).get(stage_id))

    stages = [make("a", [], "A"), make("b", ["a"], "B"), make("c", ["b"], "C"), make("d", ["c"], "D")]

    def verify(binding_):
        if (control / "unmount").exists():
            raise PrerequisiteError("the volume is not mounted: marker file missing")

    plan = {"marker": plan_marker}
    import hashlib
    import json
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    campaign = Campaign(plan=plan, plan_sha256=plan_sha, plan_dir=plan_dir, run_root=run_root, stages=stages, verify_runtime=verify,
                        accelerator=False, options={"reviews": "reviews.jsonl"} if reviews else {},
                        **({"clock": clock} if clock else {}))
    return campaign, binding["manifest_sha256"]


def main(argv):
    work = Path(argv[0])
    resume = "--resume" in argv
    deadline = argv[argv.index("--deadline") + 1] if "--deadline" in argv else "2099-01-01T00:00:00Z"
    estimates = {}
    for index, token in enumerate(argv):
        if token == "--estimate":
            name, _, seconds = argv[index + 1].partition("=")
            estimates[name] = float(seconds)
    campaign, binding_sha = build(work, deadline=deadline, estimates=estimates, reviews="--reviews" in argv,
                                  plan_marker=argv[argv.index("--plan-marker") + 1] if "--plan-marker" in argv else "plan-v1")
    return campaign.run(resume=resume, initial_binding_sha256=binding_sha)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
