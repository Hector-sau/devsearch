"""Glue between a plan file and a runnable `Campaign`: binding resolution, world, stages. Imports the ML stack."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devsearch.errors import ManifestError
from devsearch.execution.controller import Campaign
from devsearch.execution.hostfile import verify_runtime
from devsearch.execution.plan import load_plan
from devsearch.execution.records import read_binding
from devsearch.execution.stages import Runner
from devsearch.execution.world import World


def latest_binding(plan_dir: Path, initial_sha256: str) -> dict[str, Any]:
    """The highest-revision binding in the plan directory whose chain leads back to the plan's initial binding."""
    candidates = sorted((read_binding(p) for p in plan_dir.glob("host-binding-*.json")), key=lambda b: b["revision"])
    if not candidates:
        raise ManifestError(f"{plan_dir} holds no host-binding-*.json; the plan's host binding is missing")
    from devsearch.execution.records import binding_chain
    chosen = candidates[-1]
    chain = binding_chain(plan_dir, chosen["manifest_sha256"])
    if chain[0]["manifest_sha256"] != initial_sha256:
        raise ManifestError("the host-binding chain does not start at the binding the plan froze")
    return chosen


def open_campaign(plan_path: Path, *, reviews: Path | None = None) -> tuple[Campaign, str, str, Mapping[str, Any], Mapping[str, Any]]:
    """Returns (campaign, initial binding sha, active binding sha, plan, active binding)."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan, sha = load_plan(plan_path)
    plan_dir = plan_path.parent
    initial = plan["host_binding"]["sha256"]
    binding = latest_binding(plan_dir, initial)
    world = World(plan, binding)
    stages = Runner(plan, world).stages()
    campaign = Campaign(plan=plan, plan_sha256=sha, plan_dir=plan_dir, run_root=Path(binding["paths"]["run_root"]), stages=stages,
                        verify_runtime=verify_runtime, accelerator=bool(plan["profile"]["accelerator"]),
                        options={"reviews": str(reviews)} if reviews else {})
    return campaign, initial, binding["manifest_sha256"], plan, binding
