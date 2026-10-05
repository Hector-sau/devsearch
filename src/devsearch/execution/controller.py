"""The sequential campaign controller: stage graph, atomic commits, recovery, leases and exit codes (docs/10 §5).

Contract in one place:

* One process owns a campaign (`campaign.lock`); an accelerator profile also holds `accelerator.lease`, which
  serving shares. A second owner is refused; a stale lease is recovered only after its identity is verified.
* A stage writes its outputs into `stages/<stage>/<attempt>/`. Success is *committed* by atomically writing
  `stages/<stage>/stage-manifest.json` (plan, stage, attempt, input hashes, output hashes) and only then
  recorded in `run-state.json`. A crash before the commit leaves an ignored attempt directory. A crash after
  the commit but before the state update is repaired on resume by verifying that exact record; any mismatch
  fails instead of overwriting a committed artifact. A log line saying "succeeded" never replaces an artifact check.
* A succeeded stage is reused only while its recorded input and output hashes still verify. A corrupted
  committed artifact is a failed gate (exit 1), never a silent regeneration.
* Fatal failures (OOM, non-finite values, device loss) stop with evidence retained and are not retried or
  worked around; an interruption is resumable. Changing a recipe means a new plan, not a retry.
* Exit codes: 0 complete, 1 invalid plan/artifact, 2 usage/unsupported, 3 prerequisite unavailable,
  4 execution failure, 5 awaiting review, 6 budget stop. Shell success never means an incomplete campaign.

The controller imports no ML library: stages bring their own dependencies, so the state machine is tested
with plain fake stages.
"""

import errno
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devsearch.errors import (
    AwaitingReview, BudgetStop, ConfigError, DevSearchError, ExecutionFailure, Interrupted, ManifestError, PrerequisiteError,
)
from devsearch.execution import faults
from devsearch.execution.lock import Lease
from devsearch.execution.records import (
    EventLog, binding_chain, describe_outputs, new_attempt_id, new_state, parse_utc, read_stage_manifest, read_state, stage_manifest,
    verify_outputs, write_json, write_state,
)

EXIT_COMPLETE, EXIT_INVALID, EXIT_USAGE, EXIT_PREREQUISITE, EXIT_FAILURE, EXIT_AWAITING_REVIEW, EXIT_BUDGET = range(7)
FATAL, RESUMABLE = "fatal", "interrupted"


@dataclass(frozen=True)
class StageOutcome:
    outputs: tuple[str, ...]
    notes: Mapping[str, Any] = field(default_factory=dict)
    status: str = "succeeded"                  # or "awaiting_review"


@dataclass(frozen=True)
class Stage:
    stage_id: str
    needs: tuple[str, ...]
    run: Callable[["StageContext"], StageOutcome]
    estimate_seconds: float | None = None      # a calibrated estimate; None is "unverified", not zero


class StageContext:
    def __init__(self, campaign: "Campaign", stage: Stage, attempt_id: str, work_dir: Path) -> None:
        self.campaign, self.stage, self.attempt_id, self.work_dir = campaign, stage, attempt_id, work_dir

    @property
    def stage_id(self) -> str:
        return self.stage.stage_id

    @property
    def plan(self) -> Mapping[str, Any]:
        return self.campaign.plan

    @property
    def options(self) -> Mapping[str, Any]:
        return self.campaign.options

    @property
    def binding(self) -> Mapping[str, Any]:
        return self.campaign.active_binding

    @property
    def run_root(self) -> Path:
        return self.campaign.run_root

    def persistent_dir(self, name: str) -> Path:
        """Work that must survive crashes (training runs resume from here). Created beside the stages, inside the run root."""
        path = self.run_root / "work" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def dependency_dir(self, stage_id: str) -> Path:
        manifest = self.campaign.committed(stage_id)
        return self.campaign.stages_dir / stage_id / manifest["attempt_id"]

    def dependency_manifest(self, stage_id: str) -> dict[str, Any]:
        return self.campaign.committed(stage_id)

    def event(self, name: str, **fields: Any) -> None:
        self.campaign.events.append(name, stage_id=self.stage_id, **fields)

    def remaining_seconds(self) -> float:
        return self.campaign.remaining_seconds()

    def guard(self) -> None:
        """Re-check mount, free space and the deadline; stages call this between units of work (e.g. epochs)."""
        self.campaign.guard()

    def crash_point(self, name: str = "mid_stage") -> None:
        faults.point(name, self.stage_id)


class Campaign:
    def __init__(
        self, *, plan: Mapping[str, Any], plan_sha256: str, plan_dir: Path, run_root: Path, stages: Sequence[Stage],
        verify_runtime: Callable[[Mapping[str, Any]], None], accelerator: bool, options: Mapping[str, Any] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc), sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.plan, self.plan_sha256, self.plan_dir = plan, plan_sha256, Path(plan_dir)
        self.plan_id = "plan-" + plan_sha256[:32]
        self.run_root = Path(run_root)
        self.stages = list(stages)
        self.options = dict(options or {})
        self.verify_runtime, self.accelerator, self._clock = verify_runtime, accelerator, clock
        self.stages_dir = self.run_root / "stages"
        self.state_path = self.run_root / "run-state.json"
        self.events = EventLog(self.run_root / "events.jsonl") if self.run_root.is_dir() else None
        self.state: dict[str, Any] = {}
        self.chain: list[dict[str, Any]] = []
        ids = [s.stage_id for s in self.stages]
        if len(set(ids)) != len(ids) or any(n not in ids[:i] for i, s in enumerate(self.stages) for n in s.needs):
            raise ConfigError("the stage graph must list each stage once, after everything it needs")

    # -- helpers ---------------------------------------------------------------------------------------
    @property
    def active_binding(self) -> dict[str, Any]:
        return self.chain[-1]

    def now(self) -> str:
        return self._clock().strftime("%Y-%m-%dT%H:%M:%SZ")

    def remaining_seconds(self) -> float:
        """Seconds until the effective deadline minus the export/backup reserve (may be negative)."""
        deadline = parse_utc(self.active_binding["deadline"], "binding deadline")
        return (deadline - self._clock()).total_seconds() - float(self.active_binding["reserve_seconds"])

    def guard(self) -> None:
        self.verify_runtime(self.active_binding)
        if self.remaining_seconds() <= 0:
            raise BudgetStop("the execution deadline minus the export/backup reserve has passed; no new work starts")

    def committed(self, stage_id: str) -> dict[str, Any]:
        entry = self.state["stages"].get(stage_id)
        if entry is None or entry["status"] != "succeeded":
            raise ManifestError(f"stage {stage_id} has not succeeded")
        return read_stage_manifest(self.stages_dir / stage_id / "stage-manifest.json")

    def _save(self) -> None:
        write_state(self.state_path, self.state, now=self.now())

    def _inputs(self, stage: Stage) -> dict[str, str]:
        return {name: self.committed(name)["manifest_sha256"] for name in stage.needs}

    # -- loading and reconciliation ----------------------------------------------------------------------
    def _load_binding(self, sha256: str) -> None:
        self.chain = binding_chain(self.plan_dir, sha256)

    def _start_state(self, initial_binding_sha: str) -> None:
        self._load_binding(initial_binding_sha)
        binding = self.active_binding
        self.state = new_state(self.plan_id, self.plan_sha256, [s.stage_id for s in self.stages],
                               {"id": "binding-" + binding["manifest_sha256"][:32], "revision": binding["revision"], "sha256": binding["manifest_sha256"]},
                               now=self.now())

    def _reconcile(self) -> None:
        """Bring `run-state.json` in line with committed artifacts after a crash; verify every succeeded stage."""
        for stage in self.stages:
            entry = self.state["stages"][stage.stage_id]
            manifest_path = self.stages_dir / stage.stage_id / "stage-manifest.json"
            committed = read_stage_manifest(manifest_path) if manifest_path.is_file() else None
            if committed is not None:
                if committed["plan_id"] != self.plan_id or committed["stage_id"] != stage.stage_id:
                    raise ManifestError(f"stage {stage.stage_id}: the committed manifest belongs to another plan or stage; nothing is overwritten")
            if entry["status"] == "succeeded":
                if committed is None or committed["attempt_id"] != entry["attempt_id"] or committed["status"] != "succeeded":
                    raise ManifestError(f"stage {stage.stage_id}: run-state says succeeded but its committed manifest is missing or different")
                self._verify_committed(stage, committed)
                continue
            if committed is not None and committed["status"] == "succeeded":
                if entry["attempt_id"] != committed["attempt_id"]:
                    raise ManifestError(
                        f"stage {stage.stage_id}: a committed manifest from attempt {committed['attempt_id']} exists but run-state expects "
                        f"{entry['attempt_id']}; refusing to overwrite committed output")
                self._verify_committed(stage, committed)
                entry.update(status="succeeded", outputs=committed["outputs"], inputs=committed["inputs"], finished_at=self.now(), error=None)
                self.events.append("stage_reconstructed", stage_id=stage.stage_id, attempt_id=committed["attempt_id"])
                continue
            if entry["status"] == "failed" and (entry["error"] or {}).get("kind") == FATAL:
                raise ExecutionFailure(f"stage {stage.stage_id} failed fatally ({entry['error']['message']}); a changed recipe needs a new plan, not a retry")
            if entry["status"] == "running":
                entry.update(status="failed", error={"kind": RESUMABLE, "message": "the process stopped while this stage was running"})
                self.events.append("stage_interrupted_found", stage_id=stage.stage_id, attempt_id=entry["attempt_id"])
            if entry["attempt_id"] is not None and (committed is None or committed["attempt_id"] != entry["attempt_id"]):
                self._set_aside(stage.stage_id, entry["attempt_id"])        # an uncommitted attempt; a committed one stays referenced
        if self.state["status"] in ("failed",):
            self.state["status"] = "running"
        self._save()

    def _verify_committed(self, stage: Stage, committed: Mapping[str, Any]) -> None:
        expected_inputs = self._inputs(stage)
        if dict(committed["inputs"]) != expected_inputs:
            raise ManifestError(f"stage {stage.stage_id}: its recorded inputs no longer match the current dependency outputs; stale output is not reused")
        verify_outputs(self.stages_dir / stage.stage_id / committed["attempt_id"], committed["outputs"])

    def _set_aside(self, stage_id: str, attempt_id: str) -> None:
        directory = self.stages_dir / stage_id / attempt_id
        if directory.is_dir():
            os.rename(directory, self.stages_dir / stage_id / f".abandoned-{attempt_id}")

    # -- execution ---------------------------------------------------------------------------------------
    def run(self, *, resume: bool, initial_binding_sha256: str, active_binding_sha256: str | None = None) -> int:
        """Run (or resume) the campaign; returns the documented exit code."""
        try:
            return self._run(resume, initial_binding_sha256, active_binding_sha256)
        except (ManifestError, ConfigError) as error:
            return self._finish_error(EXIT_INVALID, error)
        except PrerequisiteError as error:
            return self._finish_error(EXIT_PREREQUISITE, error)
        except BudgetStop as error:
            return self._finish_budget(error)
        except AwaitingReview as error:
            return self._finish_awaiting(error)
        except (ExecutionFailure, Interrupted, DevSearchError) as error:
            return self._finish_error(EXIT_FAILURE, error)
        except OSError as error:
            code = EXIT_PREREQUISITE if error.errno in (errno.ENOSPC, errno.EROFS, errno.ENOENT, errno.EIO, errno.EACCES) else EXIT_FAILURE
            return self._finish_error(code, error)

    def _finish_error(self, code: int, error: BaseException) -> int:
        print(f"devsearch.run: error: {error}", file=sys.stderr)
        if self.state and self.state_path.is_file():
            self.state["last_error"] = {"message": str(error), "exit_code": code}
            if self.state["status"] == "running":
                self.state["status"] = "failed"
            self.state["next_action"] = "inspect the error; fix the prerequisite or open a new plan, then run with --resume" if code != EXIT_INVALID else \
                "an artifact or plan is invalid; resolve it, then run with --resume"
            self._save()
            self.events.append("campaign_stopped", exit_code=code, message=str(error))
        return code

    def _finish_budget(self, error: BaseException) -> int:
        print(f"devsearch.run: budget stop: {error}", file=sys.stderr)
        if self.state and self.state_path.is_file():
            self.state["status"] = "stopped_budget"
            self.state["last_error"] = {"message": str(error), "exit_code": EXIT_BUDGET}
            self.state["next_action"] = "extend the deadline with a new host-binding revision, then run with --resume"
            self._save()
            self.events.append("budget_stop", message=str(error))
        return EXIT_BUDGET

    def _finish_awaiting(self, error: BaseException) -> int:
        print(f"devsearch.run: awaiting review: {error}", file=sys.stderr)
        if self.state and self.state_path.is_file():
            self.state["status"] = "awaiting_review"
            self.state["next_action"] = "supply the reviewed pool: run with --resume --reviews <file>"
            self._save()
        return EXIT_AWAITING_REVIEW

    def _run(self, resume: bool, initial_binding_sha256: str, active_binding_sha256: str | None) -> int:
        if not self.run_root.is_dir():
            raise PrerequisiteError(f"run root {self.run_root} does not exist; the controller never creates it (a missing mount must fail, not relocate)")
        self.stages_dir.mkdir(exist_ok=True)
        self.events = self.events or EventLog(self.run_root / "events.jsonl")
        exists = self.state_path.is_file()
        if exists and not resume:
            print("devsearch.run: error: run-state.json already exists; use --resume to continue this campaign", file=sys.stderr)
            return EXIT_USAGE
        if resume and not exists:
            print("devsearch.run: error: nothing to resume: no run-state.json", file=sys.stderr)
            return EXIT_USAGE
        leases = [Lease(self.run_root / "campaign.lock", purpose="campaign", plan_id=self.plan_id)]
        if self.accelerator:
            leases.append(Lease(self.run_root / "accelerator.lease", purpose="campaign", plan_id=self.plan_id))
        acquired: list[Lease] = []
        try:
            for lease in leases:
                lease.acquire()
                acquired.append(lease)
            if exists:
                self.state = read_state(self.state_path)
                if self.state["plan_id"] != self.plan_id or self.state["plan_sha256"] != self.plan_sha256:
                    raise ManifestError("run-state.json belongs to a different plan; a changed recipe needs a new run directory")
                self._load_binding(active_binding_sha256 or self.state["active_binding"]["sha256"])
                if self.chain[0]["manifest_sha256"] != initial_binding_sha256:
                    raise ManifestError("the host-binding chain does not start at the plan's initial binding")
                self.state["active_binding"] = {"id": "binding-" + self.active_binding["manifest_sha256"][:32], "revision": self.active_binding["revision"],
                                                "sha256": self.active_binding["manifest_sha256"]}
            else:
                self._start_state(initial_binding_sha256)
            self.state["lease"] = leases[0].owner.to_record() if leases[0].owner else None
            self.state["status"] = "running"
            self.state["last_error"] = None
            self._save()
            self.events.append("campaign_started", resume=resume, binding=self.state["active_binding"]["id"])
            self.guard()
            if exists:
                self._reconcile()
            for stage in self.stages:
                entry = self.state["stages"][stage.stage_id]
                if entry["status"] == "succeeded":
                    continue
                self._execute(stage)
            self.state["status"] = "succeeded"
            self.state["next_action"] = "complete: collect the report, the release bundle and the backup receipt"
            self._save()
            self.events.append("campaign_succeeded")
            return EXIT_COMPLETE
        finally:
            for lease in reversed(acquired):
                lease.release()

    def _execute(self, stage: Stage) -> None:
        entry = self.state["stages"][stage.stage_id]
        self.guard()
        remaining = self.remaining_seconds()
        if stage.estimate_seconds is not None and stage.estimate_seconds > remaining:
            self.events.append("stage_not_started_budget", stage_id=stage.stage_id, estimate_seconds=stage.estimate_seconds, remaining_seconds=remaining)
            raise BudgetStop(f"stage {stage.stage_id} needs about {stage.estimate_seconds:.0f}s but only {remaining:.0f}s remain after the reserve")
        inputs = self._inputs(stage)
        attempt_id = new_attempt_id()
        stage_dir = self.stages_dir / stage.stage_id
        stage_dir.mkdir(exist_ok=True)
        work = stage_dir / attempt_id
        work.mkdir()
        entry.update(status="running", attempt=entry["attempt"] + 1, attempt_id=attempt_id, inputs=inputs, started_at=self.now(), finished_at=None, error=None)
        self.state["next_action"] = f"running stage {stage.stage_id}"
        self._save()
        self.events.append("stage_started", stage_id=stage.stage_id, attempt_id=attempt_id, attempt=entry["attempt"])
        started = time.monotonic()
        try:
            outcome = stage.run(StageContext(self, stage, attempt_id, work))
            faults.point("stage_outputs_written", stage.stage_id)
            outputs = describe_outputs(work, list(outcome.outputs))
            if outcome.status not in ("succeeded", "awaiting_review"):
                raise ManifestError(f"stage {stage.stage_id} returned status {outcome.status!r}")
            manifest = stage_manifest(plan_id=self.plan_id, stage_id=stage.stage_id, attempt_id=attempt_id, status=outcome.status, inputs=inputs,
                                      outputs=outputs, notes=dict(outcome.notes))
            write_json(stage_dir / "stage-manifest.json", manifest)
            faults.point("stage_manifest_committed", stage.stage_id)
        except AwaitingReview:
            self._mark(stage, "awaiting_review", None, started)
            raise
        except BudgetStop as error:
            self._mark(stage, "stopped_budget", {"kind": "budget", "message": str(error)}, started)
            raise
        except KeyboardInterrupt:
            self._mark(stage, "failed", {"kind": RESUMABLE, "message": "interrupted"}, started)
            raise Interrupted("interrupted by the operator; resume with --resume") from None
        except Interrupted as error:
            self._mark(stage, "failed", {"kind": RESUMABLE, "message": str(error)}, started)
            raise
        except (ManifestError, ConfigError) as error:                       # inputs can be restored; the stage then re-runs
            self._mark(stage, "failed", {"kind": "invalid_input", "message": str(error)}, started)
            raise
        except PrerequisiteError as error:
            self._mark(stage, "failed", {"kind": RESUMABLE, "message": str(error)}, started)
            raise
        except OSError as error:
            self._mark(stage, "failed", {"kind": RESUMABLE, "message": f"{type(error).__name__}: {error}"}, started)
            raise
        except BaseException as error:                                      # OOM, non-finite values, device loss, bugs: never retried silently
            self._mark(stage, "failed", {"kind": FATAL, "message": f"{type(error).__name__}: {error}"}, started)
            if isinstance(error, DevSearchError):
                raise
            raise ExecutionFailure(f"stage {stage.stage_id} failed: {type(error).__name__}: {error}") from error
        entry.update(status="succeeded" if outcome.status == "succeeded" else "awaiting_review", outputs=outputs,
                     finished_at=self.now(), error=None)
        self.state["next_action"] = f"stage {stage.stage_id} {entry['status']}"
        self._save()
        faults.point("state_updated", stage.stage_id)
        self.events.append("stage_" + entry["status"], stage_id=stage.stage_id, attempt_id=attempt_id,
                           seconds=round(time.monotonic() - started, 3), outputs=sorted(outputs))
        if outcome.status == "awaiting_review":
            raise AwaitingReview(f"stage {stage.stage_id} is waiting for reviewed judgments")

    def _mark(self, stage: Stage, status: str, error: Mapping[str, Any] | None, started: float) -> None:
        entry = self.state["stages"][stage.stage_id]
        entry.update(status=status, error=dict(error) if error else None, finished_at=self.now())
        self._save()
        self.events.append("stage_" + status, stage_id=stage.stage_id, attempt_id=entry["attempt_id"],
                           seconds=round(time.monotonic() - started, 3), error=dict(error) if error else None)

    # -- status ------------------------------------------------------------------------------------------
    def status_report(self) -> dict[str, Any]:
        """A verified read-only summary: nothing is trusted from the state file without checking committed artifacts."""
        state = read_state(self.state_path)
        if state["plan_id"] != self.plan_id:
            raise ManifestError("run-state.json belongs to a different plan")
        stages = {}
        for stage in self.stages:
            entry = state["stages"][stage.stage_id]
            check = "not committed"
            manifest_path = self.stages_dir / stage.stage_id / "stage-manifest.json"
            if manifest_path.is_file():
                try:
                    manifest = read_stage_manifest(manifest_path)
                    verify_outputs(self.stages_dir / stage.stage_id / manifest["attempt_id"], manifest["outputs"])
                    check = "verified"
                except ManifestError as error:
                    check = f"FAILED: {error}"
            stages[stage.stage_id] = {"status": entry["status"], "attempt": entry["attempt"], "artifact_check": check, "error": entry["error"]}
        events = self.events.read() if self.events else []
        return {
            "plan_id": state["plan_id"], "status": state["status"], "next_action": state["next_action"], "last_error": state["last_error"],
            "active_binding": state["active_binding"], "stages": stages, "events": len(events), "updated_at": state["updated_at"],
            "complete": state["status"] == "succeeded" and all(s["status"] == "succeeded" and s["artifact_check"] == "verified" for s in stages.values()),
        }


