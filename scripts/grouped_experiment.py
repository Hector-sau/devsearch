#!/usr/bin/env python3
"""One offline, bounded, serial R6 grouped-loss campaign; see docs/16.

plan/check/status never resolve an accelerator. run requires --authorize-cuda and
an already running, otherwise idle CUDA host; this script never starts/rents one.
The persistent 45-minute deadline cannot be reset by resume. No test evaluator,
network, installer, model promotion or checkpoint deletion is reachable here.
"""

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from devsearch.artifacts.canonical import digest_of, sha256_file
from devsearch.artifacts.identity import code_identity, source_inventory
from devsearch.errors import DevSearchError, ManifestError

R6_SHA = "f9a091365f8cba75cfc8ee5e2973049692941a3114c19a0ed55f73106d773bcb"
PREPARED_SHA = "793840c54cdb166921dab1bda9bed4c05d776e485b406c02a8622e9ecb47e854"
REVISION = "44acabbec0ef496f6dbc93adadea57f376b7c0ec"
MODEL_REPO = "microsoft/MiniLM-L12-H384-uncased"
# Original GPU asset bytes (weights 8c27feb...); the CI-converted safetensors
# receipt has a different identity and cannot substitute for this campaign.
ASSETS_ID = "dea5ea7128c636c7cde9a91fe94326fc232095066727b0d712e85f1931da7c40"
ACTIVE_SECONDS, CLEANUP_SECONDS = 45 * 60, 120
MIN_FREE_BYTES = 16 * 2**30
ARMS = ("group_bce_weight4", "pairwise_logistic_scale_4_over_5")
ORDER = ((42, ARMS[0]), (42, ARMS[1]), (43, ARMS[1]), (43, ARMS[0]), (44, ARMS[0]), (44, ARMS[1]))
INPUT_NAMES = ("prepared_manifest", "pairs_manifest", "assets", "data_audit", "replacement_review", "prior_checkpoints")
POLICY = {
    "schema_version": 1, "active_seconds": ACTIVE_SECONDS, "cleanup_seconds": CLEANUP_SECONDS,
    "runs": 6, "epochs": 3, "queries": 515, "records": 2575, "updates_per_run": 387,
    "record_exposures_per_run": 7725, "warmup_updates": 39, "selection": "fixed-final-epoch-3-v1",
    "precision": "fp32", "tf32": False, "fusion_alpha": 0.5, "rank_constant": 60,
    "test_access": False, "promotion": False,
    "smoke_recovery": {"protocol": "production-epoch1-resume-v1", "query_groups": 32, "epochs": 3,
                       "updates_per_path": 24, "parameter_atol": 1e-6, "parameter_rtol": 1e-5},
    "pool_extension": "Freeze six epoch3 raw/fused rankings; union top10 with BM25, old25 systems and known-positive anchors; blind/adjudicate novel pairs before common v2.1 comparison. Anchors never enter retrieval.",
    "continuation": {"mean_mrr_delta": 0.005, "mean_ndcg_delta": 0.0, "positive_seeds": 2,
                     "mrr_ci_lower_strictly_above": 0, "bootstrap_repetitions": 10000, "bootstrap_seed": 20261008},
}


def require(condition, message):
    if not condition:
        raise ManifestError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, exclusive=False):
    path = Path(path)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(data)
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def inside(path, root):
    path = Path(path)
    require(not path.is_symlink(), f"symlink refused: {path}")
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), f"outside runtime root: {path}")
    return resolved


def source_identity():
    extra = ["scripts/grouped_experiment.py", "tests/test_grouped_experiment.py", "docs/16-grouped-loss-experiment.md",
             "locks/nvidia-cuda-cp312.lock", "src/devsearch/profiles/nvidia-cuda.toml"]
    from devsearch.modeling.devices import load_profile
    profile = load_profile("nvidia-cuda")
    validate_cuda_policy(profile)
    return {"package": code_identity(), "sources": source_inventory(),
            "runner_tests_spec_lock": {name: sha256_file(REPO / name) for name in extra},
            "resolved_cuda_profile": json.loads(json.dumps(asdict(profile)))}


def validate_cuda_policy(profile, torch_module=None):
    require(profile.name == "nvidia-cuda" and profile.device == "cuda:0" and profile.accelerator is True and
            profile.precision == POLICY["precision"] == "fp32" and profile.tf32 is POLICY["tf32"] is False,
            "resolved CUDA profile must use FP32 with TF32 disabled")
    if torch_module is not None:
        require(torch_module.get_default_dtype() == torch_module.float32 and
                torch_module.backends.cuda.matmul.allow_tf32 is False and torch_module.backends.cudnn.allow_tf32 is False,
                "actual PyTorch settings must use FP32 with CUDA/cuDNN TF32 disabled")


def config_text(seed, arm, paths):
    name = f"{arm}-seed{seed}"
    return f'''schema_version = 1
[run]
recipe_id = "{name}"
mode = "full"
seed = {seed}
epochs = 3
diagnostic = false
[optimizer]
learning_rate = 0.00005
weight_decay = 0.01
betas = [0.9, 0.999]
eps = 0.00000001
max_grad_norm = 1.0
warmup_fraction = 0.10
[batch]
effective_batch = 20
microbatch = 5
[model]
dropout = 0.1
head_seed = 1234
[loss]
pos_weight = 4
kind = "{arm}"
[validation]
batch_size = 8
[data]
prepared_manifest = {json.dumps(paths["prepared_manifest"])}
pairs_manifest = {json.dumps(paths["pairs_manifest"])}
pairs_manifest_sha256 = "{R6_SHA}"
[assets]
directory = {json.dumps(paths["assets"])}
[device]
profile = "nvidia-cuda"
[policy]
max_pair_tokens = 256
max_query_tokens = 64
'''


def retained_checkpoints(path, root, campaign, *, verify_bytes=True):
    inventory = read(path)
    rows = inventory.get("checkpoints", [])
    require(inventory.get("count") == 60 and len(rows) == 60, "exactly 60 historical checkpoints must be inventoried")
    paths = set()
    for row in rows:
        item = Path(row["path"])
        require(item.is_absolute() and item.suffix == ".pt" and not item.is_relative_to(campaign), "historical checkpoint/output overlap")
        require(item not in paths and row.get("verified") is True, "duplicate or unverified historical checkpoint")
        require(type(row.get("bytes")) is int and row["bytes"] > 0 and re.fullmatch("[a-f0-9]{64}", row.get("sha256", "")), "invalid historical receipt")
        if verify_bytes:
            item = inside(item, root)
            require(item not in paths, "duplicate resolved historical checkpoint")
            require(item.stat().st_size == row["bytes"], f"historical checkpoint size differs: {item}")
            require(sha256_file(item) == row["sha256"], f"historical checkpoint hash differs: {item}")
        paths.add(item)
    return rows


def validate_original_assets_identity(assets):
    """Require the sealed original GPU manifest, after physical verification."""
    from devsearch.artifacts.canonical import verify_manifest_hash
    verify_manifest_hash(assets, where="pinned MiniLM assets receipt")
    require(assets["manifest_sha256"] == ASSETS_ID, "exact original model asset identity required; no silent re-registration")
    require(assets["asset_kind"] == "pretrained" and assets["pinned"] is True and
            assets["source"].get("revision") == REVISION and assets["source"].get("repo") == MODEL_REPO,
            "exact pinned pretrained MiniLM assets required")


def validate_inputs(paths, root, campaign, *, verify_history=True, verify_model=True):
    # Check these exact files before following any prepared/pairs references.
    from devsearch.modeling.assets import verify_assets
    from devsearch.training.inputs import load_training_inputs
    from devsearch.training.grouped import grouped_order
    for name in INPUT_NAMES:
        inside(paths[name], root)
    prepared, pairs = Path(paths["prepared_manifest"]), Path(paths["pairs_manifest"])
    require(sha256_file(prepared) == PREPARED_SHA, "original prepared manifest differs; no resealing or adoption")
    require(sha256_file(pairs) == R6_SHA, "exact reviewed R6 manifest is required")
    audit, review = read(paths["data_audit"]), read(paths["replacement_review"])
    require(audit.get("pairs_manifest_sha256") == R6_SHA and audit.get("source_hashes", {}).get("prepared_manifest") == PREPARED_SHA,
            "R6 audit does not bind the original prepared data")
    require(audit.get("test_accessed") is False and audit.get("changed_slots") == 8 and audit.get("pool_training_owned") is True,
            "R6 audit prerequisites differ")
    require(review.get("approved") is True and review.get("data_audit_sha256") == sha256_file(Path(paths["data_audit"])),
            "replacement approval does not bind this exact R6 audit")
    inputs = load_training_inputs(prepared, root, grouped=True, pairs_manifest=pairs, pairs_manifest_sha256=R6_SHA)
    require(len(inputs.grouped_data.groups) == 515 and len(inputs.train_pairs) == 2575, "R6 query/record counts differ")
    require(len(inputs.validation.queries) == 202 and len(inputs.validation_queries) == 202, "original 202-query validation required")
    require({q.query_id for q in inputs.validation.queries} == {q.query_id for q in inputs.validation_queries}, "validation query coverage differs")
    assets = verify_assets(Path(paths["assets"])) if verify_model else read(Path(paths["assets"]) / "assets-manifest.json")
    validate_original_assets_identity(assets)
    if verify_model:
        from devsearch.modeling.tokenization import WordPieceTokenizer
        model_cfg = read(Path(paths["assets"]) / "config.json")
        require(model_cfg.get("hidden_dropout_prob") == 0.1 and model_cfg.get("attention_probs_dropout_prob") == 0.1,
                "encoder dropout must remain 0.1")
        tokenizer_identity = WordPieceTokenizer(Path(paths["assets"])).fingerprint
    else:
        tokenizer_identity = digest_of({n: assets["files"][n]["sha256"] for n in sorted(assets["tokenizer_files"])})
    schedules = {}
    for seed in (42, 43, 44):
        schedules[str(seed)] = []
        for epoch in range(3):
            groups = [inputs.grouped_data.groups[i] for i in grouped_order(inputs.grouped_data, seed, epoch)]
            schedules[str(seed)].append({"query_order_sha256": digest_of([g.query_id for g in groups]),
                                        "exposure_sha256": digest_of([p.record.pair_id for g in groups for p in g.records])})
    retained_checkpoints(paths["prior_checkpoints"], root, campaign, verify_bytes=verify_history)
    identity = {"grouped_data": inputs.grouped_data.identity, "sources": dict(inputs.grouped_data.source_sha256),
                "validation": inputs.validation.identity, "model": assets["manifest_sha256"],
                "tokenizer": tokenizer_identity, "schedules": schedules,
                "receipts": {n: sha256_file(Path(paths[n])) for n in ("data_audit", "replacement_review", "prior_checkpoints")}}
    return identity, inputs


def make_plan(args):
    root = args.runtime_root.resolve(strict=True)
    campaign = args.campaign_dir.resolve()
    require(campaign.is_relative_to(root) and not campaign.exists(), "campaign directory must be new and inside runtime root")
    paths = {name: str(getattr(args, name).resolve(strict=True)) for name in INPUT_NAMES}
    identity, _ = validate_inputs(paths, root, campaign, verify_history=not args.defer_host_checkpoints, verify_model=not args.defer_host_assets)
    require(not any(campaign.is_relative_to(Path(p)) for p in paths.values()), "output directory overlaps frozen inputs")
    plan = {"policy": POLICY, "runtime_root": str(root), "campaign_dir": str(campaign), "paths": paths,
            "source": source_identity(), "inputs": identity, "jobs": []}
    for seed, arm in ORDER:
        text = config_text(seed, arm, paths)
        name = f"{arm}-seed{seed}"
        plan["jobs"].append({"seed": seed, "arm": arm, "name": name, "config": text, "config_sha256": digest_of(text)})
    plan["plan_sha256"] = digest_of(plan)
    campaign.mkdir(parents=True)
    (campaign / "configs").mkdir()
    (campaign / "runs").mkdir()
    (campaign / "smoke").mkdir()
    for job in plan["jobs"]:
        (campaign / "configs" / (job["name"] + ".toml")).write_text(job["config"], encoding="utf-8")
        (campaign / "runs" / job["name"]).mkdir()
    write(campaign / "plan.json", plan, exclusive=True)
    return {"status": "planned_cpu_only", "plan": str(campaign / "plan.json"), "plan_sha256": plan["plan_sha256"],
            "historical_bytes_verified_on_this_host": not args.defer_host_checkpoints,
            "model_bytes_verified_on_this_host": not args.defer_host_assets,
            "data_verified": True, "actual_host_ready": False, "cuda_verified": False,
            "cuda_started": False, "next": "check on existing host; run needs separate explicit authorization"}


def load_plan(path):
    plan = read(path)
    expected = plan.pop("plan_sha256")
    require(digest_of(plan) == expected, "plan integrity mismatch")
    plan["plan_sha256"] = expected
    require(plan["policy"] == POLICY, "campaign policy differs from the fixed proposal")
    require(Path(path).resolve() == Path(plan["campaign_dir"]) / "plan.json", "plan cannot silently move between hosts/roots")
    campaign = Path(plan["campaign_dir"])
    for directory in (campaign, campaign / "configs", campaign / "runs", campaign / "smoke"):
        require(not directory.is_symlink() and directory.resolve() == directory and directory.is_dir(), "output directory symlink/drift")
    require(not any(p.is_symlink() for p in campaign.rglob("*")), "output tree contains a symlink")
    require([(j["seed"], j["arm"]) for j in plan["jobs"]] == list(ORDER), "exact six-run order required")
    for job in plan["jobs"]:
        require(job["name"] == f'{job["arm"]}-seed{job["seed"]}', "run identity differs")
        text = config_text(job["seed"], job["arm"], plan["paths"])
        require(text == job["config"] and digest_of(text) == job["config_sha256"], "frozen configuration differs")
        cfg = Path(plan["campaign_dir"]) / "configs" / (job["name"] + ".toml")
        directory = campaign / "runs" / job["name"]
        require(not cfg.is_symlink() and not directory.is_symlink() and directory.is_dir(), "run/config symlink/drift")
        require(cfg.read_text(encoding="utf-8") == text, "on-disk configuration drift")
    return plan


def check_plan(path, *, verify_history=True, verify_model=True):
    plan = load_plan(path)
    require(source_identity() == plan["source"], "source/tests/spec/lock drift; freeze a reviewed plan again before starting")
    identity, inputs = validate_inputs(plan["paths"], Path(plan["runtime_root"]), Path(plan["campaign_dir"]), verify_history=verify_history, verify_model=verify_model)
    require(identity == plan["inputs"], "input/model/tokenizer/exposure identity drift")
    return plan, inputs


def checkpoint_identity(plan, job):
    from devsearch.training.config import load_train_config
    from devsearch.training.grouped import GROUPED_ORDER, GROUPED_PROTOCOL, GROUPED_REDUCTION, GROUPED_SELECTION
    cfg = load_train_config(Path(plan["campaign_dir"]) / "configs" / (job["name"] + ".toml"))
    return {"config_sha256": cfg.config_sha256, "pairs": "2575", "pos_weight": "4.0", "validation": plan["inputs"]["validation"],
                "policy": "256/64", "pairs_manifest": R6_SHA, "tokenizer": plan["inputs"]["tokenizer"], "model": plan["inputs"]["model"],
                "code": plan["source"]["package"]["source_snapshot_sha256"], "grouped_data": plan["inputs"]["grouped_data"],
                "grouped_protocol": GROUPED_PROTOCOL, "grouped_order": GROUPED_ORDER, "grouped_reduction": GROUPED_REDUCTION,
                "checkpoint_selection": GROUPED_SELECTION}


def checkpoint_progress(plan, job):
    """Inspect complete-epoch checkpoints without unpickling; the loop rechecks tensor identity on resume."""
    directory = Path(plan["campaign_dir"]) / "runs" / job["name"]
    identity = checkpoint_identity(plan, job)
    cp = directory / "checkpoints"
    if not (cp / "latest.json").exists():
        require(not cp.exists() or not any(cp.iterdir()), "orphan checkpoint; manual integrity review required")
        return -1
    pointer = read(cp / "latest.json")
    require(re.fullmatch(r"epoch-000[0-3]", pointer.get("checkpoint_id", "")) is not None, "latest must be a complete epoch 0..3")
    completed = int(pointer["checkpoint_id"].split("-")[1])
    for epoch in range(completed + 1):
        name = f"epoch-{epoch:04d}"
        manifest = read(cp / (name + ".json"))
        tensor = inside(cp / (name + ".pt"), Path(plan["runtime_root"]))
        require(manifest.get("checkpoint_id") == name and manifest.get("file") == name + ".pt" and
                manifest.get("resume_kind") == "epoch_boundary" and manifest.get("completed_epoch") == epoch and
                manifest.get("next_epoch") == epoch and manifest.get("global_step") == epoch * 129 and
                manifest.get("identity_sha256") == digest_of(identity), "checkpoint protocol/identity differs")
        require(tensor.stat().st_size == manifest["bytes"] and sha256_file(tensor) == manifest["sha256"], "checkpoint file drift")
    require(pointer["sha256"] == manifest["sha256"], "latest pointer differs")
    if completed == 3:
        require(read(cp / "best.json") == pointer, "fixed epoch3 selection differs")
        # Completion evidence is independently committed after the trainer returned.
        if (directory / "completed.json").exists():
            receipt = read(directory / "completed.json")
            require(receipt["plan_sha256"] == plan["plan_sha256"] and receipt["checkpoint_sha256"] == pointer["sha256"], "completion receipt differs")
            expected = {"training-summary.json", "validation-epoch1.json", "validation-epoch2.json", "validation-epoch3.json"}
            require(set(receipt["outputs"]) == expected, "incomplete retained score/output list")
            for name, sha in receipt["outputs"].items():
                require(Path(name).name == name and sha256_file(directory / name) == sha, "retained score/output drift")
    elif (cp / "best.json").exists():
        raise ManifestError("best checkpoint cannot exist before epoch3")
    return completed


def installed_lock():
    require(sys.version_info[:2] == (3, 12) and platform.system() == "Linux" and platform.machine() == "x86_64",
            "run requires the existing Linux x86_64 CPython3.12 CUDA lock environment")
    pins = re.findall(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", (REPO / "locks/nvidia-cuda-cp312.lock").read_text(), re.M)
    require(bool(pins), "CUDA lock has no exact versions")
    actual = {}
    for name, expected in pins:
        actual[name] = importlib.metadata.version(name)
        require(actual[name] == expected, f"installed CUDA lock mismatch: {name} expected {expected}, got {actual[name]}")
    return actual


def remaining_seconds(state, now=None):
    require(state["deadline_unix"] - state["started_unix"] == ACTIVE_SECONDS, "persistent deadline was altered")
    now = time.time() if now is None else now
    require(now >= state["started_unix"], "clock moved backward; refuse to expand deadline")
    return max(0.0, state["deadline_unix"] - now)


def live_group_members(pgid, timeout=1):
    """Zombies cannot execute work; every other member must disappear before cleanup succeeds."""
    try:
        result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        raise ManifestError("cannot confirm process-group termination: process inspection failed") from error
    require(result.returncode == 0, "cannot confirm process-group termination: ps failed")
    live = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit() and int(fields[1]) == pgid and not fields[2].startswith("Z"):
            live.append(int(fields[0]))
    return live


def terminate_group(child, cleanup_seconds):
    end = time.monotonic() + cleanup_seconds
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            child.poll()
            require(not live_group_members(child.pid), "permission denied while terminating live process-group members")
    send(signal.SIGTERM)
    kill_reserve = min(2, cleanup_seconds / 2)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=max(0, end - time.monotonic() - kill_reserve))
    # A successful wait only reaps the leader. Descendants can ignore TERM and
    # remain in its group, so KILL is unconditional even after leader exit.
    send(signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=max(0, end - time.monotonic()))
    while time.monotonic() < end:
        child.poll()
        if not live_group_members(child.pid, timeout=max(.01, min(1, end - time.monotonic()))):
            return
        time.sleep(min(.025, max(0, end - time.monotonic())))
    raise ManifestError("cleanup deadline exhausted; process-group termination was not confirmed")


def supervise(command, log, remaining, *, cleanup_seconds=CLEANUP_SECONDS):
    """Bound all child work including imports/model loading/scoring/saving; kill its process group."""
    require(remaining > 0, "campaign deadline exhausted")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(REPO / "src")}
    child = None
    with Path(log).open("ab") as handle:
        try:
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            return child.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            signal.setitimer(signal.ITIMER_REAL, 0)  # cleanup gets its one bounded allowance, without a second alarm
            if child is not None:
                terminate_group(child, cleanup_seconds)
            return 6


@contextlib.contextmanager
def deadline_guard(seconds):
    """Also bound parent-side revalidation/hashing, not only GPU children."""
    def interrupt(*_):
        raise KeyboardInterrupt("campaign deadline or termination signal")
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def completed_job(plan, job):
    progress = checkpoint_progress(plan, job)
    return progress == 3 and (Path(plan["campaign_dir"]) / "runs" / job["name"] / "completed.json").exists()


def fit_estimate(smoke, remaining):
    # Worst arm, twice measured cost: 18 full validations and 24 saves (including epoch0).
    require(len(smoke) == 2 and {x.get("arm") for x in smoke} == set(ARMS) and
            all(x["updates"] == 24 and x.get("recovery_passed") is True and
                x.get("recovery_protocol") == POLICY["smoke_recovery"]["protocol"] for x in smoke),
            "both 24-update production recovery smoke probes must pass")
    estimate = 2 * (max(x["train_seconds"] / 24 for x in smoke) * 2322 +
                    max(x["validation_seconds"] for x in smoke) * 18 + max(x["save_seconds"] for x in smoke) * 24 +
                    max(x["load_seconds"] for x in smoke) * 6) + 60
    require(math.isfinite(estimate) and estimate < remaining, "measured smoke cannot fit six runs in remaining deadline; stop without launching")
    return estimate


def compare_recovery_states(continuous, resumed):
    import torch
    tolerances = POLICY["smoke_recovery"]
    def compare(a, b, name):
        if isinstance(a, torch.Tensor):
            require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, f"recovery tensor structure mismatch: {name}")
            if a.is_floating_point() and not name.startswith("rng"):
                require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and
                        torch.allclose(a, b, atol=tolerances["parameter_atol"], rtol=tolerances["parameter_rtol"]),
                        f"recovery parity mismatch: {name}")
            else:
                require(torch.equal(a, b), f"recovery state mismatch: {name}")
        elif isinstance(a, dict):
            require(isinstance(b, dict) and a.keys() == b.keys(), f"recovery keys differ: {name}")
            for key in a:
                compare(a[key], b[key], f"{name}.{key}")
        elif isinstance(a, (list, tuple)):
            require(isinstance(b, type(a)) and len(a) == len(b), f"recovery sequence differs: {name}")
            for index, (left, right) in enumerate(zip(a, b)):
                compare(left, right, f"{name}.{index}")
        else:
            require(a == b, f"recovery value differs: {name}")
    for key in ("model", "optimizer", "scheduler", "rng", "global_step", "completed_epoch", "next_epoch", "lr_trace", "identity"):
        compare(continuous[key], resumed[key], key)


def recovery_smoke(*, config, data, validation, tokenizer, policy, device, fresh, directory, identities, stopping, deadline):
    """Nonformal subset: production 3-epoch train versus epoch1 stop + fresh-process-state resume.

    Real weights/backend are supplied by worker. CPU tests use this same helper
    with original tiny fixtures; its output cannot satisfy a CUDA receipt alone.
    """
    import torch
    from devsearch.training import loop
    from devsearch.training.checkpoint import load_checkpoint
    directory.mkdir()
    continuous_dir, resumed_dir = directory / "continuous", directory / "resumed"
    continuous_dir.mkdir()
    resumed_dir.mkdir()
    write(directory / "probe.json", {"formal": False, "kind": "production-checkpoint-recovery-probe", "query_groups": len(data.groups),
                                    "grouped_data": data.identity, "config": config.to_dict(), "policy": POLICY["smoke_recovery"]})
    loads = []
    events = []
    def model_load():
        start = time.monotonic()
        model = fresh(config.seed)
        loads.append(time.monotonic() - start)
        return model
    def run(model, destination, **kwargs):
        return loop.train(config=config, model=model, tokenizer=tokenizer, policy=policy, pairs=data.pairs,
                          grouped_data=data, validation=validation, device=device, run_dir=destination, identities=identities,
                          should_stop=stopping, deadline_monotonic=deadline, on_event=events.append, **kwargs)
    model = model_load()
    continuous = run(model, continuous_dir)
    updates = math.ceil(len(data.groups) / 4) * 3
    require(continuous.status == "completed" and continuous.global_step == updates, "continuous recovery probe incomplete")
    baseline, _ = load_checkpoint(continuous_dir, "epoch-0003", torch.device("cpu"), require_resumable=True)
    del model
    model = model_load()
    interrupted = run(model, resumed_dir, epoch_gate=lambda epoch, _: "intentional recovery probe boundary" if epoch == 1 else None)
    require(interrupted.completed_epochs == 1 and interrupted.global_step == updates // 3 and interrupted.status == "stopped_budget",
            "recovery probe did not stop at the exact epoch1 boundary")
    boundary, boundary_manifest = load_checkpoint(resumed_dir, "epoch-0001", torch.device("cpu"), require_resumable=True)
    require(all(k in boundary for k in ("optimizer", "scheduler", "rng", "model")), "production checkpoint lacks recovery state")
    del boundary, model
    model = model_load()
    # A reload must restore saved stochastic state, not accidentally inherit the
    # correct stream from an uninterrupted object/process.
    torch.manual_seed(987654321)
    start = time.monotonic()
    resumed = run(model, resumed_dir, resume=True)
    elapsed = time.monotonic() - start
    require(resumed.status == "completed" and resumed.global_step == updates and resumed.completed_epochs == 3 and
            any(e.get("event") == "resumed" and e.get("from_epoch") == 1 and e.get("global_step") == updates // 3 for e in events),
            "production resume did not restore counters and finish the remaining epochs")
    restored, final_manifest = load_checkpoint(resumed_dir, "epoch-0003", torch.device("cpu"), require_resumable=True)
    compare_recovery_states(baseline, restored)
    continuation_seconds = sum(sum(h["seconds"].values()) for h in resumed.history[1:])
    loads.append(max(0.0, elapsed - continuation_seconds))  # includes production checkpoint loading/restoration
    report = {"arm": config.loss_kind, "updates": updates, "train_seconds": sum(h["seconds"]["train"] for h in continuous.history),
              "save_seconds": max(h["seconds"]["save"] for h in continuous.history), "load_seconds": max(loads),
              "recovery_passed": True, "recovery_protocol": POLICY["smoke_recovery"]["protocol"], "formal": False,
              "resumed_from_epoch": 1, "resumed_from_step": updates // 3, "final_step": resumed.global_step,
              "boundary_checkpoint_sha256": boundary_manifest["sha256"], "final_checkpoint_sha256": final_manifest["sha256"],
              "parity": ["model", "optimizer", "scheduler", "rng", "steps", "learning_rate_trace"],
              "atol": POLICY["smoke_recovery"]["parameter_atol"], "rtol": POLICY["smoke_recovery"]["parameter_rtol"]}
    write(directory / "recovery.json", report)
    return report, model


def verify_run_orders(result, inputs, plan, job):
    ids = [p.record.pair_id for group in inputs.grouped_data.groups for p in group.records]
    for epoch, order in result.orders.items():
        require(type(epoch) is int and 0 <= epoch < 3 and sorted(order) == list(range(2575)), "runtime exposure count/order differs")
        expected = plan["inputs"]["schedules"][str(job["seed"])][epoch]["exposure_sha256"]
        require(digest_of([ids[i] for i in order]) == expected, "runtime exposure hash differs from the matched frozen schedule")


def run_campaign(path, authorized=False):
    require(authorized, "run requires explicit --authorize-cuda after separate user authorization")
    plan, _ = check_plan(path)
    root, campaign = Path(plan["runtime_root"]), Path(plan["campaign_dir"])
    versions = installed_lock()  # no device resolution before this gate
    require(shutil.disk_usage(campaign).free >= MIN_FREE_BYTES, "at least 16 GiB free space is required for preserved outputs")
    # Match the existing gpu.py outer lock; refuse an old campaign accelerator lease too.
    require((root / "state").is_dir(), "existing GPU runtime state directory required")
    require(not (root / "campaign/run/accelerator.lease").exists(), "existing accelerator lease must be resolved by its owner")
    with (root / "state/gpu.lock").open("a") as host_lock, (campaign / "runner.lock").open("a") as campaign_lock:
        for handle in (host_lock, campaign_lock):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ManifestError("another campaign owns the runtime/campaign lock") from error
        state_path = campaign / "state.json"
        if state_path.exists():
            state = read(state_path)
            require(state["plan_sha256"] == plan["plan_sha256"], "state plan mismatch")
            if state["status"] == "completed":
                require(all(completed_job(plan, j) for j in plan["jobs"]), "incomplete completed campaign")
                return state
            require(state["status"] in ("running", "interrupted"), "failed/exhausted campaign cannot silently restart")
        else:
            now = time.time()
            state = {"plan_sha256": plan["plan_sha256"], "started_unix": now, "deadline_unix": now + ACTIVE_SECONDS,
                     "status": "running", "environment": versions, "completed": []}
            write(state_path, state, exclusive=True)
        remaining_seconds(state)
        tasks = [("smoke", None)] + [("train", j) for j in plan["jobs"]]
        try:
            guard = deadline_guard(remaining_seconds(state))
            guard.__enter__()
            for phase, job in tasks:
                require(remaining_seconds(state) > 0, "campaign deadline exhausted")
                if phase == "smoke" and (campaign / "smoke/completed.json").exists():
                    receipt = read(campaign / "smoke/completed.json")
                    require(receipt["plan_sha256"] == plan["plan_sha256"], "smoke identity differs")
                    fit_estimate(receipt["arms"], ACTIVE_SECONDS)
                    continue
                if job and completed_job(plan, job):
                    continue
                check_plan(path, verify_history=False)
                name = job["name"] if job else "smoke"
                state.update(status="running", current=name, supervisor_pid=os.getpid())
                write(state_path, state)
                result = supervise([sys.executable, str(Path(__file__).resolve()), "_worker", "--plan", str(path),
                                    "--task", name], campaign / (name + ".log"), remaining_seconds(state))
                if result != 0:
                    state.update(status="interrupted" if result == 6 and remaining_seconds(state) > 0 else "stopped_budget" if result == 6 else "failed",
                                 child_exit=result)
                    write(state_path, state)
                    return state
                if phase == "smoke":
                    receipt = read(campaign / "smoke/completed.json")
                    state["estimated_remaining_seconds"] = fit_estimate(receipt["arms"], remaining_seconds(state))
                else:
                    require(completed_job(plan, job), "child did not complete epoch3 and retained scoring")
                    state["completed"] = [j["name"] for j in plan["jobs"] if completed_job(plan, j)]
                write(state_path, state)
            retained_checkpoints(plan["paths"]["prior_checkpoints"], root, campaign)
            require(remaining_seconds(state) > 0, "campaign deadline exhausted during integrity checks")
            state.update(status="completed", quality_claim=False, promoted=False, test_accessed=False,
                         next="Freeze outputs and finish blind v2.1 shared-pool review before comparing quality.")
            write(state_path, state)
            return state
        except BaseException as error:
            state.update(status=("stopped_budget" if remaining_seconds(state) <= 0 else "interrupted") if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            write(state_path, state)
            raise
        finally:
            guard.__exit__(None, None, None)


def score_validation(model, tokenizer, policy, inputs, device, path, stopping):
    """Persist all raw and fixed-RRF rankings; no label appears in the score artifact."""
    from devsearch.modeling.scoring import score_pairs
    from devsearch.evaluation.ranking import order_by_score, ranked_record, score_with_positives
    validation = inputs.validation
    pairs = [(q.text, text) for q in validation.queries for text in q.candidate_texts]
    scores = score_pairs(model, tokenizer, policy, pairs, device=device, batch_size=8, should_stop=stopping)
    raw, fused, cursor = [], [], 0
    candidates = {q["query_id"]: q for q in inputs.validation_candidates}
    for q in validation.queries:
        count = len(q.candidate_ids)
        ordering = order_by_score(q.candidate_ids, scores[cursor:cursor + count])
        cursor += count
        candidate = candidates[q.query_id]
        raw.append(ranked_record(q.query_id, candidate["candidate_set_id"], ordering))
        bm25 = {c["chunk_id"]: c["rank"] for c in candidate["candidates"]}
        combined = [0.5 / (60 + rank) + 0.5 / (60 + bm25[cid]) for cid, rank, _ in ordering]
        fused.append(ranked_record(q.query_id, candidate["candidate_set_id"], order_by_score([c for c, _, _ in ordering], combined)))
    write(path, {"split": "validation", "label_blind": True, "alpha": 0.5, "rank_constant": 60, "raw": raw, "fused": fused})
    metrics = score_with_positives(raw, {q.query_id: q.positives for q in validation.queries}, validation.group_of)
    return {k: metrics[k] for k in ("group_macro", "query_macro", "queries_scored", "groups_scored")}


def worker(path, task):
    # The supervisor owns the deadline and process group. An internal invocation
    # without its live parent is refused before importing/initializing CUDA.
    plan, inputs = check_plan(path, verify_history=False)
    campaign = Path(plan["campaign_dir"])
    state = read(campaign / "state.json")
    require(state.get("supervisor_pid") == os.getppid(), "worker requires its live supervisor parent")
    require(state["status"] == "running" and state.get("current") == task and state["plan_sha256"] == plan["plan_sha256"], "worker is not the current supervised task")
    require(remaining_seconds(state) > 0, "campaign deadline exhausted")
    # Probe nvidia-smi before our process creates its own CUDA context.
    probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    require(probe.returncode == 0 and not probe.stdout.strip(), "GPU is unavailable or another compute process is active")
    import torch
    from devsearch.modeling.assets import load_encoder
    from devsearch.modeling.cross_encoder import CrossEncoder
    from devsearch.modeling.devices import resolve_device, load_profile, backend_fingerprint
    from devsearch.modeling.policy import PairEncodingPolicy
    from devsearch.modeling.tokenization import WordPieceTokenizer
    from devsearch.training.job import load_train_job
    from devsearch.training import loop
    profile = load_profile("nvidia-cuda")
    require(json.loads(json.dumps(asdict(profile))) == plan["source"]["resolved_cuda_profile"], "resolved CUDA profile differs from frozen plan")
    validate_cuda_policy(profile)
    device = resolve_device(profile)
    validate_cuda_policy(profile, torch)
    require(torch.cuda.device_count() == 1, "exactly one visible CUDA device required")
    fingerprint = backend_fingerprint(profile, device)
    if task != "smoke":
        require(read(campaign / "smoke/completed.json")["device"] == fingerprint, "CUDA hardware/software fingerprint changed since smoke")
    tokenizer, policy = WordPieceTokenizer(Path(plan["paths"]["assets"])), PairEncodingPolicy(256, 64)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    deadline = time.monotonic() + remaining_seconds(state)
    stopping = lambda: stop[0] or time.monotonic() >= deadline

    def fresh(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CrossEncoder(load_encoder(Path(plan["paths"]["assets"])), dropout=0.1, head_seed=1234)
        require(all(p.dtype == torch.float32 for p in model.parameters()), "all model parameters must be FP32")
        return model

    if task == "smoke":
        from devsearch.training.data import GroupedTrainSet, ValidationSet
        reports = []
        # Longest groups exercise padding/memory; both arms use the same probe.
        groups = tuple(sorted(inputs.grouped_data.groups, key=lambda g: (-sum(len(x.pair.passage_text) for x in g.records), g.query_id))[:32])
        require(len(groups) == 32, "smoke requires exactly 32 R6 query groups")
        subset = GroupedTrainSet(groups, inputs.grouped_data.source_sha256)
        validation_subset = ValidationSet(inputs.validation.queries[:2], ())
        for arm in ARMS:
            job = next(j for j in plan["jobs"] if j["arm"] == arm and j["seed"] == 42)
            cfg = load_train_job(campaign / "configs" / (job["name"] + ".toml")).config
            cfg = replace(cfg, recipe_id="smoke-recovery-" + arm)
            report, model = recovery_smoke(config=cfg, data=subset, validation=validation_subset, tokenizer=tokenizer, policy=policy,
                                           device=device, fresh=fresh, directory=campaign / "smoke" / arm,
                                           identities={"pairs_manifest": R6_SHA, "model": plan["inputs"]["model"],
                                                       "code": plan["source"]["package"]["source_snapshot_sha256"],
                                                       "probe": "nonformal-production-recovery-v1"}, stopping=stopping, deadline=deadline)
            started = time.monotonic()
            score_validation(model, tokenizer, policy, inputs, device, campaign / "smoke" / (arm + "-scores.json"), stopping)
            report["validation_seconds"] = time.monotonic() - started
            reports.append(report)
            del model
            torch.cuda.empty_cache()
        estimate = fit_estimate(reports, max(0, deadline - time.monotonic()))
        write(campaign / "smoke/completed.json", {"plan_sha256": plan["plan_sha256"], "arms": reports, "estimate_seconds": estimate, "device": fingerprint})
        return 0
    job = next((j for j in plan["jobs"] if j["name"] == task), None)
    require(job is not None, "unknown worker task")
    cfg_path = campaign / "configs" / (task + ".toml")
    config = load_train_job(cfg_path).config
    directory = campaign / "runs" / task
    progress = checkpoint_progress(plan, job)
    model = fresh(job["seed"])
    epoch = [max(0, progress)]
    def validation_hook(model, tokenizer, policy, validation, device, batch_size):
        epoch[0] += 1
        return score_validation(model, tokenizer, policy, inputs, device, directory / f"validation-epoch{epoch[0]}.json", stopping)
    loop.evaluate_validation = validation_hook  # child-local output hook; shared scoring and metrics, unchanged training loop
    def emit(event):
        with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
    result = loop.train(config=config, model=model, tokenizer=tokenizer, policy=policy, pairs=inputs.train_pairs,
                        validation=inputs.validation, device=device, run_dir=directory, grouped_data=inputs.grouped_data,
                        identities={"pairs_manifest": R6_SHA, "model": plan["inputs"]["model"], "tokenizer": plan["inputs"]["tokenizer"],
                                    "code": plan["source"]["package"]["source_snapshot_sha256"]}, resume=progress >= 0,
                        should_stop=stopping, deadline_monotonic=deadline, on_event=emit)
    require(result.status == "completed" and result.completed_epochs == 3 and result.best_epoch == 3 and result.global_step == 387,
            "training did not complete fixed epoch3")
    verify_run_orders(result, inputs, plan, job)
    pointer = read(directory / "checkpoints/latest.json")
    outputs = {name: sha256_file(directory / name) for name in ("training-summary.json", "validation-epoch1.json", "validation-epoch2.json", "validation-epoch3.json")}
    write(directory / "completed.json", {"plan_sha256": plan["plan_sha256"], "checkpoint_sha256": pointer["sha256"], "outputs": outputs,
                                          "device": fingerprint, "selection": "epoch3", "test_accessed": False, "quality_claim": False,
                                          "schedule": plan["inputs"]["schedules"][str(job["seed"])]})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("plan", help="verify frozen actual inputs and create six configs; CPU only")
    for name in ("runtime_root", "campaign_dir", *INPUT_NAMES):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--defer-host-checkpoints", action="store_true", help="verify the historical receipt only; run still requires all 60 actual target files")
    p.add_argument("--defer-host-assets", action="store_true", help="verify exact sealed model receipt only; run still requires physical model/tokenizer files")
    for phase in ("check", "run", "status", "_worker"):
        p = sub.add_parser(phase)
        p.add_argument("--plan", type=Path, required=True)
        if phase == "run":
            p.add_argument("--authorize-cuda", action="store_true")
        if phase == "_worker":
            p.add_argument("--task", required=True)
        if phase == "check":
            p.add_argument("--defer-host-checkpoints", action="store_true")
            p.add_argument("--defer-host-assets", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.phase == "plan":
            result = make_plan(args)
        elif args.phase == "check":
            plan, _ = check_plan(args.plan, verify_history=not args.defer_host_checkpoints, verify_model=not args.defer_host_assets)
            result = {"status": "cpu_input_checks_passed", "plan_sha256": plan["plan_sha256"], "cuda_started": False,
                      "historical_bytes_verified_on_this_host": not args.defer_host_checkpoints,
                      "model_bytes_verified_on_this_host": not args.defer_host_assets,
                      "data_verified": True, "actual_host_ready": False, "cuda_verified": False,
                      "remaining_gates": ["explicit CUDA authorization", "existing locked host/environment", "free disk/idle device", "both real-model smoke arms", "v2.1 blind review before any quality claim"]}
        elif args.phase == "run":
            result = run_campaign(args.plan, args.authorize_cuda)
        elif args.phase == "_worker":
            return worker(args.plan, args.task)
        else:
            plan = load_plan(args.plan)
            state_path = Path(plan["campaign_dir"]) / "state.json"
            result = {"state": read(state_path) if state_path.exists() else {"status": "not_started"},
                      "runs": [{"name": j["name"], "completed_epoch": checkpoint_progress(plan, j)} for j in plan["jobs"]],
                      "test_accessed": False, "quality_claim": False}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") not in ("failed", "interrupted", "stopped_budget") else 6 if result["status"] == "stopped_budget" else 4
    except (DevSearchError, OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        print(f"grouped_experiment: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("grouped_experiment: interrupted; inspect the preserved state and deadline", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
