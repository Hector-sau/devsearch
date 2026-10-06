#!/usr/bin/env python3
"""DevSearch on a rented NVIDIA GPU server: one entry point, separate stages (docs/GPU_QUICKSTART.md).

    python3 scripts/gpu.py check    --config configs/gpu.toml   host, driver, storage, permissions, network, inputs
    python3 scripts/gpu.py prepare  --config configs/gpu.toml   environment, receipts, pinned model, sources, data, review bank, bundle
    python3 scripts/gpu.py diagnose --config configs/gpu.toml   small real-model training, inference, crash and resume check
    python3 scripts/gpu.py run      --config configs/gpu.toml   the declared campaign: matrix, validation-only selection, sealed
                                                                test evaluation, export, reload verification, backup
    python3 scripts/gpu.py run      --config configs/gpu.toml --extend-hours 6   after a time-limit stop: new deadline, then resume
    python3 scripts/gpu.py status   --config configs/gpu.toml   what is done, what is next, where the outputs are
    python3 scripts/gpu.py cmd      --config configs/gpu.toml -- review progress ...   any devsearch command, right interpreter and code
    python3 scripts/gpu.py env-python --config configs/gpu.toml prints the interpreter of the prepared environment
    python3 scripts/gpu.py rehearsal-judge --config C           CPU rehearsal only: two scripted judgment passes (refused on a GPU profile)
    python3 scripts/gpu.py fetch-model --revision COMMIT --output DIR   any connected machine: download + verify the pinned model
    python3 scripts/gpu.py standin-model --package P --output DIR      CPU rehearsal only: tiny random encoder, reference vocabulary
    python3 scripts/gpu.py package  ...                         authoring tool: build the dataset package (connected machine)

Every stage is resumable: a finished step leaves a record under `<root>/state/` and is skipped next time; an interrupted step
is redone from its last committed output (downloads keep verified files, the campaign resumes from its last committed stage).
Everything large (environment, wheelhouse, model, sources, data, checkpoints, logs) goes under the configured runtime root;
nothing is written into the checkout. Logs: `<root>/logs/gpu.log` (one line per step) and one full log per command.

Nothing is changed silently. The first `prepare` freezes the backend profile, model repository and commit, dataset package,
dependency lock and protocol in `<root>/state/frozen.json`; a later configuration that differs in any of them is refused (exit 1).
A missing credential, input or review artifact stops with a line `PREREQUISITE <name>: <what to do>` and exit 3; nothing is
substituted and no output is faked.

Exit: 0 done; 1 invalid or refused (configuration, integrity, frozen identity); 2 usage; 3 named prerequisite missing;
4 a step failed or the campaign is incomplete; 5 the campaign awaits review; 6 the time limit stopped the campaign.

Standard library only (Python 3.11+, for `tomllib`); the work itself runs in the prepared environment.
"""

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMMIT = re.compile(r"^[0-9a-f]{40}$")
RESOLVE_ONCE = "resolve-main-once"
TOKENIZER_FILES = ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
PACKAGE_DIR = "devsearch-dataset"
FETCH_MANIFEST = "fetch-manifest.json"
PROFILES = ("nvidia-cuda", "apple-mps", "cpu-diagnostic")
DIAGNOSE_FAULT = "mid_stage:train:full:42"
EXIT = {"ok": 0, "invalid": 1, "usage": 2, "prerequisite": 3, "failed": 4, "review": 5, "budget": 6}


class Stop(Exception):
    """Ends the stage with an exit code and a message (a prerequisite carries its name)."""

    def __init__(self, code: int, message: str, prerequisite: str | None = None):
        super().__init__(message)
        self.code, self.prerequisite = code, prerequisite


def prerequisite(name: str, message: str) -> Stop:
    return Stop(EXIT["prerequisite"], message, name)


def utc(moment: datetime | None = None) -> str:
    return (moment or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------- configuration

SCHEMA = {
    "runtime": ({"root", "time_limit_hours"}, {"mount", "require_mount_point", "volume_id", "min_free_gib", "step_timeout_minutes"}),
    "backend": ({"profile"}, {"min_cuda", "min_gpu_memory_gib", "device_memory_fraction"}),
    "environment": ({"mode"}, {"python", "lock"}),
    "model": ({"repo", "revision", "license", "source"}, {"endpoint", "local_dir", "kind", "token_env"}),
    "dataset": ({"package", "sha256"}, {"url"}),
    "review": (set(), {"min_primary_reviews", "batch_size"}),
    "backup": (set(), {"dir", "require_independent_device"}),
    "receipts": (set(), {"record"}),
}


class Config:
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        try:
            raw = tomllib.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise Stop(EXIT["invalid"], f"{self.path}: {error}")
        if raw.get("schema_version") != 1:
            raise Stop(EXIT["invalid"], f"{self.path}: schema_version must be 1")
        unknown = set(raw) - set(SCHEMA) - {"schema_version"}
        if unknown:
            raise Stop(EXIT["invalid"], f"{self.path}: unknown sections {sorted(unknown)}")
        for section, (required, optional) in SCHEMA.items():
            table = raw.get(section, {})
            if not isinstance(table, dict):
                raise Stop(EXIT["invalid"], f"{self.path}: [{section}] must be a table")
            missing, extra = required - set(table), set(table) - required - optional
            if missing or extra:
                raise Stop(EXIT["invalid"], f"{self.path}: [{section}] missing {sorted(missing)} unknown {sorted(extra)}")
        self.raw = raw
        r, b, e, m, d = (raw[s] for s in ("runtime", "backend", "environment", "model", "dataset"))
        self.root = Path(r["root"])
        if not self.root.is_absolute():
            raise Stop(EXIT["invalid"], "runtime.root must be an absolute path on the persistent volume")
        self.mount = Path(r["mount"]) if r.get("mount") else None
        self.require_mount_point = bool(r.get("require_mount_point", False))
        self.volume_id = str(r.get("volume_id", "devsearch-runtime"))
        self.min_free_bytes = int(float(r.get("min_free_gib", 60)) * 2**30)
        self.time_limit_hours = float(r["time_limit_hours"])
        self.step_timeout = int(float(r.get("step_timeout_minutes", 240)) * 60)
        if self.time_limit_hours <= 0 or self.step_timeout <= 0:
            raise Stop(EXIT["invalid"], "runtime.time_limit_hours and step_timeout_minutes must be positive")
        self.profile = b["profile"]
        if self.profile not in PROFILES:
            raise Stop(EXIT["invalid"], f"backend.profile must be one of {PROFILES}")
        self.min_cuda = str(b.get("min_cuda", "13.0"))
        self.min_gpu_memory = int(float(b.get("min_gpu_memory_gib", 16)) * 2**30)
        self.device_memory_fraction = float(b.get("device_memory_fraction", 0.9))
        self.env_mode = e["mode"]
        if self.env_mode not in ("lock", "current"):
            raise Stop(EXIT["invalid"], "environment.mode is 'lock' (a new environment from the hashed lock) or 'current' (CPU fixture tests only)")
        self.python = e.get("python", "python3.12")
        self.lock = (REPO / e["lock"]).resolve() if e.get("lock") else None
        if self.env_mode == "lock" and (self.lock is None or not self.lock.is_file()):
            raise Stop(EXIT["invalid"], "environment.lock must name an existing lock file in the checkout for mode 'lock'")
        self.model_repo, self.model_revision, self.model_license = m["repo"], m["revision"], m["license"]
        self.model_source, self.model_kind = m["source"], m.get("kind", "pretrained")
        self.endpoint = m.get("endpoint", "https://huggingface.co").rstrip("/")
        self.local_dir = Path(m["local_dir"]) if m.get("local_dir") else None
        self.token_env = m.get("token_env", "HF_TOKEN")
        if self.model_source not in ("huggingface", "local") or self.model_kind not in ("pretrained", "diagnostic"):
            raise Stop(EXIT["invalid"], "model.source is 'huggingface' or 'local'; model.kind is 'pretrained' or 'diagnostic'")
        if self.model_revision != RESOLVE_ONCE and not COMMIT.match(self.model_revision):
            raise Stop(EXIT["invalid"], f"model.revision must be a full 40-hex commit or '{RESOLVE_ONCE}' (a branch name is never a pin)")
        if self.model_source == "local" and (self.local_dir is None or not COMMIT.match(self.model_revision)):
            raise Stop(EXIT["invalid"], "model.source 'local' needs model.local_dir and a 40-hex model.revision")
        package = Path(d["package"])
        self.package = package if package.is_absolute() else self.root / package
        self.package_sha256 = d["sha256"]
        self.package_url = d.get("url", "")
        if not re.fullmatch(r"[0-9a-f]{64}", self.package_sha256):
            raise Stop(EXIT["invalid"], "dataset.sha256 must be the 64-hex SHA-256 of the dataset package")
        review = raw.get("review", {})
        self.min_primary_reviews = int(review.get("min_primary_reviews", 2))
        self.review_batch_size = int(review.get("batch_size", 40))
        backup = raw.get("backup", {})
        self.backup_dir = Path(backup["dir"]) if backup.get("dir") else self.root / "backup"
        self.require_independent_backup = bool(backup.get("require_independent_device", False))
        self.record_receipts = bool(raw.get("receipts", {}).get("record", True))


# ---------------------------------------------------------------------------------------------------------------- runtime

class Runtime:
    """Paths, the stage lock, logging, step records and the subprocess runner for one runtime root."""

    def __init__(self, config: Config):
        self.config = config
        self.root = config.root
        self.state = self.root / "state"
        self.logs = self.root / "logs"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @contextlib.contextmanager
    def locked(self):
        self.state.mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        handle = (self.state / "gpu.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Stop(EXIT["prerequisite"], "another gpu.py stage holds this runtime root; wait for it or stop it first", "runtime-lock")
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def log(self, message: str) -> None:
        line = f"{utc()} {message}"
        print(line, file=sys.stderr, flush=True)                   # stdout stays machine-readable (status JSON, env-python)
        if self.logs.is_dir():
            with (self.logs / "gpu.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def done(self, step: str):
        path = self.state / f"{step}.json"
        return read_json(path) if path.is_file() else None

    def record(self, step: str, **fields) -> dict:
        record = {"step": step, "finished_at": utc(), **fields}
        write_json(self.state / f"{step}.json", record)
        return record

    # ---- environment of the work processes
    def code_key(self) -> str:
        return code_key()

    def env_python(self) -> Path:
        if self.config.env_mode == "current":
            return Path(sys.executable)
        return self.path("envs", self.code_key(), "bin", "python")

    def child_env(self, extra: dict | None = None) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("DEVSEARCH_FAULT") and k != "PYTHONPATH"}
        env.update({"DEVSEARCH_RUNTIME_ROOT": str(self.root), "TMPDIR": str(self.path("tmp")), "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        if self.config.env_mode == "current":
            env["PYTHONPATH"] = str(REPO / "src")
        for key, value in (extra or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return env

    def run(self, name: str, argv: list, *, ok=(0,), env_extra: dict | None = None, timeout: int | None = None, cwd: Path | None = None):
        """Run one command with its full output in its own log file; returns the CompletedProcess. Raises Stop on a code outside `ok`."""
        self.path("tmp").mkdir(exist_ok=True)
        self.logs.mkdir(exist_ok=True)
        argv = [str(a) for a in argv]
        log = self.logs / f"{utc().replace(':', '')}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', name)}.log"
        started = time.monotonic()
        self.log(f"run {name}: {' '.join(argv)}")
        limit = timeout or self.config.step_timeout
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self.child_env(env_extra), cwd=cwd or self.root,
                                   start_new_session=True)              # its own process group: a timeout or Ctrl-C stops grandchildren too
        try:
            stdout, stderr = process.communicate(timeout=limit)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            log.write_text(f"$ {' '.join(argv)}\nSTOPPED ({type(error).__name__}) after {time.monotonic() - started:.0f}s\n{stdout}\n{stderr}", encoding="utf-8")
            if isinstance(error, KeyboardInterrupt):
                raise
            raise Stop(EXIT["failed"], f"{name} exceeded its time limit ({limit}s) and was stopped with its child processes; log: {log}")
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        log.write_text(f"$ {' '.join(argv)}\nexit {result.returncode} after {time.monotonic() - started:.1f}s\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}",
                       encoding="utf-8")
        self.log(f"exit {result.returncode} {name} ({time.monotonic() - started:.0f}s) log: {log.name}")
        if result.returncode not in ok:
            tail = (result.stderr.strip() or result.stdout.strip())[-1200:]
            code = result.returncode if result.returncode in (3, 4, 5, 6) else EXIT["failed"]
            raise Stop(code, f"{name} exited {result.returncode}: {tail}\nfull log: {log}", "step:" + name if code == 3 else None)
        return result

    def devsearch(self, name: str, command: str, *args, **kwargs):
        return self.run(name, [self.env_python(), "-m", f"devsearch.{command}", *args], **kwargs)

    def script(self, name: str, script: str, *args, **kwargs):
        return self.run(name, [self.env_python(), REPO / "scripts" / script, *args], **kwargs)


def code_key() -> str:
    """Identity of the code being installed: the package sources and build metadata (not the checkout's other files)."""
    digest = hashlib.sha256()
    files = [REPO / "pyproject.toml", REPO / "README.md"] + sorted(p for p in (REPO / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts
                                                                    and not p.name.endswith((".pyc", ".egg-info")) and ".egg-info" not in str(p))
    for path in files:
        digest.update(str(path.relative_to(REPO)).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- frozen identity

def frozen_values(rt: Runtime, model_commit: str | None) -> dict:
    c = rt.config
    return {"profile": c.profile, "model_repo": c.model_repo, "model_commit": model_commit, "model_kind": c.model_kind,
            "dataset_sha256": c.package_sha256, "lock_sha256": sha256_file(c.lock) if c.lock else None, "lock": c.lock.name if c.lock else None,
            "protocol_sha256": sha256_file(REPO / "configs" / "protocol.toml"), "environment_mode": c.env_mode}


def check_frozen(rt: Runtime) -> dict | None:
    """Refuse any change of model, dataset, backend, lock or protocol after they were frozen in this runtime root."""
    path = rt.state / "frozen.json"
    if not path.is_file():
        return None
    frozen = read_json(path)
    configured = rt.config.model_revision if COMMIT.match(rt.config.model_revision) else frozen["model_commit"]
    current = frozen_values(rt, configured)
    changed = {k: (frozen.get(k), v) for k, v in current.items() if frozen.get(k) != v}
    if changed:
        details = "; ".join(f"{k}: frozen {a!r}, configured {b!r}" for k, (a, b) in changed.items())
        raise Stop(EXIT["invalid"], f"this runtime root is frozen to another configuration ({details}). Nothing is changed silently: "
                                    "restore the frozen values, or use a new runtime root for a different model, dataset, backend, lock or protocol")
    return frozen


def freeze(rt: Runtime, model_commit: str) -> dict:
    existing = check_frozen(rt)
    if existing is not None:
        if existing["model_commit"] != model_commit:
            raise Stop(EXIT["invalid"], f"model commit {model_commit} differs from the frozen {existing['model_commit']}")
        return existing
    values = {**frozen_values(rt, model_commit), "frozen_at": utc()}
    write_json(rt.state / "frozen.json", values)
    rt.log(f"frozen: profile {values['profile']}, model {values['model_repo']}@{model_commit}, dataset {values['dataset_sha256'][:12]}, "
           f"lock {values['lock']}, protocol {values['protocol_sha256'][:12]}")
    return values


# ---------------------------------------------------------------------------------------------------------------- check

def _version_tuple(text: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3])


def gpu_facts() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"error": "nvidia-smi not found: no NVIDIA driver is visible"}
    try:
        query = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
        header = subprocess.run([smi], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"nvidia-smi failed: {error}"}
    if query.returncode != 0:
        return {"error": f"nvidia-smi exited {query.returncode}: {query.stderr.strip()[:300]}"}
    gpus = []
    for line in query.stdout.strip().splitlines():
        name, driver, memory = [part.strip() for part in line.split(",")]
        gpus.append({"name": name, "driver": driver, "memory_bytes": int(float(memory)) * 2**20})
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header.stdout)
    return {"gpus": gpus, "driver_cuda": cuda.group(1) if cuda else None}


def reachable(url: str, timeout: float = 15) -> str | None:
    """None when an HTTPS request reached the host (any HTTP status counts), else the error."""
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout).close()
        return None
    except urllib.error.HTTPError as error:
        error.close()
        return None
    except (urllib.error.URLError, OSError) as error:
        return str(getattr(error, "reason", error))


def filesystem_of(path: Path) -> str | None:
    best, kind = "", None
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount = parts[1]
            inside = str(path) == mount or str(path).startswith(mount.rstrip("/") + "/")
            if inside and len(mount) >= len(best):
                best, kind = mount, parts[2]
    except OSError:
        return None
    return kind


def run_check(rt: Runtime, *, quiet: bool = False) -> list[dict]:
    """Every host check as {name, status: passed|failed|warning|not_applicable, detail, prerequisite?}."""
    c = rt.config
    checks = []

    def add(name, status, detail, prereq=None):
        checks.append({"name": name, "status": status, "detail": detail, **({"prerequisite": prereq} if prereq and status == "failed" else {})})

    add("runner-python", "passed" if sys.version_info >= (3, 11) else "failed", platform.python_version(), "python3.11+")
    system = f"{platform.system()}/{platform.machine()}"
    if c.profile == "nvidia-cuda":
        add("platform", "passed" if system == "Linux/x86_64" else "failed", f"{system}; the nvidia-cuda lock is for Linux/x86_64", "linux-x86_64-host")
        facts = gpu_facts()
        if "error" in facts:
            add("gpu-driver", "failed", facts["error"], "nvidia-driver")
        else:
            gpu = facts["gpus"][0] if facts["gpus"] else None
            cuda_ok = facts["driver_cuda"] is not None and _version_tuple(facts["driver_cuda"]) >= _version_tuple(c.min_cuda)
            add("gpu-driver", "passed" if gpu and cuda_ok else "failed",
                f"{len(facts['gpus'])} GPU(s); {gpu['name'] if gpu else '-'}; driver {gpu['driver'] if gpu else '-'} supports CUDA {facts['driver_cuda']}; "
                f"the lock's torch needs CUDA >= {c.min_cuda} (NVIDIA driver 580 or newer for CUDA 13)", "nvidia-driver")
            if gpu:
                add("gpu-memory", "passed" if gpu["memory_bytes"] >= c.min_gpu_memory else "failed",
                    f"{gpu['memory_bytes'] / 2**30:.1f} GiB; at least {c.min_gpu_memory / 2**30:.0f} GiB configured", "gpu-memory")
            if len(facts["gpus"]) > 1:
                add("gpu-count", "warning", "several GPUs: the campaign uses device 0 only (CUDA_VISIBLE_DEVICES is not changed)")
    elif c.profile == "apple-mps":
        add("platform", "passed" if system == "Darwin/arm64" else "failed", f"{system}; the apple-mps lock is for Darwin/arm64", "apple-silicon-host")
    else:
        add("gpu-driver", "not_applicable", "cpu-diagnostic profile: no accelerator; every result is diagnostic")
    if c.env_mode == "lock":
        found = shutil.which(c.python) or (c.python if Path(c.python).is_file() else None)
        if not found:
            add("python3.12", "failed", f"{c.python} not found; the lock is for CPython 3.12", "python3.12")
        else:
            probe = subprocess.run([found, "-c", "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True)
            good = probe.returncode == 0 and probe.stdout.strip() == "3.12"
            add("python3.12", "passed" if good else "failed",
                f"{found}: {probe.stdout.strip() or probe.stderr.strip()[-200:]} (needs 3.12 with venv and ensurepip)", "python3.12")
    else:
        add("environment", "warning", "environment.mode 'current': the running interpreter is used (CPU fixture tests only); formal gates cannot pass")
    add("git", "passed" if shutil.which("git") else "failed", shutil.which("git") or "git not found (needed to fetch the pinned sources)", "git")
    root = c.root
    if not root.is_dir():
        add("runtime-root", "failed", f"{root} does not exist; create it on the persistent volume (it is never created implicitly)", "runtime-root")
    else:
        try:
            probe = root / f".write-probe-{os.getpid()}"
            probe.write_bytes(b"x")
            probe.unlink()
            add("runtime-root", "passed", f"{root} is writable")
        except OSError as error:
            add("runtime-root", "failed", f"{root} is not writable: {error}", "runtime-root")
        kind = filesystem_of(root.resolve())
        if kind in ("tmpfs", "ramfs"):
            add("persistent-storage", "failed", f"{root} is on {kind}: it does not survive a restart", "persistent-storage")
        elif kind == "overlay":
            add("persistent-storage", "warning", f"{root} is on the container overlay: confirm with the provider that it persists across restarts")
        else:
            add("persistent-storage", "passed", f"filesystem {kind or 'unknown'}")
        if c.mount is not None:
            inside = str(root.resolve()).startswith(str(c.mount.resolve()) + "/") or root.resolve() == c.mount.resolve()
            mounted = os.path.ismount(c.mount)
            status = "passed" if inside and (mounted or not c.require_mount_point) else "failed"
            add("mount", status, f"{c.mount} {'is' if mounted else 'is not'} a mount point; root inside: {inside}", "persistent-storage")
        free = shutil.disk_usage(root).free
        add("free-space", "passed" if free >= c.min_free_bytes else "failed",
            f"{free / 2**30:.1f} GiB free; {c.min_free_bytes / 2**30:.0f} GiB configured (see the quickstart storage estimate)", "disk-space")
    if c.env_mode == "lock" and c.record_receipts and platform.system() == "Linux":
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, text=True) if shutil.which("unshare") else None
        ok = probe is not None and probe.returncode == 0
        add("network-namespace", "passed" if ok else "failed",
            "unprivileged `unshare -rn` works (the portable receipt's offline install)" if ok else
            f"`unshare -rn true` failed ({probe.stderr.strip()[:200] if probe else 'unshare missing'}); the portable-bundle gate cannot pass on this host",
            "user-namespaces")
    needs = []
    if c.env_mode == "lock" and not rt.done("environment"):
        needs += [("pypi", "https://pypi.org/simple/"), ("pypi-files", "https://files.pythonhosted.org/")]
    if c.model_source == "huggingface" and not rt.done("model"):
        needs.append(("model-host", c.endpoint + "/api/models/" + c.model_repo))
    if not rt.done("data"):
        needs.append(("github", "https://github.com/"))
    if c.package_url and not (c.package.is_file()):
        needs.append(("dataset-url", c.package_url))
    for name, url in needs:
        error = reachable(url)
        add(f"network:{name}", "passed" if error is None else "failed", url if error is None else f"{url}: {error}", f"network-{name}")
    if c.package.is_file():
        actual = sha256_file(c.package)
        add("dataset-package", "passed" if actual == c.package_sha256 else "failed",
            f"{c.package} sha256 {actual[:16]}... (pinned {c.package_sha256[:16]}...)", "dataset-package")
    elif not c.package_url:
        add("dataset-package", "failed", f"{c.package} is missing: copy the dataset package there (see the quickstart) or set dataset.url", "dataset-package")
    token = os.environ.get(c.token_env)
    add("credentials", "passed", f"{c.token_env} {'is set (sent only to ' + c.endpoint + ')' if token else 'is not set; the public model needs none'}")
    if not quiet:
        for check in checks:
            print(f"  [{check['status']:>14}] {check['name']}: {check['detail']}")
    return checks


def stage_check(rt: Runtime) -> int:
    checks = run_check(rt)
    if rt.root.is_dir():
        rt.state.mkdir(exist_ok=True)
        write_json(rt.state / "check-latest.json", {"at": utc(), "checks": checks})
    failed = [c for c in checks if c["status"] == "failed"]
    for check in failed:
        print(f"PREREQUISITE {check.get('prerequisite', check['name'])}: {check['detail']}")
    return EXIT["prerequisite"] if failed else EXIT["ok"]


# ---------------------------------------------------------------------------------------------------------------- prepare steps

def step_volume(rt: Runtime) -> None:
    """The volume identity marker the host binding checks: in the runtime root, and at the configured mount point if there is one."""
    for directory in [rt.root] + ([rt.config.mount] if rt.config.mount else []):
        marker = directory / ".devsearch-volume"
        if marker.is_file():
            if marker.read_text().strip() != rt.config.volume_id:
                raise Stop(EXIT["invalid"], f"{marker} names volume {marker.read_text().strip()!r}, not runtime.volume_id {rt.config.volume_id!r}")
        else:
            marker.write_text(rt.config.volume_id + "\n")


def step_environment(rt: Runtime) -> dict:
    c, key = rt.config, rt.code_key()
    record = rt.done("environment")
    if record and record.get("code_key") == key and Path(record["python"]).exists():
        return record
    if c.env_mode == "current":
        return rt.record("environment", code_key=key, python=sys.executable, wheel=None, wheel_sha256=None, mode="current")
    tools = rt.path("tools")
    if not (tools / "bin" / "python").exists():
        rt.run("tools-venv", [shutil.which(c.python) or c.python, "-m", "venv", tools])
    source = rt.path("build", key)
    if not (source / "pyproject.toml").is_file():
        staging = rt.path("build", key + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(REPO / name, staging / name)
        shutil.copytree(REPO / "src", staging / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
        os.replace(staging, source)
    dist = rt.path("dist", key)
    wheels = sorted(dist.glob("devsearch-*.whl")) if dist.is_dir() else []
    if not wheels:
        dist.mkdir(parents=True, exist_ok=True)
        rt.run("build-wheel", [tools / "bin" / "python", "-m", "pip", "wheel", "--disable-pip-version-check", "--no-deps", "-w", dist, source])
        wheels = sorted(dist.glob("devsearch-*.whl"))
    wheel = wheels[-1]
    wheelhouse = rt.path("wheelhouse", c.lock.stem)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    rt.run("wheelhouse-download", [tools / "bin" / "python", REPO / "scripts" / "bootstrap_env.py", "download", "--lock", c.lock, "--dest", wheelhouse],
           timeout=max(c.step_timeout, 7200))
    venv = rt.path("envs", key)
    if venv.exists():                                   # a partial environment from an interrupted install is kept aside, never reused
        os.replace(venv, rt.path("envs", f"{key}.failed-{utc().replace(':', '')}"))
    rt.run("environment-install", [tools / "bin" / "python", REPO / "scripts" / "bootstrap_env.py", "install", "--lock", c.lock, "--wheelhouse", wheelhouse,
                                   "--wheel", wheel, "--venv", venv])
    return rt.record("environment", code_key=key, python=str(venv / "bin" / "python"), wheel=str(wheel), wheel_sha256=sha256_file(wheel),
                     wheelhouse=str(wheelhouse), lock=c.lock.name, lock_sha256=sha256_file(c.lock), mode="lock")


def step_receipts(rt: Runtime, environment: dict) -> dict | None:
    c, key = rt.config, rt.code_key()
    if not c.record_receipts or c.env_mode != "lock":
        rt.log("receipts: not recorded (receipts.record = false or environment.mode = 'current'); a formal plan will refuse the two gates")
        return None
    record = rt.done("receipts")
    if record and record.get("code_key") == key:
        return record
    directory = rt.path("receipts", key)
    if directory.exists() and any(directory.iterdir()):
        os.replace(directory, rt.path("receipts", f"{key}.incomplete-{utc().replace(':', '')}"))
    directory.mkdir(parents=True)
    plain = {"DEVSEARCH_RUNTIME_ROOT": None, "HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}      # the suite runs as in CI
    rt.script("receipt-implementation", "record_receipt.py", "implementation", "--output", directory, "--python", environment["python"],
              timeout=max(c.step_timeout, 7200), env_extra=plain)
    rt.script("receipt-portable", "record_receipt.py", "portable", "--wheel", environment["wheel"], "--lock", c.lock, "--wheelhouse", environment["wheelhouse"],
              "--output", directory, "--python", shutil.which(c.python) or c.python, timeout=max(c.step_timeout, 7200), env_extra=plain)
    receipts = {p.name: read_json(p) for p in sorted(directory.glob("*.json"))}
    failed = [name for name, r in receipts.items() if not r.get("commands") or any(cmd.get("exit_status") != 0 for cmd in r["commands"])]
    if failed:
        raise Stop(EXIT["failed"], f"receipt(s) {failed} did not pass; see {directory} and the logs. A formal plan would refuse them")
    return rt.record("receipts", code_key=key, directory=str(directory), files=sorted(receipts))


def _http(url: str, token: str | None, *, timeout: int = 120):
    request = urllib.request.Request(url, headers={"User-Agent": "devsearch-gpu/1", **({"Authorization": f"Bearer {token}"} if token else {})})
    return urllib.request.urlopen(request, timeout=timeout)


def _api(rt: Runtime, path: str) -> dict:
    c = rt.config
    token = os.environ.get(c.token_env)
    try:
        with _http(f"{c.endpoint}/api/models/{c.model_repo}/{path}", token) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        error.close()
        if error.code in (401, 403):
            raise prerequisite("model-credentials", f"{c.endpoint} refused {c.model_repo} ({error.code}): set {c.token_env} to a token with read access")
        raise Stop(EXIT["failed"], f"model metadata request failed: HTTP {error.code} for {path}")
    except (urllib.error.URLError, OSError) as error:
        raise prerequisite("network-model-host", f"cannot reach {c.endpoint}: {getattr(error, 'reason', error)}")


def _download(rt: Runtime, url: str, target: Path, expect: dict) -> None:
    """Download into `<target>.part` (resuming a partial file), verify size and hash, then rename. A mismatch deletes the part file."""
    token = os.environ.get(rt.config.token_env)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, 5):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "devsearch-gpu/1", **({"Authorization": f"Bearer {token}"} if token else {}), **({"Range": f"bytes={offset}-"} if offset else {})}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with part.open(mode) as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            break
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 416:
                break
            if error.code in (401, 403):
                raise prerequisite("model-credentials", f"download of {target.name} refused ({error.code}): set {rt.config.token_env}")
            if attempt == 4:
                raise Stop(EXIT["failed"], f"download of {target.name} failed: HTTP {error.code}")
        except (urllib.error.URLError, OSError) as error:
            if attempt == 4:
                raise prerequisite("network-model-host", f"download of {target.name} failed: {getattr(error, 'reason', error)}")
        time.sleep(2 ** attempt)
    size = part.stat().st_size
    actual = sha256_file(part) if "sha256" in expect else git_blob_sha1(part)
    wanted = expect.get("sha256") or expect.get("blob_sha1")
    if size != expect["size"] or actual != wanted:
        part.unlink()
        raise Stop(EXIT["invalid"], f"integrity check failed for {target.name}: {size} bytes, hash {actual} (expected {expect['size']} bytes, {wanted}); "
                                    "the partial file was removed and nothing was registered")
    os.replace(part, target)


CONVERT = """import sys, torch
from safetensors.torch import save_file
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
state = {k: v.detach().contiguous().clone() for k, v in state.items() if hasattr(v, "detach")}
save_file(state, sys.argv[2], metadata={"format": "pt", "converted_from": "pytorch_model.bin"})
print(len(state))
"""


def download_model_files(rt, commit: str, download: Path) -> tuple[dict, str, list, dict]:
    """The model files at `commit` into `download`, each verified against the host's LFS SHA-256 or git blob SHA-1.
    `rt` only needs `.config` (endpoint, model_repo, token_env) and `.log`. Returns (files, weights name, wanted names, host info)."""
    c = rt.config
    info = _api(rt, f"revision/{commit}?blobs=true")
    if info.get("sha") != commit:
        raise Stop(EXIT["invalid"], f"the model host answered for commit {info.get('sha')!r}, not {commit}")
    siblings = {s["rfilename"]: s for s in info.get("siblings", [])}
    weights = "model.safetensors" if "model.safetensors" in siblings else "pytorch_model.bin" if "pytorch_model.bin" in siblings else None
    if weights is None or "config.json" not in siblings or not ({"vocab.txt", "tokenizer.json"} & set(siblings)):
        raise Stop(EXIT["invalid"], f"{c.model_repo}@{commit} lacks config.json, a tokenizer or PyTorch weights: {sorted(siblings)}")
    wanted = ["config.json", weights] + [name for name in TOKENIZER_FILES if name in siblings]
    download.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in wanted:
        sibling = siblings[name]
        lfs = sibling.get("lfs") or {}
        expect = {"size": lfs.get("size", sibling.get("size")), **({"sha256": lfs["sha256"]} if lfs.get("sha256") else {"blob_sha1": sibling.get("blobId")})}
        if expect["size"] is None or not (expect.get("sha256") or expect.get("blob_sha1")):
            raise Stop(EXIT["invalid"], f"the model host gave no size or hash for {name}; it cannot be verified")
        path = download / name
        if path.is_file() and path.stat().st_size == expect["size"] and (sha256_file(path) if "sha256" in expect else git_blob_sha1(path)) == (expect.get("sha256") or expect.get("blob_sha1")):
            rt.log(f"model: {name} already downloaded and verified")
        else:
            _download(rt, f"{c.endpoint}/{c.model_repo}/resolve/{commit}/{name}", path, expect)
            rt.log(f"model: {name} downloaded and verified ({expect['size']} bytes)")
        files[name] = {**expect, "sha256_actual": sha256_file(path)}
    return files, weights, wanted, info


def step_model(rt: Runtime) -> dict:
    """Download the pinned model files at an immutable commit, verify every byte, record provenance and register the asset set."""
    c = rt.config
    record = rt.done("model")
    if record:
        return record                                   # the registered files are re-verified by `bundle assemble` and by the doctor
    frozen = check_frozen(rt)
    target = rt.path("assets", "model")
    if c.model_source == "local":
        commit = c.model_revision
        fetched = c.local_dir / FETCH_MANIFEST
        declared = read_json(fetched) if fetched.is_file() else None
        if declared is not None:
            if declared.get("commit") != commit or declared.get("repo") != c.model_repo:
                raise Stop(EXIT["invalid"], f"{fetched} is for {declared.get('repo')}@{declared.get('commit')}, not {c.model_repo}@{commit}")
            for name, entry in declared["files"].items():
                if not (c.local_dir / name).is_file() or sha256_file(c.local_dir / name) != entry["sha256_actual"]:
                    raise Stop(EXIT["invalid"], f"{c.local_dir / name} does not match the hash recorded by `gpu.py fetch-model`; it is not used")
        skip = {"assets-manifest.json", FETCH_MANIFEST, "pytorch_model.bin"}
        names = sorted(p.name for p in c.local_dir.iterdir() if p.is_file() and p.name not in skip and not p.name.startswith("."))
        files = {name: {"sha256": sha256_file(c.local_dir / name), "size": (c.local_dir / name).stat().st_size, "origin": str(c.local_dir / name)} for name in names}
        staging = rt.path("assets", "model.partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for name in names:
            shutil.copy2(c.local_dir / name, staging / name)
        conversion = None
        if "model.safetensors" not in names and (c.local_dir / "pytorch_model.bin").is_file():
            rt.run("convert-weights", [rt.env_python(), "-I", "-c", CONVERT, c.local_dir / "pytorch_model.bin", staging / "model.safetensors"])
            conversion = {"from": "pytorch_model.bin", "from_sha256": sha256_file(c.local_dir / "pytorch_model.bin"),
                          "method": "torch.load(weights_only=True) -> safetensors.save_file", "to_sha256": sha256_file(staging / "model.safetensors")}
        provenance = {"source": "local", "directory": str(c.local_dir), "declared_repo": c.model_repo, "declared_revision": commit, "files": files,
                      "conversion": conversion, "hub_hashes_verified_by_fetch_model": declared is not None,
                      "fetched": None if declared is None else {k: declared[k] for k in ("endpoint", "retrieved_at", "last_modified") if k in declared}}
    else:
        if c.model_revision == RESOLVE_ONCE:
            commit = frozen["model_commit"] if frozen and frozen.get("model_commit") else _api(rt, "revision/main")["sha"]
            rt.log(f"model: '{RESOLVE_ONCE}' resolved {c.model_repo}@main to {commit}; it is frozen from now on (set it in model.revision to make it explicit)")
        else:
            commit = c.model_revision
        if not COMMIT.match(commit):
            raise Stop(EXIT["invalid"], f"the model host returned {commit!r}, not a 40-hex commit")
        files, weights, wanted, info = download_model_files(rt, commit, rt.path("assets", "download", commit))
        download = rt.path("assets", "download", commit)
        staging = rt.path("assets", "model.partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for name in wanted:
            if name != "pytorch_model.bin":
                shutil.copy2(download / name, staging / name)
        conversion = None
        if weights == "pytorch_model.bin":
            rt.run("convert-weights", [rt.env_python(), "-I", "-c", CONVERT, download / "pytorch_model.bin", staging / "model.safetensors"])
            conversion = {"from": "pytorch_model.bin", "from_sha256": files["pytorch_model.bin"]["sha256_actual"], "method": "torch.load(weights_only=True) -> safetensors.save_file",
                          "to_sha256": sha256_file(staging / "model.safetensors")}
        provenance = {"source": "huggingface", "endpoint": c.endpoint, "repo": c.model_repo, "commit": commit, "retrieved_at": utc(),
                      "last_modified": info.get("lastModified"), "files": files, "conversion": conversion, "license_declared": c.model_license}
    freeze(rt, commit)
    if target.exists():
        os.replace(target, rt.path("assets", f"model.replaced-{utc().replace(':', '')}"))
    os.replace(staging, target)
    write_json(rt.path("assets", "model-provenance.json"), provenance)
    result = rt.devsearch("register-assets", "bundle", "register-assets", "--directory", target, "--repo", c.model_repo, "--revision", commit,
                          "--license", c.model_license, "--kind", c.model_kind)
    manifest = json.loads(result.stdout)
    return rt.record("model", commit=commit, directory=str(target), assets_manifest_sha256=manifest["manifest_sha256"], pinned=manifest["pinned"],
                     asset_kind=manifest["asset_kind"], provenance=str(rt.path("assets", "model-provenance.json")))


def step_dataset(rt: Runtime) -> dict:
    """Fetch (optional) and verify the dataset package, then unpack it with path filtering and check its inner file list."""
    c = rt.config
    record = rt.done("dataset")
    if record and record.get("sha256") == c.package_sha256:
        return record
    if not c.package.is_file():
        if not c.package_url:
            raise prerequisite("dataset-package", f"{c.package} is missing: copy the dataset package (sha256 {c.package_sha256}) there, "
                                                  "or set dataset.url. See docs/GPU_QUICKSTART.md")
        c.package.parent.mkdir(parents=True, exist_ok=True)
        part = c.package.with_name(c.package.name + ".part")
        try:
            with urllib.request.urlopen(c.package_url, timeout=120) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except (urllib.error.URLError, OSError) as error:
            raise prerequisite("dataset-package", f"cannot download {c.package_url}: {getattr(error, 'reason', error)}")
        os.replace(part, c.package)
    actual = sha256_file(c.package)
    if actual != c.package_sha256:
        raise Stop(EXIT["invalid"], f"{c.package} has sha256 {actual}, not the pinned {c.package_sha256}; it is not used")
    target = rt.path("dataset")
    if target.exists():
        os.replace(target, rt.path(f"dataset.replaced-{utc().replace(':', '')}"))
    staging = rt.path("dataset.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    with tarfile.open(c.package, "r:gz") as archive:
        archive.extractall(staging, filter="data")
    inner = staging / PACKAGE_DIR
    meta = read_json(inner / "package.json")
    for name, digest in meta["files"].items():
        if sha256_file(inner / name) != digest:
            raise Stop(EXIT["invalid"], f"dataset package file {name} does not match its recorded hash")
    present = sorted(str(p.relative_to(inner)) for p in inner.rglob("*") if p.is_file() and p.name != "package.json")
    if present != sorted(meta["files"]):
        raise Stop(EXIT["invalid"], "the dataset package holds files that its package.json does not list")
    os.replace(inner, target)
    shutil.rmtree(staging)
    return rt.record("dataset", sha256=actual, directory=str(target), frozen=meta["frozen"], labels=meta.get("labels", {}))


def materialize_configs(dataset: Path, tokenizer: Path, snapshots: Path, out: Path) -> tuple[Path, Path]:
    """The package's sources and data configs with this host's paths filled in (only the two placeholders change)."""
    out.mkdir(parents=True, exist_ok=True)
    sources = out / "sources.toml"
    sources.write_text((dataset / "sources.toml").read_text(encoding="utf-8").replace("@PINNED_TOKENIZER@", str(tokenizer)).replace("@SNAPSHOTS@", str(snapshots)),
                       encoding="utf-8")
    data = dataset / "data.toml"
    text = data.read_text(encoding="utf-8")
    if "@PINNED_TOKENIZER@" not in text:
        raise Stop(EXIT["invalid"], "the package's data.toml has no @PINNED_TOKENIZER@ placeholder for its [lengths] check")
    host_data = dataset / "data.host.toml"
    host_data.write_text(text.replace("@PINNED_TOKENIZER@", str(tokenizer)), encoding="utf-8")
    return sources, host_data


def membership(prepared: Path, evaluation: Path, corpus_manifest: Path) -> dict:
    """The values that define the frozen data: corpus, index, every split's questions and the test membership."""
    ev = read_json(evaluation / "evaluation-manifest.json")
    pm = read_json(prepared / "prepared-data-manifest.json")
    request = evaluation / "review" / "review-request.jsonl"
    values = {
        "corpus_id": read_json(corpus_manifest)["corpus_id"], "prepared_corpus_id": ev["corpus_id"], "index_id": ev["index_id"],
        "test_queries_sha256": ev["queries"]["sha256"], "test_queries": ev["queries"]["count"], "test_qrels_sha256": ev["qrels_v1"]["sha256"],
        "test_groups_sha256": ev["test_groups_sha256"], "protocol": ev["protocol"],
        "review_request_sha256": sha256_file(request) if request.is_file() else None,
        "review_items": sum(1 for line in request.read_text(encoding="utf-8").splitlines() if line.strip()) if request.is_file() else 0,
        "training_files": {name: entry["sha256"] for name, entry in sorted(pm["data"].items())},
        "pairs_sampling_id": pm["pairs"]["sampling_manifest_id"], "pairs_manifest_sha256": pm["pairs"]["manifest_sha256"],
        "validation_candidates_sha256": pm["validation_candidates"]["manifest_sha256"],
    }
    return values


def build_data(rt_or_runner, *, python: Path, root: Path, dataset: Path, tokenizer: Path, snapshots: Path, offline: bool = False) -> dict:
    """corpus (prepare) -> groups -> materialize, all under `root`. Shared by `prepare` and `package` so both build identically."""
    run = rt_or_runner
    sources, data = materialize_configs(dataset, tokenizer, snapshots, root / "sources")
    corpus = root / "data" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    prepared, evaluation = root / "bundle" / "prepared", root / "evaluation"
    prepared.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    run("prepare-corpus", [python, "-m", "devsearch.prepare", "--config", sources, "--output", corpus, "--runtime-root", root, *(["--offline"] if offline else [])])
    run("prepare-groups", [python, "-m", "devsearch.prepare_data", "--config", data, "--stage", "groups", "--corpus", corpus / "corpus-manifest.json",
                           "--output", prepared / "split-manifest.json", "--runtime-root", root])
    run("prepare-materialize", [python, "-m", "devsearch.prepare_data", "--config", data, "--stage", "materialize", "--split-manifest", prepared / "split-manifest.json",
                                "--evaluation-dir", evaluation, "--output", prepared / "prepared-data-manifest.json", "--runtime-root", root])
    return membership(prepared, evaluation, corpus / "corpus-manifest.json")


def step_tokenizer(rt: Runtime, dataset: Path) -> dict:
    record = rt.done("tokenizer")
    model = rt.done("model")
    if record and record.get("assets_manifest_sha256") == model["assets_manifest_sha256"]:
        return record
    snapshots = rt.path("sources", "snapshots")
    snapshots.mkdir(parents=True, exist_ok=True)
    texts = [dataset / "reference-texts.jsonl"] if (dataset / "reference-texts.jsonl").is_file() else []
    texts += sorted((dataset / "drafts").glob("queries-*.jsonl"))
    output = rt.path("data", f"tokenizer-comparison-{model['assets_manifest_sha256'][:12]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    args = ["--candidate", rt.path("assets", "model"), "--reference", dataset / "reference-tokenizer", "--output", output]
    for path in texts:
        args += ["--texts", path]
    result = rt.script("compare-tokenizers", "compare_tokenizers.py", *args, ok=(0, 1))
    if result.returncode != 0:
        raise prerequisite("protocol-revision", f"the pinned tokenizer does not encode the frozen texts like the reference the questions were built with "
                                                f"(report {output}). The data must be rebuilt under a recorded protocol revision before any sealing; "
                                                "this script stops instead of relabelling or replacing questions")
    return rt.record("tokenizer", identical=True, report=str(output), assets_manifest_sha256=model["assets_manifest_sha256"])


def step_data(rt: Runtime, dataset_record: dict) -> dict:
    record = rt.done("data")
    if record:
        return record
    dataset = Path(dataset_record["directory"])
    built = build_data(lambda name, argv: rt.run(name, argv), python=rt.env_python(), root=rt.root, dataset=dataset,
                       tokenizer=rt.path("assets", "model"), snapshots=rt.path("sources", "snapshots"))
    frozen = dataset_record["frozen"]
    differing = {k: {"frozen": frozen.get(k), "rebuilt": v} for k, v in built.items() if frozen.get(k) != v}
    audit = {"at": utc(), "identical": not differing, "differences": differing, "rebuilt": built, "frozen": frozen}
    write_json(rt.path("data", "membership-audit.json"), audit)
    if differing:
        raise prerequisite("protocol-revision", f"the rebuilt data differ from the frozen dataset in {sorted(differing)} (see {rt.path('data', 'membership-audit.json')}). "
                                                "Test membership is frozen: nothing is sealed, and a rebuild needs a recorded protocol revision")
    rt.log(f"data: rebuilt with the pinned tokenizer; corpus {built['corpus_id']}, {built['test_queries']} test questions, "
           f"{built['review_items']} review items; identical to the frozen dataset")
    return rt.record("data", membership=built, audit=str(rt.path("data", "membership-audit.json")))


def step_review(rt: Runtime) -> dict:
    """Judge batches for two passes; then the coverage audit; only a complete audit reaches the bank and the initial seal."""
    record = rt.done("review")
    if record:
        return record
    c = rt.config
    request = rt.path("evaluation", "review", "review-request.jsonl")
    batches = rt.path("review", "batches")
    for name in ("A", "B"):
        directory = batches / name
        if not directory.is_dir() or not any(directory.iterdir()):
            directory.mkdir(parents=True, exist_ok=True)
            rt.devsearch(f"review-batches-{name}", "review", "batches", "--request", request, "--output-dir", directory, "--pass", name,
                         "--batch-size", str(c.review_batch_size))
    responses = rt.path("review", "responses")
    responses.mkdir(parents=True, exist_ok=True)
    stamp = utc().replace(":", "")
    merged = rt.path("review", f"all-responses-{stamp}.jsonl")
    audit = rt.path("review", f"audit-{stamp}.json")
    gpu = f"python3 {REPO / 'scripts' / 'gpu.py'} cmd --config {rt.config.path} --"
    instructions = (f"judge every batch in {batches}/A and {batches}/B (two independent primary passes; the judgment format is in "
                    f"scripts/realdata/prompts/judge-candidate-v1.md), import each pass with\n"
                    f"  {gpu} review import --request {request} --judgments <files> --reviewer-id <id> --reviewer-type agent|human "
                    f"--role primary [--revision <prompt revision>] --output {responses}/<pass>.jsonl\n"
                    f"track progress and size sessions with `{gpu} review progress ... --budget-tokens N` (an estimate for choosing batches, not a spending cap; docs/15-review-workflow.md), "
                    f"then adjudicate what the audit lists ({rt.path('review')}/audit-*.json, `adjudication_queue`; batches via `devsearch.review batches --items`) "
                    "and rerun `gpu.py prepare`")
    if not any(responses.glob("*.jsonl")):
        raise Stop(EXIT["prerequisite"], f"no review judgments yet: {instructions}", "review-bank")
    result = rt.devsearch("review-audit", "review", "audit", "--request", request, "--responses-dir", responses, "--output", audit,
                          "--merged", merged, "--min-primary-reviews", str(c.min_primary_reviews), ok=(0, 3))
    if result.returncode == 3:
        summary = read_json(audit)
        raise Stop(EXIT["prerequisite"], f"review coverage incomplete: {summary['final']} of {summary['items']} items final, states {summary['states']}. "
                                         f"Sealing and formal evaluation stay blocked. Next: {instructions}", "review-bank")
    prepared = rt.path("bundle", "prepared")
    rt.devsearch("prepare-bank", "prepare_data", "--config", rt.path("dataset", "data.host.toml"), "--stage", "bank", "--prepared-manifest",
                 prepared / "prepared-data-manifest.json", "--evaluation-manifest", rt.path("evaluation", "evaluation-manifest.json"), "--reviews", merged,
                 "--output", prepared / "initial-seal.json", "--runtime-root", rt.root)
    return rt.record("review", audit=str(audit), merged=str(merged), seal=str(prepared / "initial-seal.json"))


def step_bundle(rt: Runtime) -> dict:
    record = rt.done("bundle")
    if record:
        return record
    bundle = rt.path("bundle")
    protocol = bundle / "protocol.toml"
    source = REPO / "configs" / "protocol.toml"
    if protocol.exists() and protocol.read_bytes() != source.read_bytes():
        raise Stop(EXIT["invalid"], f"{protocol} differs from configs/protocol.toml; the protocol is frozen")
    shutil.copyfile(source, protocol)
    assets = bundle / "model-assets"
    if not assets.exists():
        shutil.copytree(rt.path("assets", "model"), assets)
    args = ["assemble", "--bundle", bundle] + (["--lock", rt.config.lock] if rt.config.env_mode == "lock" else [])
    manifest = json.loads(rt.devsearch("bundle-assemble", "bundle", *args).stdout)
    return rt.record("bundle", manifest_sha256=manifest["manifest_sha256"], diagnostic=manifest["diagnostic"], corpus_id=manifest["corpus_id"])


def stage_prepare(rt: Runtime) -> int:
    check_frozen(rt)
    failed = [c for c in run_check(rt, quiet=True) if c["status"] == "failed"]
    if failed:
        for check in failed:
            print(f"PREREQUISITE {check.get('prerequisite', check['name'])}: {check['detail']}")
        return EXIT["prerequisite"]
    step_volume(rt)
    environment = step_environment(rt)
    step_receipts(rt, environment)
    step_model(rt)
    dataset = step_dataset(rt)
    step_tokenizer(rt, Path(dataset["directory"]))
    step_data(rt, dataset)
    step_review(rt)
    bundle = step_bundle(rt)
    rt.record("prepare", bundle=bundle, code_key=rt.code_key())
    rt.log(f"prepare: complete; bundle {bundle['manifest_sha256'][:12]} (diagnostic: {bundle['diagnostic']}). Next: gpu.py diagnose")
    return EXIT["ok"]


# ---------------------------------------------------------------------------------------------------------------- host file

def gpu_memory_cap(rt: Runtime) -> int:
    if rt.config.profile != "nvidia-cuda":
        return 0
    facts = gpu_facts()
    if "error" in facts or not facts["gpus"]:
        raise prerequisite("nvidia-driver", facts.get("error", "no GPU visible"))
    return int(facts["gpus"][0]["memory_bytes"] * rt.config.device_memory_fraction)


def host_memory_cap() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(int(line.split()[1]) * 1024 * 0.8)
    except OSError:
        pass
    return 8 * 2**30


def host_toml(*, profile: str, runtime_root: Path, bundle: Path, run_root: Path, evaluation_root: Path, backup_dir: Path, mount: Path, volume_id: str,
              deadline: str, min_free_bytes: int, device_cap: int, independent_backup: bool, require_mount_point: bool = False) -> str:
    return f"""# Written by scripts/gpu.py; edit only through gpu.py's configuration (an edit invalidates the doctor report).
schema_version = 1
[device]
profile = "{profile}"
[paths]
runtime_root = "{runtime_root}"
bundle = "{bundle}"
run_root = "{run_root}"
evaluation_root = "{evaluation_root}"
backup_dir = "{backup_dir}"
[mount]
path = "{mount}"
marker = ".devsearch-volume"
expected_id = "{volume_id}"
require_mount_point = {str(require_mount_point).lower()}
[deadline]
at = "{deadline}"
reserve_seconds = 1800
[resources]
min_free_bytes = {min_free_bytes}
host_memory_cap_bytes = {host_memory_cap()}
device_memory_cap_bytes = {device_cap}
[limits]
process_timeout_seconds = 900
doctor_budget_seconds = 3600
cleanup_grace_seconds = 30
io_retries = 2
[backup]
require_independent_device = {str(independent_backup).lower()}
"""


def _status(rt: Runtime, plan: Path) -> tuple[int, dict | None]:
    result = rt.devsearch("status", "status", "--plan", plan, ok=(0, 1, 4, 5, 6))
    try:
        return result.returncode, json.loads(result.stdout)
    except ValueError:
        return result.returncode, None


# ---------------------------------------------------------------------------------------------------------------- diagnose

def identity(rt: Runtime) -> dict:
    model = rt.done("model") or {}
    return {"code_key": rt.code_key(), "profile": rt.config.profile, "assets_manifest_sha256": model.get("assets_manifest_sha256"),
            "lock_sha256": sha256_file(rt.config.lock) if rt.config.lock else None}


def stage_diagnose(rt: Runtime) -> int:
    check_frozen(rt)
    if not rt.done("model") or not rt.done("environment") or rt.done("environment").get("code_key") != rt.code_key():
        raise prerequisite("prepare", "the environment and the pinned model come first: run `gpu.py prepare` (it may stop later at the review bank; that is fine)")
    record = rt.done("diagnose")
    if record and record.get("identity") == identity(rt) and record.get("passed"):
        rt.log(f"diagnose: already passed for this code, model and profile ({record['directory']})")
        return EXIT["ok"]
    directory = rt.path("diagnose", utc().replace(":", ""))
    directory.mkdir(parents=True)
    rt.devsearch("diagnose-rehearsal", "rehearsal", "--output", directory, "--topics", "24", "--assets", rt.path("assets", "model"))
    for name in ("plan",):
        (directory / name).mkdir()
    deadline = utc(datetime.now(timezone.utc) + timedelta(hours=min(rt.config.time_limit_hours, 6)))
    host = directory / "host.toml"
    host.write_text(host_toml(profile=rt.config.profile, runtime_root=directory / "runtime", bundle=directory / "bundle", run_root=directory / "runtime" / "run",
                              evaluation_root=directory / "evaluation", backup_dir=directory / "backup", mount=directory, volume_id="rehearsal-volume",
                              deadline=deadline, min_free_bytes=10**9, device_cap=gpu_memory_cap(rt), independent_backup=False), encoding="utf-8")
    plan = directory / "plan" / "execution-plan.json"
    rt.devsearch("diagnose-doctor", "doctor", "--profile", rt.config.profile, "--host", host, "--output", directory / "doctor-report.json")
    rt.devsearch("diagnose-plan", "plan", "--host", host, "--doctor-report", directory / "doctor-report.json", "--output", plan)
    crashed = rt.devsearch("diagnose-run-crash", "run", "--plan", plan, ok=(137,), env_extra={"DEVSEARCH_FAULT": DIAGNOSE_FAULT, "DEVSEARCH_FAULT_AFTER": "1"})
    rt.devsearch("diagnose-run-resume", "run", "--plan", plan, "--resume")
    code, status = _status(rt, plan)
    stages = (status or {}).get("stages", {})
    train = stages.get("train:full:42", {}) if isinstance(stages, dict) else next((s for s in stages if s.get("stage_id") == "train:full:42"), {})
    recovered = int(train.get("attempt", 0) or 0) >= 2
    passed = code == 0 and recovered and crashed.returncode == 137
    summary = {"passed": passed, "directory": str(directory), "identity": identity(rt), "status_exit": code, "crash_exit": crashed.returncode,
               "train_full_attempts": train.get("attempt"), "fault": DIAGNOSE_FAULT,
               "note": "diagnostic rehearsal with the pinned model: synthetic data, plumbing evidence only (training, inference, crash recovery, export, "
                       "reload, backup on this device); never a ranking result"}
    rt.record("diagnose", **summary)
    if not passed:
        raise Stop(EXIT["failed"], f"diagnose did not pass (status exit {code}, train:full:42 attempts {train.get('attempt')}); see {directory} and the logs")
    rt.log(f"diagnose: passed on {rt.config.profile} with the pinned model (crash at {DIAGNOSE_FAULT}, resumed, status complete). Next: gpu.py run")
    return EXIT["ok"]


# ---------------------------------------------------------------------------------------------------------------- run

def explain_lock_stop(stop: Stop) -> Stop:
    """A campaign process that outlived `gpu.py` (for example after an SSH drop) still holds the campaign lock: say so by name."""
    if stop.code == EXIT["prerequisite"] and "campaign.lock is held" in str(stop):
        held = re.search(r"campaign\.lock is held by [^;]*", str(stop))
        return prerequisite("campaign-lock", f"{held.group(0) if held else 'the campaign lock is held'}. The campaign keeps running when the session that started it ends, "
                                              "and `gpu.py run` never starts a second one. Watch it with `gpu.py status`; when it has finished, `gpu.py run` reports "
                                              "the result. If that process is hung, stop it; the lock of a dead owner is taken over automatically on the next run")
    return stop


def stage_run(rt: Runtime, extend_hours: float | None = None) -> int:
    check_frozen(rt)
    c = rt.config
    prepared = rt.done("prepare")
    if not prepared:
        raise prerequisite("prepare", "run `gpu.py prepare` until it completes (environment, receipts, model, data, review bank, bundle)")
    bundle = rt.done("bundle")
    formal = c.profile != "cpu-diagnostic"
    if formal and bundle["diagnostic"]:
        raise prerequisite("pretrained-assets", "the bundle is diagnostic (diagnostic model assets or rehearsal data); a formal run needs the pinned pretrained model")
    if formal and not (rt.done("receipts") or {}).get("code_key") == rt.code_key():
        raise prerequisite("receipts", "no implementation and portable receipts for this code: set receipts.record = true and rerun `gpu.py prepare`")
    diagnose = rt.done("diagnose")
    if not diagnose or not diagnose.get("passed") or diagnose.get("identity") != identity(rt):
        raise prerequisite("diagnose", "run `gpu.py diagnose` first: the small real-model training, inference and recovery check must pass for this code, model and profile")
    campaign = rt.path("campaign")
    plan_dir, run_root = campaign / "plan", campaign / "run"
    plan = plan_dir / "execution-plan.json"
    if not plan.is_file():
        for directory in (plan_dir, run_root, c.backup_dir):
            directory.mkdir(parents=True, exist_ok=True)
        deadline = utc(datetime.now(timezone.utc) + timedelta(hours=c.time_limit_hours))
        host = campaign / "host.toml"
        host.write_text(host_toml(profile=c.profile, runtime_root=rt.root, bundle=rt.path("bundle"), run_root=run_root, evaluation_root=rt.path("evaluation"),
                                  backup_dir=c.backup_dir, mount=c.mount or rt.root, volume_id=c.volume_id,
                                  deadline=deadline, min_free_bytes=c.min_free_bytes, device_cap=gpu_memory_cap(rt),
                                  independent_backup=c.require_independent_backup, require_mount_point=c.require_mount_point), encoding="utf-8")
        report = campaign / f"doctor-report-{utc().replace(':', '')}.json"
        rt.devsearch("doctor", "doctor", "--profile", c.profile, "--host", host, "--output", report, timeout=max(c.step_timeout, 7200))
        args = ["--host", host, "--doctor-report", report, "--output", plan]
        receipts = rt.done("receipts")
        if receipts:
            args += ["--receipts", receipts["directory"]]
        environment = rt.done("environment")
        if environment.get("wheel_sha256"):
            args += ["--wheel-sha256", environment["wheel_sha256"]]
        rt.devsearch("plan", "plan", *args)
        rt.log(f"run: plan written ({plan}); deadline {deadline}")
    elif extend_hours:
        host = campaign / "host.toml"
        deadline = utc(datetime.now(timezone.utc) + timedelta(hours=extend_hours))
        host.write_text(re.sub(r'(?m)^at = ".*"$', f'at = "{deadline}"', host.read_text(encoding="utf-8")), encoding="utf-8")
        rt.devsearch("plan-revise-binding", "plan", "--revise-binding", "--plan", plan, "--host", host,
                     "--authorization", f"operator extended the deadline to {deadline} with `gpu.py run --extend-hours {extend_hours}` at {utc()}")
        rt.log(f"run: deadline extended to {deadline} (a new host binding; the plan itself is unchanged)")
    resume = (run_root / "run-state.json").is_file()
    try:
        result = rt.devsearch("campaign-run" + ("-resume" if resume else ""), "run", "--plan", plan, *(["--resume"] if resume else []),
                              ok=(0, 4, 5, 6), timeout=int((extend_hours or c.time_limit_hours) * 3600) + 3600)
    except Stop as stop:
        raise explain_lock_stop(stop)
    code, status = _status(rt, plan)
    rt.record("run-latest", run_exit=result.returncode, status_exit=code, next_action=(status or {}).get("next_action"), plan=str(plan),
              diagnostic_profile=not formal)
    if code == 0:
        rt.log(f"run: complete and every artifact re-verified. Outputs under {run_root} (final report, release, backup receipt); backup in {c.backup_dir}"
               + ("" if formal else ". cpu-diagnostic profile: a diagnostic run, not a formal result"))
        return EXIT["ok"]
    rt.log(f"run: not complete (run exit {result.returncode}, status exit {code}); next: {(status or {}).get('next_action')}. Rerun `gpu.py run` to resume")
    return code if code in (4, 5, 6) else EXIT["failed"]


# ---------------------------------------------------------------------------------------------------------------- status

STEPS = ("environment", "receipts", "model", "dataset", "tokenizer", "data", "review", "bundle", "prepare", "diagnose", "run-latest")


def stage_rehearsal_judge(rt: Runtime) -> int:
    """CPU DIAGNOSTIC REHEARSAL ONLY: write and import two scripted primary passes from the dataset's known positives so the rest of the
    procedure can be exercised. The records say `scripted-oracle` and the notes say it is not a review. Refused on any accelerator profile."""
    if rt.config.profile != "cpu-diagnostic":
        raise Stop(EXIT["invalid"], f"rehearsal-judge exists for the cpu-diagnostic rehearsal only; profile {rt.config.profile!r} needs real judgments (docs/15-review-workflow.md)")
    request = rt.path("evaluation", "review", "review-request.jsonl")
    chunks_file = rt.path("data", "corpus", "chunks.jsonl")
    if not request.is_file() or not chunks_file.is_file():
        raise prerequisite("prepare", "run `gpu.py prepare` first: it writes the review request and the corpus this reads")
    drafts = rt.path("dataset", "drafts")
    question = {}
    for path in sorted(drafts.glob("queries-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            question[record["query_id"]] = record["text"]
    text_of = {}
    for line in chunks_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        text_of[record["chunk_id"]] = record["text"]
    positive = set()
    for path in sorted(drafts.glob("qrels-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            positive.add((question[record["query_id"]], text_of[record["chunk_id"]]))
    for name in ("A", "B"):
        responses = rt.path("review", "responses", f"{name}.jsonl")
        if responses.exists():
            rt.log(f"rehearsal-judge: pass {name} is already imported")
            continue
        lines = []
        for batch in sorted(rt.path("review", "batches", name).glob("*.jsonl")):
            for line in batch.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                relevant = (item["question"], item["text"]) in positive
                lines.append({"item_id": item["item_id"], "relevance": 1 if relevant else 0, "evidence_quote": " ".join(item["text"].split()[:6]) if relevant else "",
                              "note": "SCRIPTED REHEARSAL ORACLE, not a review"})
        judged = rt.path("review", f"rehearsal-judgments-{name}.jsonl")
        judged.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        responses.parent.mkdir(parents=True, exist_ok=True)
        rt.devsearch(f"rehearsal-import-{name}", "review", "import", "--request", request, "--judgments", judged, "--reviewer-id", f"rehearsal-scripted-{name}",
                     "--reviewer-type", "agent", "--revision", "scripted-oracle", "--role", "primary", "--output", responses)
    rt.log("rehearsal-judge: two scripted passes imported (diagnostic; not a review). Next: gpu.py prepare")
    return EXIT["ok"]


def stage_status(rt: Runtime) -> int:
    report = {"root": str(rt.root), "frozen": read_json(rt.state / "frozen.json") if (rt.state / "frozen.json").is_file() else None, "steps": {}}
    for step in STEPS:
        record = rt.done(step)
        report["steps"][step] = None if record is None else {k: v for k, v in record.items() if k in ("finished_at", "passed", "commit", "manifest_sha256",
                                                                                                    "diagnostic", "status_exit", "next_action", "directory", "membership")}
    if not rt.done("prepare"):
        nxt = "gpu.py prepare"
    elif not (rt.done("diagnose") or {}).get("passed"):
        nxt = "gpu.py diagnose"
    elif (rt.done("run-latest") or {}).get("status_exit") != 0:
        nxt = "gpu.py run"
    else:
        nxt = "done: read the final report under <root>/campaign/run and copy the backup off the server"
    report["next"] = nxt
    plan = rt.path("campaign", "plan", "execution-plan.json")
    if plan.is_file() and rt.env_python().exists():
        report["campaign_status_exit"], campaign = _status(rt, plan)
        report["campaign_next_action"] = (campaign or {}).get("next_action")
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT["ok"] if nxt.startswith("done") else EXIT["failed"]


# ---------------------------------------------------------------------------------------------------------------- package (authoring)

def stage_package(args) -> int:
    """Build the dataset package on a connected machine and record the frozen values a server rebuild must reproduce."""
    work = args.work.resolve()
    if not work.is_dir() or any(work.iterdir()):
        raise Stop(EXIT["usage"], f"--work {work} must be an existing, empty directory")
    stage = work / "package" / PACKAGE_DIR
    (stage / "drafts").mkdir(parents=True)
    (stage / "reference-tokenizer").mkdir()
    for name in ("sources.toml", "data.toml", "index.toml"):
        source = getattr(args, name.split(".")[0].replace("-", "_"))
        shutil.copyfile(source, stage / name)
    for path in sorted(args.drafts.glob("*.jsonl")) + sorted(args.drafts.glob("drafts-summary.json")):
        shutil.copyfile(path, stage / "drafts" / path.name)
    for name in TOKENIZER_FILES:
        if (args.reference_tokenizer / name).is_file():
            shutil.copyfile(args.reference_tokenizer / name, stage / "reference-tokenizer" / name)
    snapshots = work / "snapshots"
    if args.snapshots:
        shutil.copytree(args.snapshots, snapshots, symlinks=True)
    else:
        snapshots.mkdir()
    logs = work / "logs"
    logs.mkdir()

    def runner(name, argv):
        argv = [str(a) for a in argv]
        result = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, "DEVSEARCH_RUNTIME_ROOT": str(work), "PYTHONPATH": str(REPO / "src"),
                                                                            "PYTHONHASHSEED": "0"}, check=False)
        (logs / f"{name}.log").write_text(f"$ {' '.join(argv)}\nexit {result.returncode}\n{result.stdout}\n{result.stderr}", encoding="utf-8")
        if result.returncode != 0:
            raise Stop(EXIT["failed"], f"{name} failed: {result.stderr.strip()[-1500:]}")
        return result

    frozen = build_data(runner, python=Path(sys.executable), root=work, dataset=stage, tokenizer=stage / "reference-tokenizer", snapshots=snapshots,
                        offline=bool(args.snapshots))
    os.remove(stage / "data.host.toml")
    chunks = work / "data" / "corpus" / "chunks.jsonl"
    shutil.copyfile(chunks, stage / "reference-texts.jsonl")
    files = {str(p.relative_to(stage)): sha256_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    meta = {"format": 1, "kind": "devsearch-dataset", "frozen": frozen, "files": files,
            "labels": {"questions": "synthetic (agent-generated, agent-reviewed); never human-reviewed", "notice": args.notice or ""}}
    write_json(stage / "package.json", meta)
    output = args.output.resolve()
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
            tarfile.open(fileobj=packed, mode="w", format=tarfile.PAX_FORMAT) as archive:     # byte-reproducible: fixed times and owners
        for path in sorted(stage.rglob("*")):
            info = archive.gettarinfo(str(path), arcname=f"{PACKAGE_DIR}/{path.relative_to(stage)}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    print(json.dumps({"package": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size, "frozen": frozen}, indent=2, sort_keys=True))
    return EXIT["ok"]


# ---------------------------------------------------------------------------------------------------------------- fetch-model / standin-model

def stage_fetch_model(args) -> int:
    """On ANY connected, authorized machine (stdlib only, no runtime root): download the pinned model files at an immutable commit and
    verify every byte against the host's hashes. Copy the directory to the GPU server and use `model.source = "local"` there."""
    import types
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise Stop(EXIT["usage"], f"{out} must not exist or be empty")
    out.mkdir(parents=True, exist_ok=True)
    rt = types.SimpleNamespace(config=types.SimpleNamespace(endpoint=args.endpoint.rstrip("/"), model_repo=args.repo, token_env=args.token_env),
                               log=lambda message: print(f"{utc()} {message}", file=sys.stderr, flush=True))
    commit = args.revision
    if commit == "main":
        commit = _api(rt, "revision/main")["sha"]
        rt.log(f"model: main resolved to {commit}; pin this commit in configs/gpu.toml (model.revision)")
    if not COMMIT.match(commit):
        raise Stop(EXIT["invalid"], "--revision must be a 40-hex commit (or 'main', which is resolved once and printed)")
    files, weights, wanted, info = download_model_files(rt, commit, out)
    write_json(out / FETCH_MANIFEST, {"format": 1, "repo": args.repo, "commit": commit, "endpoint": rt.config.endpoint, "retrieved_at": utc(),
                                      "last_modified": info.get("lastModified"), "weights": weights, "files": files})
    print(json.dumps({"directory": str(out), "repo": args.repo, "commit": commit, "weights": weights, "files": {k: v["sha256_actual"] for k, v in files.items()},
                      "next": f"copy {out} to the server and set [model] source = \"local\", local_dir = <that directory>, revision = \"{commit}\""}, indent=2))
    return EXIT["ok"]


def stage_standin_model(args) -> int:
    """For the CPU diagnostic rehearsal only: a tiny RANDOM encoder with the dataset package's reference vocabulary. Needs torch/transformers."""
    sys.path.insert(0, str(REPO / "src"))
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise Stop(EXIT["usage"], f"{out} must not exist or be empty")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.package, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.startswith(f"{PACKAGE_DIR}/reference-tokenizer/"):
                (out / Path(member.name).name).write_bytes(archive.extractfile(member).read())
    try:
        from safetensors.torch import save_file

        from devsearch.modeling.cross_encoder import build_tiny_encoder
    except ImportError as error:
        raise Stop(EXIT["prerequisite"], f"standin-model needs torch, transformers and safetensors in this interpreter ({error})", "ml-dependencies")
    vocab = len((out / "vocab.txt").read_text(encoding="utf-8").splitlines())
    encoder = build_tiny_encoder(vocab, hidden_size=32, layers=2, seed=3, hidden_dropout=0.1, attention_dropout=0.1)
    save_file({k: v.detach().contiguous().clone() for k, v in encoder.state_dict().items()}, str(out / "model.safetensors"))
    c = encoder.config
    write_json(out / "config.json", {"model_type": "bert", "vocab_size": c.vocab_size, "hidden_size": c.hidden_size, "num_hidden_layers": c.num_hidden_layers,
                                     "num_attention_heads": c.num_attention_heads, "intermediate_size": c.intermediate_size,
                                     "max_position_embeddings": c.max_position_embeddings, "hidden_act": c.hidden_act,
                                     "hidden_dropout_prob": c.hidden_dropout_prob, "attention_probs_dropout_prob": c.attention_probs_dropout_prob,
                                     "type_vocab_size": c.type_vocab_size, "initializer_range": c.initializer_range, "layer_norm_eps": c.layer_norm_eps,
                                     "pad_token_id": c.pad_token_id})
    print(json.dumps({"directory": str(out), "note": "random weights with the reference vocabulary: a CPU diagnostic stand-in, never a model"}, indent=2))
    return EXIT["ok"]


# ---------------------------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu.py", description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("check", "prepare", "diagnose", "run", "status", "env-python", "rehearsal-judge"):
        stage = sub.add_parser(name)
        stage.add_argument("--config", type=Path, required=True)
        if name == "run":
            stage.add_argument("--extend-hours", type=float, help="after a time-limit stop (exit 6): move the deadline to now + HOURS and resume")
    c = sub.add_parser("cmd", help="run a devsearch command (review, bundle, status, ...) with this configuration's interpreter and runtime root")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("args", nargs=argparse.REMAINDER, help="after `--`: the devsearch module and its arguments, e.g. review progress --request ...")
    f = sub.add_parser("fetch-model", help="on a connected, authorized machine: download and verify the pinned model (stdlib only)")
    f.add_argument("--repo", default="microsoft/MiniLM-L12-H384-uncased")
    f.add_argument("--revision", required=True, help="a 40-hex commit, or 'main' (resolved once and printed)")
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--endpoint", default="https://huggingface.co")
    f.add_argument("--token-env", default="HF_TOKEN")
    m = sub.add_parser("standin-model", help="CPU rehearsal only: a tiny random encoder with the package's reference vocabulary")
    m.add_argument("--package", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("package", help="authoring: build the dataset package and record its frozen values")
    for name in ("sources", "data", "index", "drafts", "reference-tokenizer", "work", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--snapshots", type=Path, help="existing pinned source snapshots to reuse (offline)")
    p.add_argument("--notice")
    return parser


def main(argv=None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as error:
        return EXIT["usage"] if error.code else EXIT["ok"]
    try:
        if args.stage == "package":
            return stage_package(args)
        if args.stage == "fetch-model":
            return stage_fetch_model(args)
        if args.stage == "standin-model":
            return stage_standin_model(args)
        config = Config(args.config)
        rt = Runtime(config)
        if args.stage == "env-python":
            print(rt.env_python())
            return EXIT["ok"]
        if args.stage == "cmd":
            words = [a for a in args.args if a != "--"] if args.args[:1] == ["--"] else args.args
            if not words or not re.fullmatch(r"[a-z_]+", words[0]):
                raise Stop(EXIT["usage"], "usage: gpu.py cmd --config C -- <devsearch module> [arguments]  (e.g. -- review progress --request R ...)")
            if not rt.env_python().exists():
                raise prerequisite("environment", "the environment is not prepared yet: run `gpu.py prepare` first")
            return subprocess.run([str(rt.env_python()), "-m", f"devsearch.{words[0]}", *words[1:]], env=rt.child_env({"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None})).returncode
        if args.stage == "check":
            return stage_check(rt)
        if not config.root.is_dir():
            raise prerequisite("runtime-root", f"{config.root} does not exist; create it on the persistent volume first (`gpu.py check` lists everything)")
        with rt.locked():
            rt.log(f"stage {args.stage} (code {rt.code_key()}, profile {config.profile}, root {config.root})")
            if args.stage == "run":
                return stage_run(rt, args.extend_hours)
            return {"prepare": stage_prepare, "diagnose": stage_diagnose, "status": stage_status, "rehearsal-judge": stage_rehearsal_judge}[args.stage](rt)
    except Stop as stop:
        if stop.prerequisite:
            print(f"PREREQUISITE {stop.prerequisite}: {stop}", file=sys.stderr)
        else:
            print(f"gpu.py: {'error' if stop.code != EXIT['ok'] else 'note'}: {stop}", file=sys.stderr)
        with contextlib.suppress(Exception):
            if args.stage not in ("package", "fetch-model", "standin-model") and Path(config.root, "logs").is_dir():
                with Path(config.root, "logs", "gpu.log").open("a", encoding="utf-8") as handle:
                    handle.write(f"{utc()} stop exit {stop.code} {stop.prerequisite or ''}: {str(stop).splitlines()[0]}\n")
        return stop.code
    except KeyboardInterrupt:
        print("gpu.py: interrupted; rerun the same stage to resume", file=sys.stderr)
        return EXIT["failed"]


if __name__ == "__main__":
    raise SystemExit(main())
