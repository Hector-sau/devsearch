"""Explicit device profiles, resolution without silent fallback, fingerprints and memory readings.

* A profile names one device. An unavailable requested accelerator raises `DeviceUnavailable`
  with the specific reason; it never turns into CPU.
* A formal MPS run refuses `PYTORCH_ENABLE_MPS_FALLBACK=1`, because CPU operator fallback would
  hide exactly what the backend gate must expose.
* TF32 on CUDA is set explicitly from the profile and recorded.
* `verification` in a profile is documentation; the authority is the acceptance report.
"""

import os
import platform
import resource
import sys
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from devsearch.contracts import PROFILE_NAMES
from devsearch.errors import ConfigError, PrerequisiteError

_KEYS = (
    "schema_version", "name", "device", "accelerator", "precision", "tf32", "workers", "effective_batch",
    "train_microbatches", "inference_batches", "support_scope", "memory_note", "verification",
)


class DeviceUnavailable(PrerequisiteError):
    """The profile's device cannot be used on this host (specific reason in the message)."""


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    device: str
    accelerator: bool
    precision: str
    tf32: bool
    workers: int
    effective_batch: int
    train_microbatches: tuple[int, ...]
    inference_batches: tuple[int, ...]
    support_scope: str
    memory_note: str
    verification: str


def _ints(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in value):
        raise ConfigError(f"{where}: expected a non-empty list of positive integers")
    if len(set(value)) != len(value):
        raise ConfigError(f"{where}: duplicates")
    return tuple(value)


def parse_profile(table: dict[str, Any], where: str) -> DeviceProfile:
    unknown, missing = sorted(table.keys() - set(_KEYS)), sorted(set(_KEYS) - table.keys())
    if unknown or missing:
        raise ConfigError(f"{where}: unknown key(s) {unknown} / missing key(s) {missing}")
    if table["schema_version"] != 1 or table["name"] not in PROFILE_NAMES:
        raise ConfigError(f"{where}: unsupported schema_version or profile name {table['name']!r}")
    if table["device"] not in ("mps", "cuda:0", "cpu") or table["precision"] != "fp32":
        raise ConfigError(f"{where}: device must be mps, cuda:0 or cpu and precision fp32 (AMP is a separate future profile)")
    if table["accelerator"] != (table["device"] != "cpu") or not isinstance(table["tf32"], bool):
        raise ConfigError(f"{where}: accelerator flag must match the device; tf32 must be a boolean")
    if table["tf32"] and table["device"] != "cuda:0":
        raise ConfigError(f"{where}: tf32 applies to CUDA only")
    if table["workers"] != 0 or isinstance(table["workers"], bool):
        raise ConfigError(f"{where}: workers must be 0 in the initial profiles")
    if isinstance(table["effective_batch"], bool) or not isinstance(table["effective_batch"], int) or table["effective_batch"] < 1:
        raise ConfigError(f"{where}: effective_batch must be a positive integer")
    if table["verification"] not in ("unverified", "verified", "not_applicable"):
        raise ConfigError(f"{where}: verification must be unverified, verified or not_applicable")
    if table["accelerator"] is False and table["verification"] == "verified":
        raise ConfigError(f"{where}: a CPU diagnostic profile can never be an accelerator verification")
    return DeviceProfile(
        table["name"], table["device"], table["accelerator"], table["precision"], table["tf32"], table["workers"],
        table["effective_batch"], _ints(table["train_microbatches"], f"{where}: train_microbatches"),
        _ints(table["inference_batches"], f"{where}: inference_batches"), str(table["support_scope"]),
        str(table["memory_note"]), table["verification"],
    )


def load_profile(name_or_path: str | Path) -> DeviceProfile:
    """A packaged profile by name, or a TOML file by path (used for tests and local overrides)."""
    candidate = Path(str(name_or_path))
    if str(name_or_path) in PROFILE_NAMES:
        text = resources.files("devsearch.profiles").joinpath(f"{name_or_path}.toml").read_text(encoding="utf-8")
        where = f"profile {name_or_path}"
    elif candidate.is_file():
        text, where = candidate.read_text(encoding="utf-8"), str(candidate)
    else:
        raise ConfigError(f"unknown profile {name_or_path!r}; choose one of {', '.join(PROFILE_NAMES)} or give a TOML path")
    try:
        return parse_profile(tomllib.loads(text), where)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{where}: invalid TOML: {error}") from error


def resolve_device(profile: DeviceProfile) -> torch.device:
    """The profile's device, verified with one real tiny operation, or DeviceUnavailable. Never CPU by accident."""
    if profile.device == "cpu":
        device = torch.device("cpu")
    elif profile.device == "mps":
        if not torch.backends.mps.is_built():
            raise DeviceUnavailable("this PyTorch build was not compiled with MPS support")
        if not torch.backends.mps.is_available():
            raise DeviceUnavailable("MPS is built but not available on this host (not macOS arm64 or the OS is too old)")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise DeviceUnavailable("PYTORCH_ENABLE_MPS_FALLBACK=1 would let operators silently run on CPU; formal MPS runs refuse it")
        device = torch.device("mps")
    else:
        if torch.version.cuda is None:
            raise DeviceUnavailable("this PyTorch build was compiled without CUDA")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise DeviceUnavailable("CUDA is built but no usable NVIDIA GPU/driver is available on this host")
        torch.backends.cuda.matmul.allow_tf32 = profile.tf32
        torch.backends.cudnn.allow_tf32 = profile.tf32
        device = torch.device("cuda:0")
    probe = torch.ones((2, 2), device=device) @ torch.ones((2, 2), device=device)
    synchronize(device)
    if not bool(torch.isfinite(probe).all().item()) or float(probe.sum().item()) != 8.0:
        raise DeviceUnavailable(f"a tiny matrix multiply on {device} returned a wrong or non-finite result")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats(device: torch.device) -> dict[str, int]:
    """Bytes, labeled by what they mean on each backend (unified memory is not VRAM)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {"host_peak_rss_bytes": usage if sys.platform == "darwin" else usage * 1024}
    if device.type == "cuda":
        stats.update(device_allocated_bytes=torch.cuda.memory_allocated(device), device_reserved_bytes=torch.cuda.memory_reserved(device),
                     device_peak_allocated_bytes=torch.cuda.max_memory_allocated(device), device_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    elif device.type == "mps":
        stats.update(device_allocated_bytes=torch.mps.current_allocated_memory(), device_driver_allocated_bytes=torch.mps.driver_allocated_memory())
    return stats


def backend_fingerprint(profile: DeviceProfile, device: torch.device) -> dict[str, Any]:
    """Exact software/hardware identity a backend verification must be tied to."""
    fingerprint: dict[str, Any] = {
        "profile": profile.name, "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "tf32": profile.tf32,
        "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        fingerprint.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(device),
                           capability=f"{capability[0]}.{capability[1]}", total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    elif device.type == "mps":
        fingerprint.update(macos=platform.mac_ver()[0], mps_built=torch.backends.mps.is_built())
        recommended = getattr(torch.mps, "recommended_max_memory", None)
        if recommended is not None:
            fingerprint["recommended_max_memory_bytes"] = int(recommended())
    return fingerprint
