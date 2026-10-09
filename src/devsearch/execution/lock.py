"""Single-owner leases: the campaign lock and the accelerator lease (docs/10 §5).

A lease is a file created with `O_CREAT | O_EXCL` that names its owner (host, pid, process start token and a
random token). A stale lease is recovered only after its *identity* is checked, never just its age:

* same host and the process is gone, or the pid now belongs to a process with a different start time -> stale;
* same host and the same process is alive -> held (a second owner is refused, also within this process);
* another host, or an unreadable file -> held: identity cannot be verified, so nothing is broken silently.

Releasing removes the file only if it still carries this owner's token. Recovery moves the stale file aside
(never deletes evidence) and retries creation once; if two recoverers race, exactly one creation wins.
"""

import json
import os
import socket
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.errors import PrerequisiteError


class LeaseHeld(PrerequisiteError):
    """Another owner holds the lease, or its owner cannot be verified as gone."""


def host_name() -> str:
    return socket.gethostname()


def start_token(pid: int) -> str | None:
    """A value that changes if the pid is reused by a different process; None when it cannot be read."""
    stat = Path(f"/proc/{pid}/stat")
    try:
        if stat.exists():
            return stat.read_text().rsplit(")", 1)[1].split()[19]        # field 22: start time in clock ticks since boot
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() or None
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                  # exists, owned by someone else
    return True


@dataclass(frozen=True)
class Owner:
    host: str
    pid: int
    start: str | None
    token: str
    purpose: str
    plan_id: str

    def to_record(self) -> dict[str, Any]:
        return {"host": self.host, "pid": self.pid, "start": self.start, "token": self.token, "purpose": self.purpose, "plan_id": self.plan_id}


def _parse(text: str) -> Owner | None:
    try:
        data = json.loads(text)
        return Owner(str(data["host"]), int(data["pid"]), data["start"], str(data["token"]), str(data["purpose"]), str(data["plan_id"]))
    except (ValueError, KeyError, TypeError):
        return None


class Lease:
    def __init__(self, path: Path, *, purpose: str, plan_id: str) -> None:
        self.path = Path(path)
        self.purpose = purpose
        self.plan_id = plan_id
        self.owner: Owner | None = None

    def _mine(self) -> Owner:
        pid = os.getpid()
        return Owner(host_name(), pid, start_token(pid), uuid.uuid4().hex, self.purpose, self.plan_id)

    def _create(self, owner: Owner) -> bool:
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(owner.to_record(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True

    def _is_stale(self, held: Owner) -> bool:
        if held.host != host_name():
            return False
        if not process_alive(held.pid):
            return True
        current = start_token(held.pid)
        return held.start is not None and current is not None and current != held.start

    def acquire(self) -> Owner:
        if self.owner is not None:
            raise LeaseHeld(f"{self.path.name} is already held by this object")
        mine = self._mine()
        if self._create(mine):
            self.owner = mine
            return mine
        try:
            held = _parse(self.path.read_text(encoding="utf-8"))
        except OSError as error:
            raise LeaseHeld(f"{self.path} exists but cannot be read: {error}") from error
        if held is None:
            raise LeaseHeld(f"{self.path} exists but is unreadable; its owner cannot be verified, so it was not broken")
        if not self._is_stale(held):
            raise LeaseHeld(
                f"{self.path.name} is held by {held.purpose!r} (host {held.host}, pid {held.pid}, plan {held.plan_id}); "
                "a second owner is refused" if held.host == host_name() else
                f"{self.path.name} is held by host {held.host}, pid {held.pid}; this host cannot verify that owner is gone")
        aside = self.path.with_name(f"{self.path.name}.stale-{held.token[:12]}")
        try:
            os.rename(self.path, aside)                              # atomic: only one recoverer moves the same file
        except FileNotFoundError:
            pass
        if not self._create(mine):
            raise LeaseHeld(f"{self.path.name} was re-acquired by another process during recovery")
        self.owner = mine
        return mine

    def release(self) -> None:
        if self.owner is None:
            return
        try:
            current = _parse(self.path.read_text(encoding="utf-8"))
            if current is not None and current.token == self.owner.token:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.owner = None

    def __enter__(self) -> "Lease":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
