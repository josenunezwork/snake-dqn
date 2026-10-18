"""Drive-safe data model for the Colab job queue.

Google Drive's sync client does not offer a useful cross-runtime atomic create:
simultaneous creates can become ``name`` and ``name (1)``.  This module therefore
uses immutable, uniquely named claim/receipt files and treats duplicate aliases
as input to scan, never as a locking primitive.  Mac-side commands serialize
their writes with :func:`mac_writer_lock`; runtime workers only write unique
claim, heartbeat, and receipt files.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

QUEUE_DIRS = ("pending", "claimed", "done", "failed", "heartbeats", "control", "locks")
GPU_KINDS = {"any", "A100", "L4", "cpu"}
TERMINAL_STATES = {"done", "failed"}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_ALIAS = re.compile(r"^(?P<base>.+?)(?: \((?P<number>[1-9][0-9]*)\))?$")
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QueueError(RuntimeError):
    """A malformed or unsafe queue operation."""


class QueueConflict(QueueError):
    """Drive aliases contain non-identical records for one logical object."""


class JobValidationError(QueueError):
    """A job specification does not match the portable queue schema."""


def utc_now() -> str:
    """Return an unambiguous UTC timestamp suitable for JSON receipts."""
    return (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueConflict(f"cannot read queue JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueConflict(f"queue JSON must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> Path:
    """Atomically replace a local/Drive JSON file; its parent must already exist."""
    if not path.parent.is_dir():
        raise QueueError(f"queue directory was not pre-created: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    return path


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise JobValidationError(f"{field} must be a non-empty relative POSIX path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise JobValidationError(f"{field} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise JobValidationError(f"{field} must stay below the repository/Drive root")
    return path.as_posix()


def validate_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a portable job specification.

    The returned value contains only documented fields so that typos cannot turn
    into silently ignored scheduling constraints.
    """
    if not isinstance(spec, dict):
        raise JobValidationError("job spec must be a JSON object")
    required = {
        "id",
        "commit",
        "script",
        "env",
        "requirements",
        "depends_on",
        "output_dir",
        "priority",
    }
    optional = {"timeout_minutes", "output_subdirs", "queue_generation", "attempt", "created_at"}
    unknown = set(spec) - required - optional
    missing = required - set(spec)
    if missing or unknown:
        raise JobValidationError(f"job fields missing={sorted(missing)} unknown={sorted(unknown)}")
    job_id = spec["id"]
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise JobValidationError(
            "id must contain only letters, digits, dot, underscore, and hyphen"
        )
    commit = spec["commit"]
    if not isinstance(commit, str) or not _SHA.fullmatch(commit):
        raise JobValidationError("commit must be a full 40-character Git SHA")
    env = spec["env"]
    if not isinstance(env, dict) or any(
        not isinstance(k, str) or not _ENV_KEY.fullmatch(k) or not isinstance(v, str) or "\0" in v
        for k, v in env.items()
    ):
        raise JobValidationError("env must be a string-to-string object")
    requirements = spec["requirements"]
    required_requirements = {"gpu", "min_vcpu", "est_minutes"}
    if not isinstance(requirements, dict) or set(requirements) != required_requirements:
        raise JobValidationError("requirements must contain exactly gpu, min_vcpu, est_minutes")
    gpu = requirements["gpu"]
    if not isinstance(gpu, str) or gpu not in GPU_KINDS:
        raise JobValidationError(f"requirements.gpu must be one of {sorted(GPU_KINDS)}")
    if type(requirements["min_vcpu"]) is not int or requirements["min_vcpu"] < 1:
        raise JobValidationError("requirements.min_vcpu must be a positive integer")
    if type(requirements["est_minutes"]) is not int or requirements["est_minutes"] < 1:
        raise JobValidationError("requirements.est_minutes must be a positive integer")
    dependencies = spec["depends_on"]
    if not isinstance(dependencies, list) or any(
        not isinstance(x, str) or not _ID.fullmatch(x) for x in dependencies
    ):
        raise JobValidationError("depends_on must be a list of valid job ids")
    if job_id in dependencies or len(set(dependencies)) != len(dependencies):
        raise JobValidationError("depends_on cannot include self or duplicates")
    if type(spec["priority"]) is not int:
        raise JobValidationError("priority must be an integer")
    normalized: dict[str, Any] = {
        "id": job_id,
        "commit": commit.lower(),
        "script": _relative_path(spec["script"], "script"),
        "env": dict(sorted(env.items())),
        "requirements": {key: requirements[key] for key in sorted(required_requirements)},
        "depends_on": dependencies,
        "output_dir": _relative_path(spec["output_dir"], "output_dir"),
        "priority": spec["priority"],
    }
    if "timeout_minutes" in spec:
        timeout = spec["timeout_minutes"]
        if type(timeout) is not int or timeout < 1:
            raise JobValidationError("timeout_minutes must be a positive integer")
        normalized["timeout_minutes"] = timeout
    if "output_subdirs" in spec:
        subs = spec["output_subdirs"]
        if not isinstance(subs, list) or any(not isinstance(x, str) for x in subs):
            raise JobValidationError("output_subdirs must be a list of relative paths")
        normalized["output_subdirs"] = [
            _relative_path(item, "output_subdirs item") for item in subs
        ]
    for key in ("queue_generation", "attempt"):
        if key in spec:
            if type(spec[key]) is not int or spec[key] < 0:
                raise JobValidationError(f"{key} must be a non-negative integer")
            normalized[key] = spec[key]
    if "created_at" in spec:
        if not isinstance(spec["created_at"], str):
            raise JobValidationError("created_at must be a string")
        normalized["created_at"] = spec["created_at"]
    return normalized


class Queue:
    """Queue filesystem facade, rooted at the Drive ``t16_colab/queue`` directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    @property
    def pending_dir(self) -> Path:
        return self.root / "pending"

    @property
    def claimed_dir(self) -> Path:
        return self.root / "claimed"

    @property
    def done_dir(self) -> Path:
        return self.root / "done"

    @property
    def failed_dir(self) -> Path:
        return self.root / "failed"

    @property
    def heartbeats_dir(self) -> Path:
        return self.root / "heartbeats"

    @property
    def control_dir(self) -> Path:
        return self.root / "control"

    @property
    def locks_dir(self) -> Path:
        return self.root / "locks"

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in QUEUE_DIRS:
            (self.root / name).mkdir(exist_ok=True)

    def _aliases(self, parent: Path, name: str, *, directories: bool | None = None) -> list[Path]:
        """Return name and Drive duplicate aliases (``name (N)``), sorted."""
        if not parent.is_dir():
            raise QueueConflict(f"queue parent is missing: {parent}")
        matcher = re.compile(rf"^{re.escape(name)}(?: \([1-9][0-9]*\))?$")
        paths = [entry for entry in parent.iterdir() if matcher.fullmatch(entry.name)]
        if directories is not None:
            paths = [entry for entry in paths if entry.is_dir() == directories]
        return sorted(paths, key=lambda path: path.name)

    def _scan_dirs(self, name: str) -> list[Path]:
        roots = self._aliases(self.root.parent, self.root.name, directories=True)
        paths = [path for root in roots for path in self._aliases(root, name, directories=True)]
        if not paths:
            raise QueueConflict(f"shared queue directory is missing: {self.root / name}")
        return paths

    def _same_by_key(self, paths: list[Path], key: str, label: str) -> dict[str, dict[str, Any]]:
        found: dict[str, tuple[dict[str, Any], Path]] = {}
        for path in paths:
            data = read_json(path)
            identity = data.get(key)
            if not isinstance(identity, str):
                raise QueueConflict(f"{label} lacks {key}: {path}")
            if identity in found and _json_bytes(found[identity][0]) != _json_bytes(data):
                raise QueueConflict(
                    f"conflicting duplicate {label} {identity}: {found[identity][1]} and {path}"
                )
            found[identity] = (data, path)
        return {identity: pair[0] for identity, pair in found.items()}

    def _records(self, directory: str, pattern: str = "*.json") -> list[Path]:
        return [
            path
            for folder in self._scan_dirs(directory)
            for path in sorted(folder.glob(pattern))
            if path.is_file()
        ]

    def pending_path(self, job_id: str) -> Path:
        return self.pending_dir / f"{job_id}.json"

    def control_path(self, job_id: str) -> Path:
        return self.control_dir / f"{job_id}.json"

    def claim_path(self, job_id: str, generation: int, worker_id: str, token: str) -> Path:
        return self.claimed_dir / job_id / f"g{generation}.{worker_id}.{token}.json"

    def receipt_path(self, state: str, job_id: str, generation: int, token: str) -> Path:
        if state not in TERMINAL_STATES:
            raise QueueError(f"not a terminal state: {state}")
        return self.root / state / job_id / f"g{generation}.{token}.json"

    def ensure_job_dirs(self, job_id: str) -> None:
        if not _ID.fullmatch(job_id):
            raise QueueError(f"invalid job id: {job_id}")
        for directory in (self.claimed_dir, self.done_dir, self.failed_dir):
            (directory / job_id).mkdir(parents=True, exist_ok=True)

    def enqueue(self, spec: dict[str, Any]) -> dict[str, Any]:
        job = validate_job(spec)
        if {"queue_generation", "attempt", "created_at"} & set(job):
            raise JobValidationError(
                "queue generation, attempt and created_at are managed by enqueue"
            )
        self.ensure_layout()
        job.setdefault("queue_generation", 0)
        job.setdefault("attempt", 0)
        job.setdefault("created_at", utc_now())
        existing = self.load_jobs()
        if job["id"] in existing:
            present = existing[job["id"]]
            if _json_bytes(present) != _json_bytes(job):
                raise QueueConflict(f"job id already exists: {job['id']}")
            return present
        output = (self.root.parent / job["output_dir"]).resolve()
        drive_root = self.root.parent.resolve()
        output_top = PurePosixPath(job["output_dir"]).parts[0]
        if (
            drive_root not in output.parents
            or output == self.root.resolve()
            or self.root.resolve() in output.parents
            or re.fullmatch(r"queue(?: \([1-9][0-9]*\))?", output_top)
        ):
            raise JobValidationError("output_dir must be inside Drive root and outside queue")
        for old in existing.values():
            old_output = (self.root.parent / old["output_dir"]).resolve()
            if output == old_output or output in old_output.parents or old_output in output.parents:
                raise QueueConflict(f"output_dir overlaps existing job {old['id']}")
        self.ensure_job_dirs(job["id"])
        output.mkdir(parents=True, exist_ok=True)
        for child in job.get("output_subdirs", []):
            (output / child).mkdir(parents=True, exist_ok=True)
        # Publish only after all workers' shared/output directories exist.
        write_json(self.pending_path(job["id"]), job)
        return job

    def load_jobs(self) -> dict[str, dict[str, Any]]:
        jobs = self._same_by_key(self._records("pending"), "id", "pending job")
        result: dict[str, dict[str, Any]] = {}
        for job_id, data in jobs.items():
            normalized = validate_job(data)
            if normalized != data:
                raise QueueConflict(f"pending job is not normalized: {job_id}")
            result[job_id] = data
        return result

    def control(self, job_id: str) -> dict[str, Any]:
        paths: list[Path] = []
        for directory in self._scan_dirs("control"):
            for path in directory.glob("*.json"):
                if not path.is_file():
                    continue
                record = read_json(path)
                if not isinstance(record.get("id"), str):
                    raise QueueConflict(f"control record lacks id: {path}")
                if record["id"] == job_id:
                    paths.append(path)
        if not paths:
            return {"id": job_id, "generation": 0, "cancelled": False, "updated_at": None}
        entries = self._same_by_key(paths, "id", "control record")
        if set(entries) != {job_id}:
            raise QueueConflict(f"unexpected control identity for {job_id}")
        control = entries[job_id]
        if not isinstance(control.get("generation"), int) or control["generation"] < 0:
            raise QueueConflict(f"invalid control generation: {job_id}")
        if not isinstance(control.get("cancelled"), bool):
            raise QueueConflict(f"invalid cancelled value: {job_id}")
        return control

    def update_control(
        self, job_id: str, *, cancelled: bool, generation: int | None = None, reason: str = ""
    ) -> dict[str, Any]:
        current = self.control(job_id)
        record = {
            "id": job_id,
            "generation": current["generation"] if generation is None else generation,
            "cancelled": cancelled,
            "updated_at": utc_now(),
            "reason": reason,
        }
        write_json(self.control_path(job_id), record)
        return record

    def claims(self, job_id: str, generation: int | None = None) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for claimed_dir in self._scan_dirs("claimed"):
            for folder in self._aliases(claimed_dir, job_id, directories=True):
                for path in sorted(folder.glob("*.json")):
                    data = read_json(path)
                    if data.get("id") != job_id:
                        raise QueueConflict(f"claim identity does not match directory: {path}")
                    if (
                        not isinstance(data.get("worker_id"), str)
                        or not _ID.fullmatch(data["worker_id"])
                        or not isinstance(data.get("token"), str)
                        or not _ID.fullmatch(data["token"])
                        or type(data.get("generation")) is not int
                        or data["generation"] < 0
                    ):
                        raise QueueConflict(f"malformed claim: {path}")
                    if generation is None or data.get("generation") == generation:
                        records.append(data)
        # A token is globally unique for an attempt.  Non-identical copies fail closed.
        keyed: dict[str, dict[str, Any]] = {}
        for claim in records:
            token = claim.get("token")
            if not isinstance(token, str):
                raise QueueConflict(f"claim has no token for {job_id}")
            if token in keyed and _json_bytes(keyed[token]) != _json_bytes(claim):
                raise QueueConflict(f"conflicting duplicate claim token {token} for {job_id}")
            keyed[token] = claim
        return sorted(
            keyed.values(), key=lambda record: (str(record.get("worker_id")), str(record["token"]))
        )

    def write_claim(
        self, job: dict[str, Any], worker_id: str, token: str, capabilities: dict[str, Any]
    ) -> dict[str, Any]:
        if not _ID.fullmatch(worker_id) or not _ID.fullmatch(token):
            raise QueueError("worker_id and token must use queue id characters")
        control = self.control(job["id"])
        generation = job["queue_generation"]
        if control["cancelled"] or control["generation"] != generation:
            raise QueueError("job generation is no longer claimable")
        if self.current_terminal_receipt(job["id"], generation):
            raise QueueError("job generation already has a terminal result")
        winner = self.winner(job["id"], generation)
        if winner:
            heartbeat = self.heartbeat_status().get(winner["worker_id"], {})
            if (
                heartbeat.get("job_id") == job["id"]
                and heartbeat.get("token") == winner["token"]
                and heartbeat.get("phase") in {"setup", "running", "finishing"}
            ):
                raise QueueError("job owner has already started; late contender refused")
        claim = {
            "id": job["id"],
            "generation": generation,
            "attempt": job.get("attempt", generation),
            "worker_id": worker_id,
            "token": token,
            "created_at": utc_now(),
            "capabilities": capabilities,
        }
        path = self.claim_path(job["id"], generation, worker_id, token)
        if path.exists():
            existing = read_json(path)
            if _json_bytes(existing) != _json_bytes(claim):
                raise QueueConflict(f"claim token already exists with different data: {path}")
            return existing
        write_json(path, claim)
        return claim

    def winner(self, job_id: str, generation: int | None = None) -> dict[str, Any] | None:
        target_generation = self.control(job_id)["generation"] if generation is None else generation
        entries = self.claims(job_id, target_generation)
        return entries[0] if entries else None

    def is_winner(self, job: str | dict[str, Any], worker_id: str, token: str) -> bool:
        job_id = job["id"] if isinstance(job, dict) else job
        generation = (
            job["queue_generation"] if isinstance(job, dict) else self.control(job_id)["generation"]
        )
        chosen = self.winner(job_id, generation)
        return bool(chosen and chosen["worker_id"] == worker_id and chosen["token"] == token)

    def receipts(self, job_id: str, generation: int | None = None) -> list[dict[str, Any]]:
        all_receipts: list[dict[str, Any]] = []
        for state in TERMINAL_STATES:
            for folder in self._scan_dirs(state):
                for job_dir in self._aliases(folder, job_id, directories=True):
                    for path in sorted(job_dir.glob("*.json")):
                        receipt = read_json(path)
                        if receipt.get("id") != job_id or receipt.get("status") != state:
                            raise QueueConflict(f"invalid terminal receipt: {path}")
                        if generation is None or receipt.get("generation") == generation:
                            all_receipts.append(receipt)
        keyed: dict[tuple[str, str], dict[str, Any]] = {}
        for receipt in all_receipts:
            token = receipt.get("token")
            status = receipt.get("status")
            if not isinstance(token, str) or not isinstance(status, str):
                raise QueueConflict(f"terminal receipt lacks token/status for {job_id}")
            key = (status, token)
            if key in keyed and _json_bytes(keyed[key]) != _json_bytes(receipt):
                raise QueueConflict(f"conflicting duplicate receipt {key} for {job_id}")
            keyed[key] = receipt
        return sorted(keyed.values(), key=lambda receipt: (receipt["token"], receipt["status"]))

    def write_receipt(
        self, job_id: str, generation: int, token: str, status: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if status not in TERMINAL_STATES or not _ID.fullmatch(token):
            raise QueueError("invalid terminal receipt arguments")
        if not isinstance(payload, dict):
            raise QueueError("receipt payload must be an object")
        receipt = {
            **payload,
            "id": job_id,
            "generation": generation,
            "token": token,
            "status": status,
            "recorded_at": utc_now(),
        }
        # Retain stale receipts for audit, but callers must consider only current generation.
        path = self.receipt_path(status, job_id, generation, token)
        if not path.parent.is_dir():
            raise QueueError(f"job terminal directory was not pre-created: {path.parent}")
        if path.exists():
            return read_json(path)
        write_json(path, receipt)
        return receipt

    def current_terminal_receipt(
        self, job_id: str, generation: int, expected_status: str | None = None
    ) -> dict[str, Any] | None:
        """Return only the current deterministic winner's terminal receipt.

        Losers may write failed receipts after discovering a lower worker id.  They
        remain useful audit evidence but must never fence the winner's result.
        """
        winner = self.winner(job_id, generation)
        if winner is None:
            return None
        matching = [
            receipt
            for receipt in self.receipts(job_id, generation)
            if receipt.get("token") == winner["token"]
        ]
        if len(matching) > 1:
            raise QueueConflict(f"conflicting done/failed receipts for current winner of {job_id}")
        if not matching or (
            expected_status is not None and matching[0].get("status") != expected_status
        ):
            return None
        return matching[0]

    def dependencies_done(
        self, job: dict[str, Any], jobs: dict[str, dict[str, Any]] | None = None
    ) -> bool:
        jobs = self.load_jobs() if jobs is None else jobs
        for dependency in job["depends_on"]:
            if dependency not in jobs:
                return False
            dependency_control = self.control(dependency)
            if dependency_control["cancelled"] or not self.current_terminal_receipt(
                dependency, dependency_control["generation"], "done"
            ):
                return False
        return True

    def eligible_jobs(self, capabilities: dict[str, Any]) -> list[dict[str, Any]]:
        """Eligible jobs ordered by highest priority then stable id.

        A claim, even with an old heartbeat, excludes a job.  Manual requeue is
        the only recovery path; this avoids two living runtimes executing one job.
        ``gpu=any`` requires an actual non-CPU GPU.  ``gpu=cpu`` only matches a
        CPU runtime, preserving paid GPU capacity for GPU work.
        """
        gpu = capabilities.get("gpu")
        vcpu = capabilities.get("vcpu")
        if not isinstance(gpu, str) or not gpu or not isinstance(vcpu, int):
            raise QueueError("capabilities require a non-empty gpu string and integer vcpu")
        jobs = self.load_jobs()
        eligible: list[dict[str, Any]] = []
        for job in jobs.values():
            control = self.control(job["id"])
            generation = control["generation"]
            if job.get("queue_generation") != generation:
                continue
            requirement = job["requirements"]
            compatible = (requirement["gpu"] == gpu) or (
                requirement["gpu"] == "any" and gpu != "cpu"
            )
            if control["cancelled"] or not compatible or vcpu < requirement["min_vcpu"]:
                continue
            if self.claims(job["id"], generation) or self.current_terminal_receipt(
                job["id"], generation
            ):
                continue
            if self.dependencies_done(job, jobs):
                eligible.append(job)
        return sorted(eligible, key=lambda job: (-job["priority"], job["id"]))

    def write_heartbeat(self, worker_id: str, payload: dict[str, Any]) -> Path:
        if not _ID.fullmatch(worker_id) or not isinstance(payload, dict):
            raise QueueError("invalid heartbeat")
        record = {"worker_id": worker_id, "updated_at": utc_now(), **payload}
        return write_json(self.heartbeats_dir / f"{worker_id}.json", record)

    def heartbeat_status(
        self, stale_seconds: int = 300, now: float | None = None
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(stale_seconds, int) or stale_seconds < 1:
            raise QueueError("stale_seconds must be positive")
        current = time.time() if now is None else now
        records = self._same_by_key(self._records("heartbeats"), "worker_id", "heartbeat")
        status: dict[str, dict[str, Any]] = {}
        for worker_id, heartbeat in records.items():
            try:
                stamp = dt.datetime.fromisoformat(
                    heartbeat["updated_at"].replace("Z", "+00:00")
                ).timestamp()
            except (KeyError, AttributeError, ValueError) as exc:
                raise QueueConflict(f"invalid heartbeat timestamp for {worker_id}") from exc
            status[worker_id] = {
                **heartbeat,
                "age_seconds": max(0, current - stamp),
                "stale": current - stamp > stale_seconds,
            }
        return status

    def job_status(self, job_id: str, stale_seconds: int = 300) -> dict[str, Any]:
        job = self.load_jobs().get(job_id)
        if job is None:
            raise QueueError(f"unknown job: {job_id}")
        control = self.control(job_id)
        generation = control["generation"]
        claims = self.claims(job_id, generation)
        receipts = self.receipts(job_id, generation)
        terminal = self.current_terminal_receipt(job_id, generation)
        workers = self.heartbeat_status(stale_seconds)
        claim_workers = {claim["worker_id"]: workers.get(claim["worker_id"]) for claim in claims}
        if control["cancelled"]:
            state = "cancelled"
        elif terminal:
            state = terminal["status"]
        elif claims:
            state = "claimed"
        else:
            state = "pending"
        return {
            "id": job_id,
            "state": state,
            "generation": generation,
            "attempt": job.get("attempt", generation),
            "cancelled": control["cancelled"],
            "claims": claims,
            "receipts": receipts,
            "claim_heartbeats": claim_workers,
        }

    def cancel(self, job_id: str, reason: str = "cancelled by operator") -> dict[str, Any]:
        if job_id not in self.load_jobs():
            raise QueueError(f"unknown job: {job_id}")
        return self.update_control(job_id, cancelled=True, reason=reason)

    def requeue(self, job_id: str, *, stale_owner_ack: bool, reason: str = "") -> dict[str, Any]:
        if not stale_owner_ack:
            raise QueueError("requeue requires acknowledgement that any stale owner has stopped")
        job = self.load_jobs().get(job_id)
        if job is None:
            raise QueueError(f"unknown job: {job_id}")
        current = self.control(job_id)
        detail = self.job_status(job_id)
        status = detail["state"]
        if status not in {"failed", "cancelled", "claimed"}:
            raise QueueError(
                f"only failed, cancelled, or stale claimed jobs may be requeued (current {status})"
            )
        if status == "claimed":
            winner = self.winner(job_id, current["generation"])
            heartbeat = detail["claim_heartbeats"].get(winner["worker_id"]) if winner else None
            if (
                heartbeat
                and not heartbeat["stale"]
                and heartbeat.get("job_id") == job_id
                and heartbeat.get("token") == winner["token"]
            ):
                raise QueueError("winning claim heartbeat is fresh; wait for it or cancel instead")
        generation = current["generation"] + 1
        updated = {
            **job,
            "queue_generation": generation,
            "attempt": job.get("attempt", current["generation"]) + 1,
        }
        write_json(self.pending_path(job_id), updated)
        return self.update_control(
            job_id, cancelled=False, generation=generation, reason=reason or "operator requeue"
        )


@contextlib.contextmanager
def mac_writer_lock(queue: Queue) -> Iterator[None]:
    """Serialize this Mac's CLI mutations; it is deliberately not on Drive."""
    lock_root = Path(tempfile.gettempdir()) / "snake-dqn-colab-queue-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(str(queue.root.resolve()).encode("utf-8")).hexdigest()
    lock_path = lock_root / f"{identity}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
