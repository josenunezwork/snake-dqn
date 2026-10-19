"""Conservative, single-study admission ledger for S1 mature collection.

The ledger is deliberately Mac-only.  It never starts a Colab process: callers reserve
*before* invoking the CLI supervisor and settle only from its create-only receipt after
the backend assignment has disappeared.  A missing, malformed, failed, or changed receipt
keeps the full reservation outstanding.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

SCHEMA = "s1-mature-ledger/v1"
REVIEW_SCHEMA = "s1-cli-acceptance-review/v1"
USABLE_CAPS = {
    "units": 48.0,
    "gpu_allocation_hours": 20.0,
    "mac_worker_hours": 19.2,
    "elapsed_hours": 38.4,
}
NOMINAL_CAPS = {
    "units": 60.0,
    "gpu_allocation_hours": 25.0,
    "mac_worker_hours": 24.0,
    "elapsed_hours": 48.0,
}
SNAPSHOT_MAX_AGE_SECONDS = 15 * 60
DIMENSIONS = tuple(USABLE_CAPS)
OPERATIONAL_FAILURES = frozenset({"disconnect", "transfer_failure", "runtime_reclaim"})


class Refused(RuntimeError):
    """Evidence or the worst-case budget does not admit the requested action."""


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _number(value: Any, name: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Refused(f"{name} must be a finite number")
    value = float(value)
    if nonnegative and value < 0:
        raise Refused(f"{name} must be nonnegative")
    return value


def _bound(value: Mapping[str, Any], name: str) -> dict[str, float]:
    if set(value) != set(DIMENSIONS):
        raise Refused(f"{name} must contain exactly {', '.join(DIMENSIONS)}")
    return {key: _number(value[key], f"{name}.{key}") for key in DIMENSIONS}


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".ledger-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


class Ledger:
    def __init__(self, path: str | Path, *, clock=time.time) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.clock = clock

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not self.path.exists():
                    raise Refused("ledger has not been initialized")
                state = json.loads(self.path.read_text())
                if state.get("schema") != SCHEMA:
                    raise Refused("unrecognized ledger schema")
                yield state
                _atomic_write(self.path, state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _evidence(path: str | Path) -> dict[str, str]:
        target = Path(path).resolve(strict=True)
        return {"path": str(target), "sha256": sha256(target)}

    def initialize(
        self,
        *,
        acceptance_review_path: str | Path,
        account_snapshot_path: str | Path,
        history_refs: Mapping[str, str | Path],
        planned_liabilities: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Create the ledger once.  The full plan becomes immutable at initialization."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self.path.exists():
                raise Refused("ledger already exists; never reset or replace it")
            if not history_refs:
                raise Refused("historical Q1/A2/A3 evidence references are required")
            planned = {
                job: _bound(bound, f"planned_liabilities.{job}")
                for job, bound in planned_liabilities.items()
            }
            if any(job.startswith("retry-") for job in planned):
                raise Refused("operational reruns are conditional, never pre-funded")
            if not planned or len(planned) != len(set(planned)):
                raise Refused("at least one uniquely named planned liability is required")
            review, acceptance_actual, study_started_at = self._review_acceptance(
                acceptance_review_path
            )
            snapshot = self._snapshot(
                account_snapshot_path,
                require_zero=True,
                remaining_units=sum(x["units"] for x in planned.values()),
            )
            evidence = {name: self._evidence(path) for name, path in history_refs.items()}
            now = self.clock()
            state = {
                "schema": SCHEMA,
                "created_at": study_started_at,
                "ledger_initialized_at": now,
                "nominal_caps": NOMINAL_CAPS,
                "usable_caps": USABLE_CAPS,
                "acceptance_review": review,
                "account_snapshot": snapshot,
                "history_refs": evidence,
                "planned": planned,
                "outstanding": {},
                "settled": {
                    "cli-acceptance": {
                        "actual": acceptance_actual,
                        "receipt": review,
                        "settled_at": now,
                    }
                },
                "events": [{"kind": "initialize", "at": now}],
            }
            self._verify_invariants(state, require_snapshot=True)
            _atomic_write(self.path, state)
            return self._summary(state)

    def record_account_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Bind fresh, zero-assignment account evidence before every new allocation."""
        with self._locked() as state:
            self._verify_static_evidence(state)
            remaining = sum(x["units"] for x in state["planned"].values()) + sum(
                x["bound"]["units"] for x in state["outstanding"].values()
            )
            state["account_snapshot"] = self._snapshot(
                path, require_zero=True, remaining_units=remaining
            )
            state["events"].append(
                {
                    "kind": "account_snapshot",
                    "at": self.clock(),
                    "evidence": state["account_snapshot"],
                }
            )
            self._verify_invariants(state, require_snapshot=True)
            return self._summary(state)

    def reserve(
        self, job_id: str, worst_case: Mapping[str, Any], manifest_path: str | Path
    ) -> dict[str, Any]:
        """Atomically move exactly one registered job from planned to outstanding."""
        with self._locked() as state:
            self._verify_invariants(state, require_snapshot=True)
            if job_id not in state["planned"]:
                raise Refused("job is not an unstarted registered liability")
            worst = _bound(worst_case, "worst_case")
            if worst != state["planned"][job_id]:
                raise Refused("reservation must equal the immutable registered liability")
            if state["outstanding"]:
                raise Refused("one active S1 job maximum; reconcile the existing reservation first")
            self._manifest_fits(Path(manifest_path), job_id, worst)
            state["outstanding"][job_id] = {
                "bound": state["planned"].pop(job_id),
                "manifest": self._evidence(manifest_path),
                "reserved_at": self.clock(),
            }
            self._verify_invariants(state, require_snapshot=True)
            state["events"].append({"kind": "reserve", "job_id": job_id, "at": self.clock()})
            return self._summary(state)

    def settle(self, job_id: str, receipt_path: str | Path) -> dict[str, Any]:
        """Settle a CLI job only when a terminal supervisor receipt proves disappearance."""
        with self._locked() as state:
            self._verify_static_evidence(state)
            entry = state["outstanding"].get(job_id)
            if entry is None:
                raise Refused("no outstanding reservation for job")
            receipt = self._receipt(receipt_path, entry["manifest"])
            actual = self._cli_actual(receipt, entry)
            settled = {
                "actual": actual,
                "receipt": self._evidence(receipt_path),
                "settled_at": self.clock(),
            }
            if "retry_of" in entry:
                settled["retry_of"] = entry["retry_of"]
                settled["source_manifest"] = entry["manifest"]
            state["settled"][job_id] = settled
            del state["outstanding"][job_id]
            self._verify_invariants(state, require_snapshot=False)
            state["events"].append({"kind": "settle", "job_id": job_id, "at": self.clock()})
            return self._summary(state)

    def reconcile_operational_failure(
        self, job_id: str, receipt_path: str | Path, review_path: str | Path
    ) -> dict[str, Any]:
        """Close a failed initial attempt while preserving its charged evidence.

        This deliberately does *not* create or reserve a retry.  An operator must
        separately prepare a bound retry manifest and explicitly reserve it.
        """
        with self._locked() as state:
            self._verify_static_evidence(state)
            entry = state["outstanding"].get(job_id)
            if entry is None:
                raise Refused("no outstanding reservation for job")
            receipt = self._operational_receipt(receipt_path, entry["manifest"])
            review = self._operational_review(review_path, job_id, receipt_path, entry["manifest"])
            actual = self._operational_actual(receipt, entry)
            state["settled"][job_id] = {
                "actual": actual,
                "receipt": self._evidence(receipt_path),
                "source_manifest": entry["manifest"],
                "operational_failure_review": self._evidence(review_path),
                "failure_classification": review["failure_classification"],
                "original_bound": entry["bound"],
                "settled_at": self.clock(),
            }
            del state["outstanding"][job_id]
            self._verify_invariants(state, require_snapshot=False)
            state["events"].append(
                {"kind": "operational_failure", "job_id": job_id, "at": self.clock()}
            )
            return self._summary(state)

    def reserve_manual_retry(
        self,
        source_job_id: str,
        retry_job_id: str,
        manifest_path: str | Path,
        *,
        account_snapshot_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Atomically admit a conditional rerun, or record the arm INCOMPLETE."""
        refusal = None
        with self._locked() as state:
            self._verify_static_evidence(state)
            source = state["settled"].get(source_job_id)
            if not source or "operational_failure_review" not in source:
                raise Refused("manual retry requires a reconciled operational failure")
            source_evidence = source.get("source_manifest")
            if not isinstance(source_evidence, Mapping):
                raise Refused("original reservation manifest evidence is absent")
            if sha256(Path(source_evidence["path"])) != source_evidence.get("sha256"):
                raise Refused("original reservation manifest changed")
            source_manifest = json.loads(Path(source_evidence["path"]).read_text())
            arm = source_manifest.get("env", {}).get("S1_ARM")
            if arm not in {"n1", "n3"}:
                raise Refused("manual retry source is not an S1 training arm")
            expected = f"retry-{arm}"
            if (
                retry_job_id != expected
                or retry_job_id in state["settled"]
                or retry_job_id in state["outstanding"]
                or arm in state.get("incomplete_arms", {})
            ):
                raise Refused("the single registered retry liability for this arm is unavailable")
            if state["outstanding"]:
                raise Refused("one active S1 job maximum; reconcile the existing reservation first")
            retry_path = Path(manifest_path)
            self._retry_manifest_fits(source_manifest, retry_path, retry_job_id)
            state["outstanding"][retry_job_id] = {
                "bound": _bound(source.get("original_bound", {}), "original_bound"),
                "manifest": self._evidence(retry_path),
                "reserved_at": self.clock(),
                "retry_of": source_job_id,
            }
            try:
                if account_snapshot_path is not None:
                    state["account_snapshot"] = self._snapshot(
                        account_snapshot_path, require_zero=True
                    )
                self._verify_invariants(state, require_snapshot=True)
            except Refused as exc:
                del state["outstanding"][retry_job_id]
                refusal = f"{arm.upper()} INCOMPLETE: conditional rerun reservation refused: {exc}"
                state.setdefault("incomplete_arms", {})[arm] = {
                    "status": "INCOMPLETE",
                    "source_job_id": source_job_id,
                    "reason": str(exc),
                    "at": self.clock(),
                }
            state["events"].append(
                {
                    "kind": "refuse_manual_retry" if refusal else "reserve_manual_retry",
                    "job_id": retry_job_id,
                    "source_job_id": source_job_id,
                    "at": self.clock(),
                }
            )
            summary = self._summary(state)
        # Persist the refusal under the same lock before reporting it to the caller.
        if refusal:
            raise Refused(refusal)
        return summary

    def settle_mac(self, job_id: str, receipt_path: str | Path) -> dict[str, Any]:
        """Settle a Mac-only evaluation receipt; units/GPU must remain exactly zero."""
        with self._locked() as state:
            self._verify_static_evidence(state)
            entry = state["outstanding"].get(job_id)
            if entry is None:
                raise Refused("no outstanding reservation for job")
            data = json.loads(Path(receipt_path).read_text())
            if data.get("status") != "PASS" or data.get("job_id") != job_id:
                raise Refused("Mac receipt is not a passing receipt for this job")
            actual = _bound(data.get("actual", {}), "mac receipt actual")
            if actual["units"] or actual["gpu_allocation_hours"]:
                raise Refused("Mac receipt cannot charge units or GPU allocation")
            if any(actual[key] > entry["bound"][key] for key in DIMENSIONS):
                raise Refused("Mac actual exceeds its reservation")
            state["settled"][job_id] = {
                "actual": actual,
                "receipt": self._evidence(receipt_path),
                "settled_at": self.clock(),
            }
            del state["outstanding"][job_id]
            self._verify_invariants(state, require_snapshot=False)
            state["events"].append({"kind": "settle_mac", "job_id": job_id, "at": self.clock()})
            return self._summary(state)

    def status(self) -> dict[str, Any]:
        with self._locked() as state:
            self._verify_static_evidence(state)
            self._verify_invariants(state, require_snapshot=False)
            return self._summary(state)

    def _review_acceptance(
        self, path: str | Path
    ) -> tuple[dict[str, str], dict[str, float], float]:
        data = json.loads(Path(path).read_text())
        if data.get("schema") != REVIEW_SCHEMA or data.get("status") != "PASS":
            raise Refused("a passing independent acceptance-review wrapper is required")
        required = {
            "supervisor_sha256",
            "acceptance_result",
            "receipts",
            "fault_receipt",
            "transport_sources",
        }
        if not required <= set(data):
            raise Refused("acceptance review lacks required binding fields")
        if not isinstance(data["receipts"], list) or not data["receipts"]:
            raise Refused("acceptance review must bind reviewed receipts")
        if data.get("watchdog_disappearance_seconds", math.inf) > 300:
            raise Refused(
                "acceptance review does not prove watchdog disappearance within 300 seconds"
            )
        study_started_at = _number(data.get("study_started_at"), "study_started_at")
        if study_started_at > self.clock():
            raise Refused("study_started_at cannot be in the future")
        root = Path(__file__).resolve().parents[2]
        if data["supervisor_sha256"] != sha256(root / "research/colab_cli/supervisor.py"):
            raise Refused("acceptance review is not bound to the current supervisor source")
        required_transport = (
            "supervisor.py",
            "runtime.py",
            "accounting.py",
            "release.py",
            "transfers.py",
            "cli_guard.py",
            "bootstrap.sh",
            "launch.sh",
        )
        sources = data["transport_sources"]
        if not isinstance(sources, list):
            raise Refused("acceptance review lacks transport source manifest")
        seen = set()
        for item in sources:
            self._verify_bound_file(item, "transport source")
            target = Path(item["path"]).resolve()
            try:
                relative = str(target.relative_to(root / "research/colab_cli"))
            except ValueError as exc:
                raise Refused("transport source lies outside research/colab_cli") from exc
            seen.add(relative)
        if not set(required_transport) <= seen:
            raise Refused("acceptance review does not bind every transport source")
        result = data["acceptance_result"]
        self._verify_bound_file(result, "acceptance result")
        result_data = json.loads(Path(result["path"]).read_text())
        execution = result_data.get("final_plan", {}).get("execution")
        if (
            result_data.get("qualified") is not True
            or not isinstance(execution, list)
            or {row.get("job"): row.get("status") for row in execution}.get("cpu") != "PASS"
            or {row.get("job"): row.get("status") for row in execution}.get("l4") != "PASS"
        ):
            raise Refused("acceptance result is not qualified")
        for item in data["receipts"]:
            self._verify_bound_file(item, "acceptance receipt")
            receipt_data = json.loads(Path(item["path"]).read_text())
            if (
                receipt_data.get("status") != "PASS"
                or receipt_data.get("stop_verified") is not True
            ):
                raise Refused("acceptance receipt is not a verified clean release")
        fault = data["fault_receipt"]
        if not isinstance(fault, Mapping):
            raise Refused("acceptance review lacks watchdog fault evidence")
        fault_receipt, observer = fault.get("receipt"), fault.get("observer")
        self._verify_bound_file(fault_receipt, "watchdog fault receipt")
        self._verify_bound_file(observer, "watchdog observer")
        fault_data = json.loads(Path(fault_receipt["path"]).read_text())
        observer_data = json.loads(Path(observer["path"]).read_text())
        watchdog = observer_data.get("gates", {}).get("mac-loss-watchdog-unassign", {})
        release_bound = _number(watchdog.get("release_bound_seconds"), "watchdog release bound")
        observed = _number(
            watchdog.get("observed_after_lease_seconds"), "watchdog observed interval"
        )
        if (
            fault_data.get("status") != "INCOMPLETE"
            or fault_data.get("cleanup_error") is not None
            or fault_data.get("stop_verified") is not True
            or not isinstance(fault_data.get("endpoint"), str)
            or watchdog.get("endpoint") != fault_data["endpoint"]
            or watchdog.get("assignment_disappeared_within_bound") is not True
            or watchdog.get("renewal_suppressed") is not True
            or watchdog.get("checkpoint_exported_during_grace") is not True
            or release_bound > 300
            or observed > release_bound
        ):
            raise Refused("watchdog fault evidence does not prove bounded exact-endpoint release")
        for item in [result, *data["receipts"]]:
            self._verify_bound_file(item, "acceptance evidence")
        charge = _bound(data.get("acceptance_charge", {}), "acceptance_charge")
        if charge["units"] > 4 or charge["gpu_allocation_hours"] > 4:
            raise Refused("acceptance charge exceeds its registered <=4-unit bound")
        return self._evidence(path), charge, study_started_at

    @staticmethod
    def _verify_bound_file(item: Any, label: str) -> None:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise Refused(f"{label} binding is malformed")
        if sha256(Path(item["path"])) != item["sha256"]:
            raise Refused(f"{label} hash mismatch")

    def _snapshot(
        self, path: str | Path, *, require_zero: bool, remaining_units: float = 0.0
    ) -> dict[str, Any]:
        data = json.loads(Path(path).read_text())
        captured = _number(data.get("captured_at"), "account snapshot captured_at")
        age = self.clock() - captured
        if age < -60 or age > SNAPSHOT_MAX_AGE_SECONDS:
            raise Refused("account snapshot is stale")
        assignments = data.get("active_assignments", data.get("assignments"))
        assignments = _number(assignments, "account snapshot active_assignments")
        if assignments != int(assignments):
            raise Refused("account assignments must be an exact integer")
        if data.get("assignments_known") is not True:
            raise Refused("unknown account assignment state blocks admission")
        balance = _number(
            data.get("balance_units", data.get("balance")), "account snapshot balance"
        )
        rate = _number(
            data.get("usage_rate_units_per_hour", data.get("rate")), "account snapshot rate"
        )
        if require_zero and assignments != 0:
            raise Refused("S1 requires zero active account assignments before a new launch")
        if require_zero and rate != 0:
            raise Refused("zero assignments with nonzero provider rate is unknown debit")
        if balance < remaining_units:
            raise Refused("account balance cannot cover all remaining S1 unit liabilities")
        return self._evidence(path)

    def _verify_static_evidence(self, state: Mapping[str, Any]) -> None:
        # Re-open the wrapper: its nested reviewed receipts and current supervisor
        # identity matter just as much as the wrapper's own bytes.
        review, _, study_started_at = self._review_acceptance(state["acceptance_review"]["path"])
        if review != state["acceptance_review"]:
            raise Refused("acceptance review evidence identity changed")
        if study_started_at != float(state["created_at"]):
            raise Refused("reviewed study start differs from frozen ledger start")
        for group in ("acceptance_review",):
            item = state[group]
            if sha256(Path(item["path"])) != item["sha256"]:
                raise Refused(f"immutable {group} evidence changed")
        for name, item in state["history_refs"].items():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise Refused(f"historical reference changed: {name}")
        snapshot = state["account_snapshot"]
        if sha256(Path(snapshot["path"])) != snapshot["sha256"]:
            raise Refused("account snapshot evidence changed")
        for job_id, item in state["settled"].items():
            if sha256(Path(item["receipt"]["path"])) != item["receipt"]["sha256"]:
                raise Refused(f"settled receipt evidence changed: {job_id}")
            for name in ("source_manifest", "operational_failure_review"):
                if name in item and sha256(Path(item[name]["path"])) != item[name]["sha256"]:
                    raise Refused(f"settled {name} evidence changed: {job_id}")

    def _receipt(self, path: str | Path, manifest: Mapping[str, str]) -> dict[str, Any]:
        receipt_path = Path(path).resolve(strict=True)
        data = json.loads(receipt_path.read_text())
        if data.get("status") != "PASS" or data.get("stop_verified") is not True:
            raise Refused("supervisor receipt does not prove completed stop/disappearance")
        if data.get("error") is not None or data.get("cleanup_error") is not None:
            raise Refused("supervisor receipt records an execution or cleanup error")
        self._receipt_manifest_bound(receipt_path, data, manifest)
        if not data.get("endpoint") or not data.get("terminal"):
            raise Refused("receipt lacks endpoint or terminal evidence")
        return data

    def _operational_receipt(self, path: str | Path, manifest: Mapping[str, str]) -> dict[str, Any]:
        """Validate a supervisor-stop receipt for a non-scientific lost attempt."""
        receipt_path = Path(path).resolve(strict=True)
        data = json.loads(receipt_path.read_text())
        if data.get("status") != "INCOMPLETE" or data.get("stop_verified") is not True:
            raise Refused("operational failure lacks verified stop/disappearance")
        if data.get("cleanup_error") is not None:
            raise Refused("operational failure cleanup is unverified")
        if not data.get("endpoint"):
            raise Refused("operational failure receipt lacks endpoint evidence")
        self._receipt_manifest_bound(receipt_path, data, manifest)
        # A present, fully verified final is decisive; a partial export index is not.
        final = data.get("verified_final_export")
        if final is True:
            raise Refused("verified final export exists; a retry cannot be prepared")
        self._operational_actual(data, {"bound": {key: math.inf for key in DIMENSIONS}})
        return data

    @staticmethod
    def _receipt_manifest_bound(
        receipt_path: Path, receipt: Mapping[str, Any], manifest: Mapping[str, str]
    ) -> None:
        if receipt.get("manifest_sha256") == manifest["sha256"]:
            return
        output_manifest = receipt_path.parent / "manifest.json"
        named_hash = receipt.get("manifest_sha256")
        if not output_manifest.is_file() or sha256(output_manifest) != named_hash:
            raise Refused("receipt manifest hash does not name its output manifest")
        original, validated = json.loads(Path(manifest["path"]).read_text()), json.loads(
            output_manifest.read_text()
        )
        # The supervisor may only normalize representation/default types. Every
        # execution input remains semantic identity for both PASS and INCOMPLETE.
        for key in (
            "job_id",
            "commit",
            "script",
            "env",
            "requirements",
            "timeout_minutes",
            "output_dir",
            "inputs",
            "budget",
            "lease_seconds",
            "max_export_bytes",
        ):
            if original.get(key) != validated.get(key):
                raise Refused("validated receipt manifest differs from reserved manifest")

    def _operational_review(
        self, path: str | Path, job_id: str, receipt_path: str | Path, manifest: Mapping[str, str]
    ) -> dict[str, Any]:
        review_path = Path(path).resolve(strict=True)
        value = json.loads(review_path.read_text())
        if (
            value.get("schema") != "s1-operational-failure-review/v1"
            or value.get("status") != "PASS"
        ):
            raise Refused("operational failure review is absent or not PASS")
        if value.get("job_id") != job_id or value.get("manifest_sha256") != manifest["sha256"]:
            raise Refused("operational failure review identity differs from the reservation")
        if value.get("receipt_sha256") != sha256(Path(receipt_path)):
            raise Refused("operational failure review is not bound to the receipt")
        if value.get("failure_classification") not in OPERATIONAL_FAILURES:
            raise Refused("failure is not an eligible operational classification")
        if (
            value.get("no_efficacy_look") is not True
            or value.get("mac_final_evaluation_planned") is not True
        ):
            raise Refused("operational failure review does not preserve the one-look rule")
        if value.get("verified_final_export") is not False:
            raise Refused("operational failure review does not exclude a verified final")
        prohibited = (
            "tripwire",
            "integrity_failure",
            "parity_failure",
            "hash_failure",
            "counter_failure",
        )
        if any(value.get(name) is not False for name in prohibited):
            raise Refused("scientific or integrity failure cannot consume the retry")
        return value

    def _cli_actual(self, receipt: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, float]:
        bound = entry["bound"]
        before, after = receipt.get("usage_before"), receipt.get("usage_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise Refused("receipt lacks final account usage; reservation remains outstanding")
        debit = _number(before.get("balance"), "receipt usage_before.balance") - _number(
            after.get("balance"), "receipt usage_after.balance"
        )
        if debit < 0:
            raise Refused("account balance increased; receipt cannot price this allocation")
        elapsed = max(
            _number(receipt.get("elapsed_seconds"), "receipt elapsed_seconds") / 3600,
            (self.clock() - float(entry["reserved_at"])) / 3600,
        )
        actual = {
            "units": debit,
            "gpu_allocation_hours": elapsed,
            "mac_worker_hours": 0.0,
            "elapsed_hours": elapsed,
        }
        if any(actual[key] > bound[key] + 1e-9 for key in DIMENSIONS):
            raise Refused("receipt exceeds reserved worst case")
        # Provider balance can lag or include unrelated account work.  Do not release a
        # request-to-disappearance reservation based on an apparently small debit.
        actual["units"] = max(actual["units"], bound["units"])
        return actual

    def _operational_actual(
        self, receipt: Mapping[str, Any], entry: Mapping[str, Any]
    ) -> dict[str, float]:
        """Price only the receipt's known request-to-disappearance interval.

        Human review time after disappearance remains real elapsed wall time via
        ``_totals``; it is not retrospectively claimed as paid GPU time.
        """
        bound = entry["bound"]
        before, after = receipt.get("usage_before"), receipt.get("usage_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise Refused("receipt lacks final account usage; reservation remains outstanding")
        debit = _number(before.get("balance"), "receipt usage_before.balance") - _number(
            after.get("balance"), "receipt usage_after.balance"
        )
        if debit < 0:
            raise Refused("account balance increased; receipt cannot price this allocation")
        elapsed = _number(receipt.get("elapsed_seconds"), "receipt elapsed_seconds") / 3600
        actual = {
            "units": max(debit, bound["units"]),
            "gpu_allocation_hours": elapsed,
            "mac_worker_hours": 0.0,
            "elapsed_hours": elapsed,
        }
        if any(actual[key] > bound[key] + 1e-9 for key in DIMENSIONS):
            raise Refused("receipt exceeds reserved worst case")
        return actual

    @staticmethod
    def _manifest_fits(path: Path, job_id: str, bound: Mapping[str, float]) -> None:
        manifest = json.loads(path.read_text())
        if manifest.get("job_id") != job_id:
            raise Refused("manifest job id differs from reservation")
        if manifest.get("kind") == "mac":
            if bound["units"] or bound["gpu_allocation_hours"]:
                raise Refused("Mac manifest must reserve zero units and GPU allocation")
            timeout = _number(manifest.get("timeout_minutes"), "Mac manifest timeout") / 60
            workers = _number(manifest.get("workers"), "Mac manifest workers")
            if workers > 2 or timeout > bound["elapsed_hours"] + 1e-9:
                raise Refused("Mac manifest worker/timeout exceeds registered liability")
            return
        budget = manifest.get("budget", {})
        rate = _number(budget.get("rate_units_per_hour"), "manifest budget rate")
        timeout = _number(manifest.get("timeout_minutes"), "manifest timeout") / 60
        if timeout * rate > bound["units"] + 1e-9 or timeout > bound["gpu_allocation_hours"] + 1e-9:
            raise Refused("manifest timeout/rate needs more than its registered reservation")

    @staticmethod
    def _retry_manifest_fits(source: Mapping[str, Any], path: Path, job_id: str) -> None:
        retry = json.loads(path.read_text())
        if retry.get("job_id") != job_id:
            raise Refused("retry manifest job id differs from its retry liability")
        if not isinstance(retry.get("output_dir"), str) or retry["output_dir"] == source.get(
            "output_dir"
        ):
            raise Refused("retry manifest requires a fresh output directory")
        attempt = retry.get("attempt")
        if not isinstance(attempt, Mapping) or attempt != {
            "kind": "manual-operational-rerun",
            "of": source.get("job_id"),
        }:
            raise Refused("retry manifest lacks the required explicit attempt metadata")
        compare_source, compare_retry = dict(source), dict(retry)
        for value in (compare_source, compare_retry):
            value.pop("job_id", None)
            value.pop("output_dir", None)
            value.pop("attempt", None)
        if compare_source != compare_retry:
            raise Refused(
                "retry manifest differs from the original seed, recipe, inputs, commit or budget"
            )

    def _verify_invariants(self, state: Mapping[str, Any], *, require_snapshot: bool) -> None:
        self._verify_static_evidence(state)
        if require_snapshot:
            snapshot = state["account_snapshot"]
            # Re-parse the captured bytes; the hash above catches any substitution.
            remaining = sum(x["units"] for x in state["planned"].values()) + sum(
                x["bound"]["units"] for x in state["outstanding"].values()
            )
            self._snapshot(snapshot["path"], require_zero=True, remaining_units=remaining)
        if len(state["outstanding"]) > 1:
            raise Refused("ledger records more than one active S1 job")
        totals = self._totals(state)["projected_total"]
        for key, cap in USABLE_CAPS.items():
            if totals[key] > cap + 1e-9:
                raise Refused(
                    f"complete upper bound exceeds usable {key} cap: {totals[key]:.6g} > {cap:.6g}"
                )

    def _summary(self, state: Mapping[str, Any]) -> dict[str, Any]:
        summary = {
            "usable_caps": dict(USABLE_CAPS),
            "nominal_caps": dict(NOMINAL_CAPS),
            "planned_jobs": sorted(state["planned"]),
            "incomplete_arms": state.get("incomplete_arms", {}),
            "outstanding_jobs": sorted(state["outstanding"]),
            "settled_jobs": sorted(state["settled"]),
            "study_started_at": state["created_at"],
            "ledger_initialized_at": state["ledger_initialized_at"],
        }
        summary.update(self._totals(state))
        return summary

    def _totals(self, state: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        """Expose current spend, reservations, future liability, and full projection."""
        result = {
            name: {key: 0.0 for key in DIMENSIONS} for name in ("spent", "outstanding", "remaining")
        }
        for group, field, destination in (
            ("settled", "actual", "spent"),
            ("outstanding", "bound", "outstanding"),
            ("planned", None, "remaining"),
        ):
            for item in state[group].values():
                values = item if field is None else item[field]
                for key in DIMENSIONS:
                    result[destination][key] += _number(values[key], f"{group}.{key}")
        projected = {key: sum(result[group][key] for group in result) for key in DIMENSIONS}
        # Real time since initialization is irrevocably consumed, including gaps between jobs.
        wall_elapsed = max(0.0, (self.clock() - float(state["created_at"])) / 3600)
        projected["elapsed_hours"] = max(
            projected["elapsed_hours"],
            wall_elapsed
            + result["outstanding"]["elapsed_hours"]
            + result["remaining"]["elapsed_hours"],
        )
        result["projected_total"] = projected
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("command", choices=("status",))
    args = parser.parse_args()
    print(json.dumps(Ledger(args.ledger).status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
