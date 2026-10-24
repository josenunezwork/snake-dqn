"""Live acceptance observer for the bounded Colab CLI qualification.

It runs only through ``acceptance --execute-after-review``.  The observer uses
an already allocated broker session and never calls ``new`` or retries an
allocation.  Each operation records an explicit evidence result; a failure
raises so the broker retains its normal cleanup and writes an INCOMPLETE receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from research.colab_cli import policy, supervisor
from research.colab_cli.transfers import (
    CONTENTS_ATTEMPTS,
    _contents_call,
    upload_contents,
    upload_file,
)

WATCHDOG_DISAPPEARANCE_BOUND_SECONDS = 300
WATCHDOG_DISAPPEARANCE_POLL_SECONDS = 10
WATCHDOG_CHECKPOINT_GRACE_SECONDS = 15
# A read dispatched before publication must not occupy the entire short
# checkpoint window. These tiny files need fresh reads during grace even when
# a preceding missing-file response stalls at the provider.
WATCHDOG_CHECKPOINT_READ_SECONDS = 3


class DirectObserverDownloadEvidence:
    """Correlate every direct observer download in one bounded operation.

    The observer's polling reads predate the transfer helper because they are
    deliberately tiny, direct CLI reads.  They still consume provider work and
    must therefore retain a prospective identity even when a poll fails.  This
    class only annotates evidence already emitted by ``CLI.call``; it neither
    performs a transfer nor changes the watchdog's retry, deadline, or release
    decisions.
    """

    def __init__(
        self,
        cli: supervisor.CLI,
        remotes: set[str],
        *,
        verification_kind: str = "observed-checkpoint-sidecar",
    ) -> None:
        self.cli = cli
        self.remotes = frozenset(remotes)
        self.verification_kind = verification_kind
        # This derives only an evidence namespace.  It must not consume the
        # UUID stream used by live request creation or otherwise perturb the
        # observed transport sequence.
        namespace = "\0".join(
            (str(getattr(cli, "session", "")), str(len(cli.call_records)), *sorted(self.remotes))
        )
        self.operation_id = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:32]
        self._seen: set[str] = set()
        self._parents: dict[str, dict[str, Any]] = {}

    def observe_since(self, first_call: int) -> None:
        """Give each newly observed direct download its stable raw ordinal."""
        for row in self.cli.call_records[first_call:]:
            remote = row.get("remote")
            if row.get("command") != "download" or remote not in self.remotes:
                continue
            row_key = row.get("attempt_id") or row.get("operation_id")
            if not isinstance(row_key, str) or not row_key or row_key in self._seen:
                continue
            self._seen.add(row_key)
            parent = self._parent_for(remote)
            child = parent["children"][0]
            ordinal = child["attempts"] + 1
            child["attempts"] = ordinal
            child["status"] = row.get("status", "failed")
            child["finished_monotonic"] = row.get("finished_monotonic")
            if child.get("started_monotonic") is None:
                child["started_monotonic"] = row.get("started_monotonic")
                parent["started_monotonic"] = row.get("started_monotonic")
            parent["finished_monotonic"] = row.get("finished_monotonic")
            parent["elapsed_seconds"] = self._elapsed(parent)
            row.update(
                parent_transfer_id=parent["transfer_id"],
                transfer_id=child["transfer_id"],
                transfer_attempt=ordinal,
                transfer_role="consumed",
            )
            # A pure declared-absence sequence has no transfer record to
            # manufacture a C6 failure. Once a real completion occurs, retain
            # the complete preceding lineage for its later verification.
            if row.get("status") == "completed" and parent not in self.cli.transfer_records:
                self.cli.transfer_records.append(parent)

    def bind_verified(self, remote: str, digest: str) -> bool:
        """Bind only the completed direct read to bytes verified in this operation."""
        if remote not in self.remotes or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
        parent = self._parents.get(remote)
        if parent is None:
            return False
        child = parent["children"][0]
        matched = [
            row
            for row in self.cli.call_records
            if row.get("command") == "download"
            and row.get("remote") == remote
            and row.get("parent_transfer_id") == parent["transfer_id"]
            and row.get("transfer_id") == child["transfer_id"]
            and row.get("status") == "completed"
        ]
        if not matched:
            return False
        # Only the final read supplied the sidecar/payload pair just checked.
        # Earlier successful reads retain their raw identity without a future digest.
        row = matched[-1]
        row["verified_sha256"] = digest
        child.update(status="completed", verified_sha256=digest)
        parent.update(
            status="verified",
            sha256=digest,
            verified_sha256=digest,
            verification_kind=self.verification_kind,
            finished_monotonic=row.get("finished_monotonic"),
            elapsed_seconds=self._elapsed(parent, row),
        )
        if parent not in self.cli.transfer_records:
            self.cli.transfer_records.append(parent)
        return True

    def _parent_for(self, remote: str) -> dict[str, Any]:
        parent = self._parents.get(remote)
        if parent is not None:
            return parent
        suffix = hashlib.sha256(remote.encode("utf-8")).hexdigest()[:16]
        parent_id = f"observer-download-{self.operation_id}-{suffix}"
        child_id = f"observer-contents-{self.operation_id}-{suffix}"
        child = {
            "transfer_id": child_id,
            "parent_transfer_id": parent_id,
            "remote": remote,
            "direction": "download",
            "attempts": 0,
            "status": "unverified",
            "started_monotonic": None,
            "finished_monotonic": None,
        }
        parent = {
            "transfer_id": parent_id,
            "direction": "download",
            "remote": remote,
            "status": "unverified",
            "verification_kind": self.verification_kind,
            "children": [child],
            "started_monotonic": None,
            "finished_monotonic": None,
        }
        self._parents[remote] = parent
        return parent

    @staticmethod
    def _elapsed(parent: dict[str, Any], final_row: dict[str, Any] | None = None) -> float | None:
        started = parent.get("started_monotonic")
        finished = (final_row or parent).get("finished_monotonic")
        if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
            return finished - started
        return None


# These snippets intentionally run in the installed CLI interpreter.  The Mac
# environment does not import ``colab_cli.state``: that would turn a missing
# local CLI package into an acceptance-path dependency.  The marker literals
# also make the offline fake OAuth interpreter emulate this exact bounded
# StateStore mutation without importing the real client.
_CLEAR_KERNEL_METADATA = """\
# SNAKE_ACCEPTANCE_KERNEL_METADATA_CLEAR
import hashlib,json,sys
from colab_cli.state import StateStore
store=StateStore(sys.argv[1])
state=store.get(sys.argv[2])
if state is None:
    raise RuntimeError('tracked acceptance session is absent')
fields=('kernel_id','endpoint','url','token')
if any(not isinstance(getattr(state,name,None),str) or not getattr(state,name) for name in fields):
    raise RuntimeError('tracked acceptance session lacks acknowledged kernel metadata')
before={name:getattr(state,name) for name in fields}
state.kernel_id=None
store.add(state)
after=store.get(sys.argv[2])
if after is None or after.kernel_id is not None:
    raise RuntimeError('kernel ID did not clear locally')
if any(getattr(after,name,None)!=before[name] for name in ('endpoint','url','token')):
    raise RuntimeError('clearing kernel ID changed tracked metadata')
print(json.dumps({'kernel_id':before['kernel_id'],'endpoint':before['endpoint'],'url':before['url'],'token_sha256':hashlib.sha256(before['token'].encode()).hexdigest()},sort_keys=True))
"""

_SNAPSHOT_KERNEL_METADATA = """\
# SNAKE_ACCEPTANCE_KERNEL_METADATA_SNAPSHOT
import hashlib,json,sys
from colab_cli.state import StateStore
state=StateStore(sys.argv[1]).get(sys.argv[2])
if state is None:
    raise RuntimeError('tracked acceptance session is absent')
fields=('kernel_id','endpoint','url','token')
if any(not isinstance(getattr(state,name,None),str) or not getattr(state,name) for name in fields):
    raise RuntimeError('tracked acceptance session lacks recovered kernel metadata')
print(json.dumps({'kernel_id':state.kernel_id,'endpoint':state.endpoint,'url':state.url,'token_sha256':hashlib.sha256(state.token.encode()).hexdigest()},sort_keys=True))
"""


class AcceptanceObserver:
    """Exercise account/kernel fault boundaries without a second allocation."""

    def __init__(
        self,
        oauth_python: str,
        profile: str = "final-only",
        injections: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        # Pre-dispatch v2 injection manifests by job ID (identity of deliberate faults).
        self.injections = injections or {}
        if profile not in {"final-only", "full-state"}:
            raise ValueError(f"unknown acceptance profile: {profile}")
        self.profile = profile
        self.oauth_python = oauth_python

    def _record(self, gate: str, **value: Any) -> None:
        if gate in self.records:
            raise supervisor.Refused(f"acceptance gate observed twice: {gate}")
        self.records[gate] = value

    def __call__(self, stage: str, context: dict[str, Any]) -> None:
        cli: supervisor.CLI = context["cli"]
        manifest: dict[str, Any] = context["manifest"]
        output: Path = context["output"]
        if stage == "runtime_started":
            if "fresh-oauth-refresh" not in self.records:
                self._refresh_oauth_then_query(cli)
            self._accounted_assignments(cli, manifest["requirements"]["gpu"])
            return
        if stage == "runtime_ready":
            self._recover_cleared_kernel_metadata(cli, manifest["requirements"]["gpu"])
            return
        if stage == "first_upload":
            self._verify_first_upload(cli, manifest["requirements"]["gpu"])
            return
        if stage == "ready" and manifest["requirements"]["gpu"] == "L4":
            probe = self._wait_phase(context, "transfer_ready")
            if self.profile == "final-only":
                self._final_artifact_download(context, probe)
            else:
                self._download_snapshot(context, probe)
            self._watchdog(context)
            return
        if stage == "ready":
            self._url_no_second_allocation(cli)
            self._state(context)
            self._silence_then_bounded_rpc(cli, manifest.get("job_id"))
            self._corrupt_export_rejection(cli, context["root"], output)
            self._busy_transfer_and_client_loss(context)
            return
        if stage == "terminal":
            exports = context["exports"]
            self._verify_terminal_exports(exports, output)
            if self.profile == "full-state":
                self._resume_import(context)

    @staticmethod
    def _arm_name(arm: str) -> str:
        if arm not in {"cpu", "L4"}:
            raise supervisor.Refused(f"acceptance has unknown hardware arm: {arm}")
        return arm

    def _cold_kernel_arm(self, arm: str, *, create: bool) -> dict[str, Any]:
        """Return the single per-arm record for the combined cold-start gate."""
        gate = "cold-kernel-metadata-recovery"
        record = self.records.get(gate)
        if record is None:
            if not create:
                raise supervisor.Refused("acceptance first upload lacks metadata recovery evidence")
            record = {"status": "INCOMPLETE", "arms": []}
            self._record(gate, **record)
        arms = record.get("arms")
        if not isinstance(arms, list):
            raise supervisor.Refused("acceptance cold-kernel gate has malformed arm evidence")
        matches = [item for item in arms if isinstance(item, dict) and item.get("arm") == arm]
        if len(matches) > 1:
            raise supervisor.Refused("acceptance cold-kernel gate repeated an arm")
        if matches:
            return matches[0]
        if not create:
            raise supervisor.Refused(
                "acceptance first upload has no matching metadata recovery evidence"
            )
        evidence: dict[str, Any] = {"arm": arm}
        arms.append(evidence)
        return evidence

    def _refresh_cold_kernel_gate_status(self) -> None:
        record = self.records["cold-kernel-metadata-recovery"]
        arms = record["arms"]
        complete = (
            isinstance(arms, list)
            and {arm.get("arm") for arm in arms if isinstance(arm, dict)} == {"cpu", "L4"}
            and len(arms) == 2
            and all(
                isinstance(arm.get("metadata_recovery"), dict)
                and isinstance(arm.get("first_upload_transfer"), dict)
                and arm["first_upload_transfer"].get("status") == "completed"
                for arm in arms
            )
        )
        record["status"] = "PASS" if complete else "INCOMPLETE"

    def _kernel_metadata(self, cli: supervisor.CLI, code: str) -> dict[str, str]:
        """Run a StateStore-only operation in the bounded installed CLI interpreter."""
        try:
            result = subprocess.run(
                [self.oauth_python, "-c", code, str(cli.config), cli.session],
                check=False,
                capture_output=True,
                text=True,
                timeout=cli.remaining_seconds(150),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise supervisor.Refused("acceptance kernel metadata helper failed") from exc
        if result.returncode:
            raise supervisor.Refused("acceptance kernel metadata helper refused")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise supervisor.Refused(
                "acceptance kernel metadata helper returned invalid JSON"
            ) from exc
        required = {"kernel_id", "endpoint", "url", "token_sha256"}
        if (
            not isinstance(value, dict)
            or set(value) != required
            or any(not isinstance(value[key], str) or not value[key] for key in required)
        ):
            raise supervisor.Refused("acceptance kernel metadata helper returned malformed state")
        return {key: value[key] for key in required}

    def _recover_cleared_kernel_metadata(self, cli: supervisor.CLI, arm: str) -> None:
        """Prove the real guard can recover a deliberately lost session kernel ID.

        This is deliberately positioned after Contents readiness and before the
        first upload.  It mutates only the local StateStore's ``kernel_id`` and
        then uses the guard's GET-only recovery helper; it never allocates or
        submits an execution.
        """
        from research.colab_cli import cli_guard

        arm = self._arm_name(arm)
        before = self._kernel_metadata(cli, _CLEAR_KERNEL_METADATA)
        try:
            idle = cli_guard.kernel_idle(
                self.oauth_python,
                cli.config,
                cli.session,
                timeout=cli.remaining_seconds(150),
            )
        except cli_guard.GuardError as exc:
            raise supervisor.Refused("acceptance kernel metadata recovery was refused") from exc
        if not idle:
            raise supervisor.Refused("acceptance kernel metadata recovery did not prove idle")
        after = self._kernel_metadata(cli, _SNAPSHOT_KERNEL_METADATA)
        if after != before:
            raise supervisor.Refused("acceptance kernel metadata recovery changed tracked identity")
        evidence = self._cold_kernel_arm(arm, create=True)
        if "metadata_recovery" in evidence:
            raise supervisor.Refused("acceptance cold-kernel recovery repeated an arm")
        evidence["metadata_recovery"] = {
            "original_kernel_id": before["kernel_id"],
            "recovered_kernel_id": after["kernel_id"],
            "same_kernel_id": True,
            "tracked_metadata_unchanged": True,
            "idle_proven_by": "cli_guard.kernel_idle GET-only helper",
        }
        self._refresh_cold_kernel_gate_status()

    def _verify_first_upload(self, cli: supervisor.CLI, arm: str) -> None:
        """Require the just-completed credential upload in the supervisor ledger."""
        arm = self._arm_name(arm)
        ledger = cli.contents_transfers
        if (
            not isinstance(ledger, list)
            or not ledger
            or len(ledger) > CONTENTS_ATTEMPTS
            or any(not isinstance(row, dict) for row in ledger)
        ):
            raise supervisor.Refused("acceptance first upload has no contents transfer ledger")
        entry = ledger[-1]
        if (
            any(row.get("direction") != "upload" for row in ledger)
            or any(row.get("remote") != entry.get("remote") for row in ledger)
            or any(row.get("status") != "failed" for row in ledger[:-1])
            or entry.get("direction") != "upload"
            or entry.get("status") != "completed"
            or not isinstance(entry.get("remote"), str)
            or not entry["remote"]
            or isinstance(entry.get("size_bytes"), bool)
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] <= 0
            or isinstance(entry.get("elapsed_seconds"), bool)
            or not isinstance(entry.get("elapsed_seconds"), (int, float))
            or not math.isfinite(float(entry["elapsed_seconds"]))
            or entry["elapsed_seconds"] <= 0
        ):
            raise supervisor.Refused("acceptance first upload ledger is incomplete")
        evidence = self._cold_kernel_arm(arm, create=False)
        if "first_upload_transfer" in evidence:
            raise supervisor.Refused("acceptance first upload repeated an arm")
        evidence["first_upload_transfer"] = {
            "remote": entry["remote"],
            "attempts": len(ledger),
            "size_bytes": entry["size_bytes"],
            "elapsed_seconds": entry["elapsed_seconds"],
            "status": "completed",
        }
        self._refresh_cold_kernel_gate_status()

    def _refresh_oauth_then_query(self, cli: supervisor.CLI) -> None:
        """Refresh in memory without modifying or printing the credential file."""
        code = (
            "from google.oauth2.credentials import Credentials\n"
            "from google.auth.transport.requests import Request\n"
            "from colab_cli.auth import TOKEN_CONFIG_PATH,PUBLIC_SCOPES\n"
            "c=Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH,PUBLIC_SCOPES)\n"
            "assert c.refresh_token\n"
            "import requests\n"
            "original=requests.sessions.Session.request\n"
            "def bounded(self,*a,**kw):\n"
            "    kw['timeout']=(30,120)\n"
            "    return original(self,*a,**kw)\n"
            "requests.sessions.Session.request=bounded\n"
            "c.refresh(Request(session=__import__('requests').Session()))\n"
            "assert c.valid\n"
            "print('REFRESH_OK')\n"
        )
        deadline = time.monotonic() + cli.remaining_seconds(150)
        for attempt in range(policy.ATTEMPTS):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise supervisor.Refused("OAuth refresh operation budget exhausted")
                result = subprocess.run(
                    [self.oauth_python, "-c", code],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                )
                if result.returncode == 0 and result.stdout.strip() == "REFRESH_OK":
                    break
                failure = RuntimeError(result.stderr)
                if not policy.transient_error(failure):
                    raise supervisor.Refused("OAuth refresh did not confirm reusable credentials")
            except (OSError, subprocess.SubprocessError) as exc:
                failure = exc
                if not policy.transient_error(exc):
                    raise supervisor.Refused("OAuth refresh subprocess failed") from exc
            if attempt == policy.ATTEMPTS - 1:
                raise supervisor.Refused("OAuth refresh transient retry budget exhausted")
            delay = min(policy.BACKOFF_SECONDS[attempt], max(0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
        # Official CLI 0.7.4 exposes whoami even though its terse help omits it.
        identity = cli.call("whoami")
        if identity.strip():
            cli._parsed_read("whoami", identity)
        self._record("fresh-oauth-refresh", refreshed_in_memory=True, usage=cli.usage())

    def _accounted_assignments(self, cli: supervisor.CLI, arm: str) -> None:
        """Prove that concurrent account debit has an explicit owner and rate.

        The supervisor owns discovery and compares this evidence with the
        manifest's reviewed endpoint map.  Acceptance does not infer a price
        from a balance delta, because two notebook workers can otherwise make
        a 4-unit qualification stop spuriously or conceal its own spend.
        """
        if getattr(cli, "accounting", None) is None:
            raise supervisor.Refused("acceptance has no supervisor accounting tracker")
        ledger = cli.accounting.accounted_assignments()
        unaccounted = ledger.get("unaccounted_endpoints")
        assignments = ledger.get("assignments")
        if unaccounted or not isinstance(assignments, list):
            raise supervisor.Refused("acceptance has unaccounted concurrent assignments")
        for assignment in assignments:
            if not isinstance(assignment, dict) or assignment.get("rate_units_per_hour", 0) <= 0:
                raise supervisor.Refused("acceptance assignment ledger lacks a positive rate")
        record = self.records.get("account-assignment-spend")
        arm_evidence = {"arm": arm, **ledger}
        if record is None:
            self._record("account-assignment-spend", arms=[arm_evidence])
            return
        # CPU and L4 are deliberately sequential arms of one qualification.
        # Keep both immutable snapshots under the single required gate instead
        # of treating the L4 admission as a duplicate observation.
        arms = record.get("arms")
        if not isinstance(arms, list) or any(item.get("arm") == arm for item in arms):
            raise supervisor.Refused("acceptance assignment ledger repeated an arm")
        arms.append(arm_evidence)

    def _url_no_second_allocation(self, cli: supervisor.CLI) -> None:
        before = cli.sessions(timeout=policy.READ_SECONDS)
        url = cli.call("url", "-s", cli.session, timeout=policy.READ_SECONDS)
        after = cli.sessions(timeout=policy.READ_SECONDS)
        if before != after or list(after.values()).count(cli.session) != 1:
            raise supervisor.Refused("url changed assignment identity or count")
        if re.search(r"https?://\S+", url):
            cli._parsed_read("url", url)
        self._record("url-no-second-allocation", sessions_before=before, sessions_after=after)

    def _silence_then_bounded_rpc(self, cli: supervisor.CLI, job_id: str | None = None) -> None:
        from research.colab_cli.injection_manifest import SILENCE_CODE

        started = time.monotonic()
        planned = [
            entry
            for entry in self.injections.get(job_id or "", {}).get("entries", [])
            if entry.get("injection_id") == "silence-timeout"
        ]
        # The runtime child is an independently supervised subprocess. This
        # bounded kernel call creates an actual >60-second execution silence;
        # its explicit timeout is evidence that no unbounded wait is hidden.
        # Its request ID is the pre-dispatch v2 injection identity when planned.
        try:
            value = cli.rpc(
                SILENCE_CODE,
                timeout=120,
                request_id=planned[0]["request_id"] if planned else None,
            )
        finally:
            for row in cli.call_records:
                if (
                    planned
                    and row.get("request_id") == planned[0]["request_id"]
                    and row.get("failure_class") == "CONTROL_RPC_TIMEOUT"
                ):
                    row["failure_type"] = "CONTROL_RPC_TIMEOUT"
        self._record(
            "silence-timeout",
            silence_seconds=time.monotonic() - started,
            rpc_timeout_seconds=policy.RPC_SECONDS,
            rpc_value=value,
        )

    @staticmethod
    def _state(context: dict[str, Any]) -> dict[str, Any]:
        cli, root, generation = context["cli"], context["root"], context["generation"]
        result = cli.poll_rpc(
            "import importlib.util,json\nfrom pathlib import Path\n"
            f"s=importlib.util.spec_from_file_location('accept_runtime',{root + '/runtime.py'!r})\n"
            "module=importlib.util.module_from_spec(s)\ns.loader.exec_module(module)\n"
            f"state=module.read_state({root!r},{generation!r},renew=True)\n"
            f"p=Path({root!r})/'acceptance-state.json'\n"
            "result={'runtime':state,'probe':json.loads(p.read_text()) if p.exists() else {}}"
        )
        try:
            # Account usage, the provider session list, and broker lane
            # records form one accounting snapshot.  Holding the broker lock
            # prevents a concurrently admitted lane from being half-visible.
            with supervisor.account_locked(cli):
                usage = cli.usage()
                sessions = cli.sessions()
                active_records = [
                    json.loads(path.read_text())
                    for path in supervisor.ACCOUNT_ROOT.glob("active-*.json")
                    if path.name != f"active-{cli.session}.json"
                ]
                cli.accounting.check(
                    usage,
                    sessions=sessions,
                    active_records=active_records,
                    budget_other_assignments=context["manifest"]["budget"].get("other_assignments"),
                )
        except (AttributeError, OSError, ValueError) as exc:
            raise supervisor.Refused(
                f"acceptance session-spend reconciliation failed: {exc}"
            ) from exc
        return result

    def _wait_phase(self, context: dict[str, Any], phase: str) -> dict[str, Any]:
        while True:
            state = self._state(context)
            if state["probe"].get("phase") == phase:
                return state["probe"]
            if state["runtime"].get("terminal"):
                raise supervisor.Refused(f"probe exited before {phase}: {state}")
            time.sleep(10)

    @staticmethod
    def _download_snapshot(context: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
        snapshot = probe["snapshot"]
        entry = {"path": snapshot["path"], "size": snapshot["bytes"], "sha256": snapshot["sha256"]}
        supervisor.export_file(
            context["cli"],
            context["root"],
            entry,
            context["output"],
            context["manifest"]["max_export_bytes"],
        )
        return entry

    def _final_artifact_download(self, context: dict[str, Any], probe: dict[str, Any]) -> None:
        """Time the complete L4 export path, including split, markers and Mac hashing."""
        snapshot = probe["snapshot"]
        if snapshot["bytes"] != 64 * 1024 * 1024 or snapshot["path"] != "final-artifact.bin":
            raise supervisor.Refused("final-only requires exactly 64 MiB final artifact")
        started = time.monotonic()
        entry = self._download_snapshot(context, probe)
        local = context["output"] / "exports" / entry["path"]
        if local.stat().st_size != entry["size"] or supervisor.sha256(local) != entry["sha256"]:
            raise supervisor.Refused("final artifact Mac hash/size mismatch")
        elapsed = time.monotonic() - started
        if elapsed <= 0:
            raise supervisor.Refused("final artifact transfer elapsed time is not positive")
        self._record(
            "final-artifact-download",
            status="PASS",
            hardware="L4",
            profile=self.profile,
            path=str(local),
            sha256=entry["sha256"],
            exported_bytes=entry["size"],
            elapsed_seconds=elapsed,
            bytes_per_second_lower_bound=int(entry["size"] / elapsed),
            verified_on_mac=True,
            scope="end-to-end immutable final export including Mac verification",
        )

    def _busy_transfer_and_client_loss(self, context: dict[str, Any]) -> None:
        """Kill one owned CLI client; observe completion via Contents, never resubmit."""
        cli, root, output = context["cli"], context["root"], context["output"]
        marker_remotes = {root + "/client-start.json", root + "/client-finished.json"}
        marker_evidence = DirectObserverDownloadEvidence(
            cli, marker_remotes, verification_kind="census-file-digest"
        )
        declared_absences = self._declared_absence_selectors(context["manifest"].get("job_id"))
        probe = self._wait_phase(context, "transfer_ready")
        identity = {key: probe[key] for key in ("pid", "execution_id")}
        before = cli.sessions(timeout=policy.READ_SECONDS)
        self._state(context)
        cli.rpc(
            "from pathlib import Path\n"
            f"Path({root + '/acceptance-transfer-start'!r}).touch()\nresult=True"
        )
        # The probe's supervised child is now computing while the notebook
        # kernel is idle.  Prepare and transfer the snapshot before injecting
        # the separate 120-second busy-kernel fault: chunk assembly/splitting
        # and lease renewal are RPCs, so queuing them behind that fault would
        # be a false recovery test rather than a bounded transfer test.
        transfer_started = time.time()
        entry = self._download_snapshot(context, probe)
        # This acknowledgement ends the paired checkpoint-overhead interval.
        # The optional full roundtrip below is integrity evidence and must not
        # inflate a gate intended to measure the single required export handoff.
        ack = output / ".acceptance-transfer-done"
        ack.write_text("complete\n")
        try:
            upload_contents(cli, ack, root + "/acceptance-transfer-done")
        finally:
            ack.unlink(missing_ok=True)
        transfer_finished = time.time()

        payload = output / "exports" / entry["path"]
        from research.colab_cli.transfers import download_file, upload_file

        if self.profile == "full-state":
            roundtrip_started = time.time()
            upload_file(cli, payload, root + "/roundtrip.bin", renew=cli.renew)
            roundtrip = Path(output) / ".acceptance-roundtrip.bin"
            try:
                download_file(
                    cli,
                    root + "/roundtrip.bin",
                    roundtrip,
                    entry["size"],
                    renew=cli.renew,
                    expected_sha256=entry["sha256"],
                )
                if supervisor.sha256(roundtrip) != entry["sha256"]:
                    raise supervisor.Refused("full-size supervised-child roundtrip hash mismatch")
            finally:
                roundtrip.unlink(missing_ok=True)
            roundtrip_finished = time.time()
        request = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(dir=output, prefix=".client-loss-") as temp:
            directory = Path(temp)
            code = directory / "silent.py"
            code.write_text(
                "import json,time\nfrom pathlib import Path\n"
                f"root=Path({root!r})\nrequest={request!r}\n"
                "assert not (root/'client-start.json').exists()\n"
                "temp=root/'client-start.tmp'\n"
                "temp.write_text(json.dumps({'request':request,'started_at':time.time()}))\n"
                "temp.replace(root/'client-start.json')\n"
                "deadline=time.monotonic()+300\n"
                "while not (root/'client-release.json').exists():\n"
                "    if time.monotonic() >= deadline:\n"
                "        raise RuntimeError('client-loss release timeout')\n"
                "    time.sleep(0.2)\n"
                "done={'request':request,'finished_at':time.time()}\n"
                "temp=root/'client-finished.tmp'\ntemp.write_text(json.dumps(done))\n"
                "temp.replace(root/'client-finished.json')\n"
            )
            environment = {
                **os.environ,
                "NO_COLOR": "1",
            }
            argv = [
                cli.binary,
                "--auth",
                "oauth2",
                "--config",
                str(cli.config),
                "exec",
                "-s",
                cli.session,
                "-f",
                str(code),
                "--timeout",
                "300",
            ]
            # This process is deliberately the fault subject; its PID is owned here.
            client = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                umask=0o077,
            )
            began = time.monotonic()
            try:
                started_file = directory / "started.json"
                started_first_call = self._wait_file(
                    cli,
                    root + "/client-start.json",
                    started_file,
                    began + 150,
                    phase="client-loss start",
                    declared_selectors=declared_absences,
                    evidence=marker_evidence,
                )
                started = json.loads(started_file.read_text())
                if client.poll() is not None:
                    raise supervisor.Refused(
                        "client exited before intentional loss could be injected"
                    )
                client.terminate()
                if client.wait(timeout=5) != -signal.SIGTERM:
                    raise supervisor.Refused("owned client termination was not confirmed")
                finished_file = directory / "finished.json"
                try:
                    finished_probe_first = len(cli.call_records)
                    _contents_call(
                        cli,
                        (
                            "download",
                            "-s",
                            cli.session,
                            root + "/client-finished.json",
                            str(finished_file),
                        ),
                        policy.READ_SECONDS,
                        None,
                        deadline=began + 300,
                    )
                except supervisor.Refused as exc:
                    if "not found" not in str(exc).lower() and "404" not in str(exc):
                        raise
                    self._tag_observed_absence(
                        cli,
                        root + "/client-finished.json",
                        declared_selectors=declared_absences,
                    )
                else:
                    raise supervisor.Refused(
                        "kernel became idle before the client-loss observation"
                    )
                finally:
                    marker_evidence.observe_since(finished_probe_first)
                release = directory / "release.json"
                release.write_text(json.dumps({"request": request}))
                upload_contents(cli, release, root + "/client-release.json", deadline=began + 300)
                finished_first_call = self._wait_file(
                    cli,
                    root + "/client-finished.json",
                    finished_file,
                    began + 300,
                    phase="client-loss completion",
                    declared_selectors=declared_absences,
                    evidence=marker_evidence,
                )
                finished = json.loads(finished_file.read_text())
                if started["request"] != request or finished["request"] != request:
                    raise supervisor.Refused("client-loss kernel request identity mismatch")
                cli.record_client_marker_download(
                    first_call=started_first_call,
                    remote=root + "/client-start.json",
                    local_size=started_file.stat().st_size,
                    local_sha256=supervisor.sha256(started_file),
                )
                cli.record_client_marker_download(
                    first_call=finished_first_call,
                    remote=root + "/client-finished.json",
                    local_size=finished_file.stat().st_size,
                    local_sha256=supervisor.sha256(finished_file),
                )
            finally:
                if client.poll() is None:
                    client.terminate()
                    try:
                        client.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        client.kill()
                        client.wait(timeout=5)
        reconnected = supervisor.CLI(cli.binary, cli.config, cli.session)
        reconnected.deadline = cli.deadline
        reconnected.priced_deadline = cli.priced_deadline
        # A CLI process is stateless; preserve the broker-owned accounting and
        # lease authority when using a fresh client object for the reconnect
        # proof.  No credentials are copied to the runtime.
        reconnected.accounting = cli.accounting
        reconnected.renew = cli.renew
        reconnected.release_credential = cli.release_credential
        reconnected.transfer_records = cli.transfer_records
        reconnected.contents_transfers = cli.contents_transfers
        reconnected.call_records = cli.call_records
        reconnected._evidence_operations = cli._evidence_operations
        reconnected.client_marker_downloads = cli.client_marker_downloads
        reconnected._lease_stop = cli._lease_stop
        reconnected._lease_thread = cli._lease_thread
        context["cli"] = reconnected
        observed = self._state(context)
        after_identity = {key: observed["probe"][key] for key in identity}
        if (
            after_identity != identity
            or reconnected.sessions(timeout=policy.READ_SECONDS) != before
        ):
            raise supervisor.Refused("job identity changed after client loss/reconnect")
        self._record(
            "client-loss-reconnect",
            killed_client_pid=client.pid,
            job=identity,
            kernel_request=request,
            finished=finished,
            sessions=before,
        )
        if self.profile == "full-state":
            self._record(
                "busy-full-snapshot",
                bytes=entry["size"],
                sha256=entry["sha256"],
                transfer_started=transfer_started,
                transfer_finished=transfer_finished,
                measured_handoff_seconds=transfer_finished - transfer_started,
                roundtrip_seconds=roundtrip_finished - roundtrip_started,
                roundtrip_outside_checkpoint_overhead_gate=True,
                kernel_started=started["started_at"],
                kernel_finished=finished["finished_at"],
                kernel_completion_marker_absent_after_client_loss=True,
            )
            measured = self._wait_phase(context, "transfer_finished")["measurement"]
            self._record(
                "checkpoint-overhead",
                **measured,
                status="PASS" if measured["overhead_percent"] <= 2 else "INCOMPLETE: overhead >2%",
            )
        else:
            self._wait_phase(context, "transfer_finished")
        reconnected.rpc(
            f"from pathlib import Path\nPath({root + '/acceptance-finish'!r}).touch()\nresult=True"
        )

    def _declared_absence_selectors(self, job_id: Any) -> dict[str, str]:
        manifest = self.injections.get(job_id) if isinstance(job_id, str) else None
        if not isinstance(manifest, dict) or manifest.get("written_before_dispatch") is not True:
            return {}
        selectors: dict[str, str] = {}
        for entry in manifest.get("entries", []):
            selector = entry.get("selector") if isinstance(entry, dict) else None
            if (
                isinstance(entry, dict)
                and entry.get("command") == "download"
                and entry.get("failure_type") == "NOT_FOUND"
                and entry.get("realized_event_required") is True
                and isinstance(entry.get("injection_id"), str)
                and isinstance(selector, dict)
                and selector.get("kind") == "planned_absence_probe"
                and isinstance(selector.get("remote_suffix"), str)
                and selector["remote_suffix"]
            ):
                selectors[selector["remote_suffix"]] = entry["injection_id"]
        return selectors

    @staticmethod
    def _tag_observed_absence(
        cli: supervisor.CLI,
        remote: str,
        *,
        declared_selectors: dict[str, str] | None = None,
    ) -> None:
        """Record only a declared terminal CLI missing-file response for its exact remote.

        ``None`` retains the original static test-helper call shape. Runtime
        observer paths pass the pre-dispatch manifest-derived mapping.
        """
        selectors = (
            declared_selectors
            if declared_selectors is not None
            else {
                "/client-start.json": "client-loss-start-absent",
                "/client-finished.json": "client-loss-finished-absent",
                "/output/lease-checkpoint.pkl.sha256": "watchdog-checkpoint-marker-absent",
                "/output/lease-checkpoint.pkl": "watchdog-checkpoint-absent",
            }
        )
        identity = next(
            (ident for suffix, ident in selectors.items() if remote.endswith(suffix)), None
        )
        if not isinstance(identity, str) or not identity:
            return
        for row in reversed(cli.call_records):
            if row.get("command") != "download" or row.get("remote") != remote:
                continue
            error = str(row.get("error", ""))
            terminal = re.search(
                r"(?:^|\n)\[colab\] Download failed: File or directory not found: ([^\r\n]+)\s*\Z",
                error,
            )
            lower = error.lower()
            terminal_text = error.strip().splitlines()[-1] if error.strip() else ""
            terminal_status = terminal_text.lower()
            if terminal_status.startswith("cli_guard: guarderror: "):
                terminal_status = terminal_status.removeprefix("cli_guard: guarderror: ")
            http_404 = bool(
                re.match(
                    r"(?:http(?:error)?|status(?: code)?)[ :=]*404\b|"
                    r"404\s+(?:client error:\s*)?not found\b",
                    terminal_status,
                )
            )
            reported_paths = re.findall(r"/[^\s\"']+", terminal_text)
            http_404 = http_404 and all(path == remote for path in reported_paths)
            has_5xx = bool(
                re.search(
                    r"\b(?:http(?:error)?|status)[ :=]*5\d\d\b|\b5\d\d (?:service|server)", lower
                )
            )
            timeout = (
                "TIMEOUT" in str(row.get("failure_class", "")).upper()
                or "timed out" in lower
                or "timeoutexpired" in lower
            )
            exact_cli = (
                declared_selectors is not None
                and terminal is not None
                and terminal.group(1) == remote
            )
            exact_cli = exact_cli and "traceback" not in lower
            # Retain the prior terminal HTTP-404 recognition. The new CLI
            # wording needs both this exact remote and a manifest selector.
            absent = (http_404 or exact_cli) and "traceback" not in lower
            if row.get("status") != "completed" and absent and not has_5xx and not timeout:
                row.update(failure_type="NOT_FOUND", realized_selector_id=identity)
            break

    @staticmethod
    def _wait_file(
        cli: supervisor.CLI,
        remote: str,
        local: Path,
        deadline: float,
        *,
        phase: str = "kernel completion",
        declared_selectors: dict[str, str] | None = None,
        evidence: DirectObserverDownloadEvidence | None = None,
    ) -> int:
        started = time.monotonic()
        deadline = min(
            deadline,
            cli.deadline if cli.deadline is not None else float("inf"),
            cli.priced_deadline if cli.priced_deadline is not None else float("inf"),
        )
        while time.monotonic() < deadline:
            try:
                first_call = len(cli.call_records)
                try:
                    cli.call(
                        "download",
                        "-s",
                        cli.session,
                        remote,
                        str(local),
                        timeout=min(policy.READ_SECONDS, deadline - time.monotonic()),
                    )
                finally:
                    if evidence is not None:
                        evidence.observe_since(first_call)
                return first_call
            except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
                missing = "404" in str(exc) or "not found" in str(exc).lower()
                if not missing and not policy.transient_error(exc):
                    raise supervisor.Refused(
                        f"{phase}: remote file {remote!r} failed after "
                        f"{time.monotonic() - started:.3f}s: {exc}; refusing exec resubmission"
                    ) from exc
                if missing:
                    AcceptanceObserver._tag_observed_absence(
                        cli, remote, declared_selectors=declared_selectors
                    )
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        raise supervisor.Refused(
            f"{phase}: remote file {remote!r} not observed after "
            f"{time.monotonic() - started:.3f}s "
            f"(wait budget {max(0, deadline - started):.3f}s); refusing exec resubmission"
        )

    def _corrupt_export_rejection(self, cli: supervisor.CLI, root: str, output: Path) -> None:
        name = "acceptance-corrupt.bin"
        fault_root = root + "-fault"
        cli.rpc(
            "from pathlib import Path\n"
            f"p=Path({fault_root!r})/'output'/{name!r}\n"
            "p.parent.mkdir(parents=True,exist_ok=True)\n"
            "p.write_bytes(b'known-good')\n"
            "p.with_name(p.name+'.sha256').write_text('0'*64+'  '+p.name+'\\n')\n"
            "p.with_name(p.name+'.complete').write_text('complete\\n')\n"
            "result=True",
            timeout=policy.RPC_SECONDS,
        )
        try:
            supervisor.export_file(
                cli,
                fault_root,
                {"path": name, "size": len(b"known-good"), "sha256": "0" * 64},
                output,
                1024,
            )
        except supervisor.Refused as exc:
            from research.colab_cli.transfers import TransferError

            # export_file's wrapper also says "hash mismatch" for provider
            # failures. Only the actual local digest/size rejection realizes
            # this planned event; a 503 on the same path must stay a real C6.
            cause: BaseException = exc
            while cause.__cause__ is not None:
                cause = cause.__cause__
            if not (
                isinstance(cause, TransferError)
                and str(cause) == "download digest or size mismatch"
            ):
                raise
            for parent in reversed(cli.transfer_records):
                if parent.get("remote") == fault_root + "/output/" + name:
                    parent.update(
                        failure_type="HASH_MISMATCH",
                        realized_selector_id="corrupt-export-rejection",
                    )
                    for row in cli.call_records:
                        if row.get("parent_transfer_id") == parent.get("transfer_id"):
                            row.update(
                                failure_type="HASH_MISMATCH",
                                realized_selector_id="corrupt-export-rejection",
                            )
                    break
            # Remove the isolated synthetic fixture after rejection.
            cli.rpc(
                "from pathlib import Path\n"
                f"p=Path({fault_root!r})/'output'/{name!r}\n"
                "files=(p,p.with_name(p.name+'.sha256'),p.with_name(p.name+'.complete'))\n"
                "[q.unlink() for q in files if q.exists()]\n"
                "result=True",
                timeout=policy.RPC_SECONDS,
            )
            self._record("failure-and-corrupt-export", corrupt_export_rejected=True)
            return
        raise supervisor.Refused("corrupt export was accepted")

    def _verify_terminal_exports(self, exports: list[dict[str, Any]], output: Path) -> None:
        names = {item["path"] for item in exports}
        required = {
            "busy-snapshot.bin",
            "toy-resume.json",
            "toy-checkpoint.pkl",
            "checkpoint-overhead.json",
            "intentional-child-failure.json",
        }
        if self.profile == "final-only":
            required = {"client-job.bin", "intentional-child-failure.json"}
        if not required <= names:
            raise supervisor.Refused(
                f"acceptance export set incomplete: missing {required - names}"
            )
        for name in required:
            local = output / "exports" / name
            if not local.is_file() or not (local.with_name(local.name + ".complete")).is_file():
                raise supervisor.Refused(f"missing verified local export: {name}")
        if self.profile == "full-state":
            toy = json.loads((output / "exports" / "toy-resume.json").read_text())
            if not toy.get("exact") or "not evidence for a trainer" not in toy.get("scope", ""):
                raise supervisor.Refused("toy resume result is not explicit/identical")
        child = json.loads((output / "exports" / "intentional-child-failure.json").read_text())
        if child.get("exit_code") != 23:
            raise supervisor.Refused("intentional child failure was not observed")
        self.records["failure-and-corrupt-export"]["intentional_child_exit"] = 23

    def _resume_import(self, context: dict[str, Any]) -> None:
        cli, root, output = context["cli"], context["root"], context["output"]
        checkpoint = output / "exports" / "toy-checkpoint.pkl"
        digest = supervisor.sha256(checkpoint)
        fault_root = root + "/resume-test"
        cli.rpc(
            "from pathlib import Path\n" f"Path({fault_root!r}).mkdir(exist_ok=False)\nresult=True"
        )
        upload_file(cli, checkpoint, fault_root + "/input.pkl", renew=cli.renew)
        result = cli.rpc(
            "import hashlib,subprocess\nfrom pathlib import Path\n"
            f"p=Path({fault_root!r})\n"
            f"assert hashlib.sha256((p/'input.pkl').read_bytes()).hexdigest()=={digest!r}\n"
            f"args=[{root + '/.venv/bin/python'!r},"
            f"{root + '/repo/research/colab_cli/acceptance_probe.py'!r},"
            "'--resume-input',str(p/'input.pkl'),'--resume-output',str(p/'output'/'resume.json')]\n"
            "subprocess.run(args,check=True,timeout=120)\n"
            "payload=p/'output'/'resume.json'\n"
            "result={'path':'resume.json','size':payload.stat().st_size,"
            "'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}",
            timeout=policy.RPC_SECONDS,
        )
        supervisor.export_file(cli, fault_root, result, output / "resume-roundtrip", 1024 * 1024)
        resumed = json.loads((output / "resume-roundtrip/exports/resume.json").read_text())
        expected = json.loads((output / "exports/toy-resume.json").read_text())
        if resumed["resumed_sha256"] != expected["uninterrupted_sha256"] or resumed["steps"] != 9:
            raise supervisor.Refused(
                "export/import full-state resume differs from uninterrupted toy"
            )
        self._record(
            "exact-resume",
            input_sha256=digest,
            resumed=resumed,
            expected=expected,
            scope="toy full-state transport; real trainers need their own resume gate",
        )

    def _watchdog(self, context: dict[str, Any]) -> None:
        cli: supervisor.CLI = context["cli"]
        lease = int(context["manifest"]["lease_seconds"])
        endpoint = getattr(getattr(cli, "accounting", None), "endpoint", None)
        if not isinstance(endpoint, str) or not endpoint:
            raise supervisor.Refused("watchdog has no allocated endpoint to verify")
        # Reconcile spend first, then quiesce every independent Contents PUT.
        # The final small RPC renews on the VM; no kernel work follows it.
        self._state(context)
        cli.stop_lease(wait=True)
        renewal_started = time.monotonic()
        cli.poll_rpc(
            "import importlib.util\nfrom pathlib import Path\n"
            "s=importlib.util.spec_from_file_location('watchdog_runtime',"
            f"{context['root']+'/runtime.py'!r})\n"
            "m=importlib.util.module_from_spec(s)\ns.loader.exec_module(m)\n"
            f"m.read_state({context['root']!r},{context['generation']!r},"
            "renew=True,disable_contents=True)\n"
            "result=True"
        )
        renewal_finished = time.monotonic()
        # The designed release makes the backend unavailable. Observe once
        # after its final kernel request, before entering the no-renewal wait.
        # Evidence collection never changes the watchdog or cleanup decision.
        try:
            cli.collect_marker_census()
        except BaseException:
            pass
        # RPC latency gives an interval, not an exact local expiry timestamp.
        # Collect immediately so slow acknowledgements cannot skip the remote
        # 15s checkpoint window. The disappearance deadline uses the EARLIEST
        # possible expiry: passing it proves the unchanged <=300s bound.
        lease_expiry = renewal_started + lease
        latest_lease_expiry = renewal_finished + lease
        cap_deadline = min(
            getattr(cli, "deadline", None) or float("inf"),
            getattr(cli, "priced_deadline", None) or float("inf"),
        )
        checkpoint_deadline = min(
            latest_lease_expiry + WATCHDOG_CHECKPOINT_GRACE_SECONDS, cap_deadline
        )
        checkpoint_hash = None
        last_checkpoint_error = None
        checkpoint_remotes = {
            context["root"] + "/output/lease-checkpoint.pkl.sha256",
            context["root"] + "/output/lease-checkpoint.pkl",
        }
        checkpoint_evidence = DirectObserverDownloadEvidence(cli, checkpoint_remotes)
        declared_absences = self._declared_absence_selectors(context["manifest"].get("job_id"))
        with tempfile.TemporaryDirectory(dir=context["output"], prefix=".watchdog-") as temp:
            marker = Path(temp) / "checkpoint.sha256"
            local = Path(temp) / "checkpoint.pkl"
            while time.monotonic() < checkpoint_deadline:
                first_call = len(cli.call_records)
                try:
                    cli.call(
                        "download",
                        "-s",
                        cli.session,
                        context["root"] + "/output/lease-checkpoint.pkl.sha256",
                        str(marker),
                        timeout=min(
                            WATCHDOG_CHECKPOINT_READ_SECONDS,
                            max(0.01, checkpoint_deadline - time.monotonic()),
                        ),
                    )
                    candidate = marker.read_text().split()[0]
                    cli.call(
                        "download",
                        "-s",
                        cli.session,
                        context["root"] + "/output/lease-checkpoint.pkl",
                        str(local),
                        timeout=min(
                            WATCHDOG_CHECKPOINT_READ_SECONDS,
                            max(0.01, checkpoint_deadline - time.monotonic()),
                        ),
                    )
                    if supervisor.sha256(local) != candidate:
                        raise supervisor.Refused("watchdog checkpoint hash mismatch")
                    checkpoint_evidence.observe_since(first_call)
                    checkpoint_evidence.bind_verified(
                        context["root"] + "/output/lease-checkpoint.pkl.sha256", candidate
                    )
                    checkpoint_evidence.bind_verified(
                        context["root"] + "/output/lease-checkpoint.pkl", candidate
                    )
                    checkpoint_hash = candidate
                    break
                except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
                    missing = "404" in str(exc) or "not found" in str(exc).lower()
                    if missing:
                        for suffix in (
                            "/output/lease-checkpoint.pkl.sha256",
                            "/output/lease-checkpoint.pkl",
                        ):
                            self._tag_observed_absence(
                                cli,
                                context["root"] + suffix,
                                declared_selectors=declared_absences,
                            )
                    if not missing and not policy.transient_error(exc):
                        raise
                    last_checkpoint_error = supervisor.redact(repr(exc))
                    time.sleep(min(1, max(0, checkpoint_deadline - time.monotonic())))
                finally:
                    checkpoint_evidence.observe_since(first_call)
            if checkpoint_hash is None:
                raise supervisor.Refused(
                    "watchdog checkpoint was not observed during grace; "
                    f"last download error: {last_checkpoint_error}"
                )
            destination = context["output"] / "watchdog-checkpoint.pkl"
            local.replace(destination)
            marker.replace(destination.with_suffix(".pkl.sha256"))
        observed_at = None
        release_wait_started = lease_expiry
        deadline = min(lease_expiry + WATCHDOG_DISAPPEARANCE_BOUND_SECONDS, cap_deadline)
        last_sessions: dict[str, str] = {}
        while time.monotonic() < deadline:
            # Read-only account polling is deliberately permitted after the
            # simulated Mac loss; it does not renew the lease or queue kernel
            # work.  A release only counts when this exact backend endpoint
            # vanishes: a `?` session label can otherwise hide a paid orphan.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                last_sessions = cli.sessions(timeout=min(policy.READ_SECONDS, remaining))
            except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
                if not policy.transient_error(exc):
                    raise
                time.sleep(min(2, max(0, deadline - time.monotonic())))
                continue
            if endpoint not in last_sessions:
                observed_at = time.monotonic()
                if observed_at <= deadline:
                    break
                observed_at = None
                break
            time.sleep(WATCHDOG_DISAPPEARANCE_POLL_SECONDS)
        if observed_at is None:
            raise supervisor.Refused(
                "watchdog lease expired but assignment did not disappear within "
                f"{WATCHDOG_DISAPPEARANCE_BOUND_SECONDS} seconds"
            )
        if observed_at < lease_expiry:
            raise supervisor.Refused("assignment disappeared before lease expiry; cause unproven")
        self._record(
            "mac-loss-watchdog-unassign",
            job_id=context["manifest"]["job_id"],
            planned_event_id="mac-loss-watchdog-unassign",
            lease_seconds=lease,
            lease_expiry_uncertainty_seconds=renewal_finished - renewal_started,
            endpoint=endpoint,
            release_bound_seconds=WATCHDOG_DISAPPEARANCE_BOUND_SECONDS,
            observed_after_lease_seconds=observed_at - release_wait_started,
            sessions_after=last_sessions,
            checkpoint_sha256=checkpoint_hash,
            checkpoint_exported_during_grace=True,
            renewal_suppressed=True,
            assignment_disappeared_within_bound=True,
        )

    def write(self, output: Path) -> None:
        path = output / "acceptance-observer.json"
        if path.exists():
            raise supervisor.Refused("acceptance observer receipt already exists")
        path.write_text(
            json.dumps({"profile": self.profile, "gates": self.records}, indent=2, sort_keys=True)
            + "\n"
        )
