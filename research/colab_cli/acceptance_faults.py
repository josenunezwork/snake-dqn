"""Live acceptance observer for the bounded Colab CLI qualification.

It runs only through ``acceptance --execute-after-review``.  The observer uses
an already allocated broker session and never calls ``new`` or retries an
allocation.  Each operation records an explicit evidence result; a failure
raises so the broker retains its normal cleanup and writes an INCOMPLETE receipt.
"""

from __future__ import annotations

import json
import math
import os
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

    def __init__(self, oauth_python: str, profile: str = "final-only") -> None:
        self.records: dict[str, dict[str, Any]] = {}
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
            self._silence_then_bounded_rpc(cli)
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
        cli.call("whoami")
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
        cli.call("url", "-s", cli.session, timeout=policy.READ_SECONDS)
        after = cli.sessions(timeout=policy.READ_SECONDS)
        if before != after or list(after.values()).count(cli.session) != 1:
            raise supervisor.Refused("url changed assignment identity or count")
        self._record("url-no-second-allocation", sessions_before=before, sessions_after=after)

    def _silence_then_bounded_rpc(self, cli: supervisor.CLI) -> None:
        started = time.monotonic()
        # The runtime child is an independently supervised subprocess. This
        # bounded kernel call creates an actual >60-second execution silence;
        # its explicit timeout is evidence that no unbounded wait is hidden.
        value = cli.rpc("import time\ntime.sleep(65)\nresult={'after_silence':True}", timeout=120)
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
                self._wait_file(
                    cli,
                    root + "/client-start.json",
                    started_file,
                    began + 150,
                    phase="client-loss start",
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
                else:
                    raise supervisor.Refused(
                        "kernel became idle before the client-loss observation"
                    )
                release = directory / "release.json"
                release.write_text(json.dumps({"request": request}))
                upload_contents(cli, release, root + "/client-release.json", deadline=began + 300)
                self._wait_file(
                    cli,
                    root + "/client-finished.json",
                    finished_file,
                    began + 300,
                    phase="client-loss completion",
                )
                finished = json.loads(finished_file.read_text())
                if started["request"] != request or finished["request"] != request:
                    raise supervisor.Refused("client-loss kernel request identity mismatch")
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

    @staticmethod
    def _wait_file(
        cli: supervisor.CLI,
        remote: str,
        local: Path,
        deadline: float,
        *,
        phase: str = "kernel completion",
    ) -> None:
        started = time.monotonic()
        deadline = min(
            deadline,
            cli.deadline if cli.deadline is not None else float("inf"),
            cli.priced_deadline if cli.priced_deadline is not None else float("inf"),
        )
        while time.monotonic() < deadline:
            try:
                cli.call(
                    "download",
                    "-s",
                    cli.session,
                    remote,
                    str(local),
                    timeout=min(policy.READ_SECONDS, deadline - time.monotonic()),
                )
                return
            except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
                missing = "404" in str(exc) or "not found" in str(exc).lower()
                if not missing and not policy.transient_error(exc):
                    raise supervisor.Refused(
                        f"{phase}: remote file {remote!r} failed after "
                        f"{time.monotonic() - started:.3f}s: {exc}; refusing exec resubmission"
                    ) from exc
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
            if "hash mismatch" not in str(exc):
                raise
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
        with tempfile.TemporaryDirectory(dir=context["output"], prefix=".watchdog-") as temp:
            marker = Path(temp) / "checkpoint.sha256"
            local = Path(temp) / "checkpoint.pkl"
            while time.monotonic() < checkpoint_deadline:
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
                    checkpoint_hash = candidate
                    break
                except (supervisor.Refused, OSError, subprocess.SubprocessError) as exc:
                    missing = "404" in str(exc) or "not found" in str(exc).lower()
                    if not missing and not policy.transient_error(exc):
                        raise
                    last_checkpoint_error = supervisor.redact(repr(exc))
                    time.sleep(min(1, max(0, checkpoint_deadline - time.monotonic())))
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
