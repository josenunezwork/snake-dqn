"""Live acceptance observer for the bounded Colab CLI qualification.

It runs only through ``acceptance --execute-after-review``.  The observer uses
an already allocated broker session and never calls ``new`` or retries an
allocation.  Each operation records an explicit evidence result; a failure
raises so the broker retains its normal cleanup and writes an INCOMPLETE receipt.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from research.colab_cli import supervisor

WATCHDOG_DISAPPEARANCE_BOUND_SECONDS = 300
WATCHDOG_DISAPPEARANCE_POLL_SECONDS = 10
WATCHDOG_CHECKPOINT_GRACE_SECONDS = 15


class AcceptanceObserver:
    """Exercise account/kernel fault boundaries without a second allocation."""

    def __init__(self, oauth_python: str) -> None:
        self.records: dict[str, dict[str, Any]] = {}
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
        if stage == "ready" and manifest["requirements"]["gpu"] == "L4":
            probe = self._wait_phase(context, "transfer_ready")
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
            self._resume_import(context)

    def _refresh_oauth_then_query(self, cli: supervisor.CLI) -> None:
        """Refresh in memory without modifying or printing the credential file."""
        code = (
            "from google.oauth2.credentials import Credentials\n"
            "from google.auth.transport.requests import Request\n"
            "from colab_cli.auth import TOKEN_CONFIG_PATH,PUBLIC_SCOPES\n"
            "c=Credentials.from_authorized_user_file(TOKEN_CONFIG_PATH,PUBLIC_SCOPES)\n"
            "assert c.refresh_token\n"
            "c.refresh(Request())\n"
            "assert c.valid\n"
            "print('REFRESH_OK')\n"
        )
        try:
            result = subprocess.run(
                [self.oauth_python, "-c", code],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise supervisor.Refused(
                f"OAuth refresh subprocess failed: {type(exc).__name__}"
            ) from exc
        if result.returncode or result.stdout.strip() != "REFRESH_OK":
            raise supervisor.Refused("OAuth refresh did not confirm reusable credentials")
        # Official CLI 0.7.4 exposes whoami even though its terse help omits it.
        cli.call("whoami", timeout=30)
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
        before = cli.sessions(timeout=10)
        cli.call("url", "-s", cli.session, timeout=20)
        after = cli.sessions(timeout=10)
        if before != after or list(after.values()).count(cli.session) != 1:
            raise supervisor.Refused("url changed assignment identity or count")
        self._record("url-no-second-allocation", sessions_before=before, sessions_after=after)

    def _silence_then_bounded_rpc(self, cli: supervisor.CLI) -> None:
        started = time.monotonic()
        # The runtime child is an independently supervised subprocess. This
        # bounded kernel call creates an actual >60-second execution silence;
        # its explicit timeout is evidence that no unbounded wait is hidden.
        value = cli.rpc("import time\ntime.sleep(65)\nresult={'after_silence':True}", timeout=90)
        self._record(
            "silence-timeout",
            silence_seconds=time.monotonic() - started,
            rpc_timeout_seconds=90,
            rpc_value=value,
        )

    @staticmethod
    def _state(context: dict[str, Any]) -> dict[str, Any]:
        cli, root, generation = context["cli"], context["root"], context["generation"]
        result = cli.rpc(
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

    def _busy_transfer_and_client_loss(self, context: dict[str, Any]) -> None:
        """Kill one owned CLI client; observe completion via Contents, never resubmit."""
        cli, root, output = context["cli"], context["root"], context["output"]
        probe = self._wait_phase(context, "transfer_ready")
        identity = {key: probe[key] for key in ("pid", "execution_id")}
        before = cli.sessions(timeout=10)
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
            cli.call(
                "upload",
                "-s",
                cli.session,
                str(ack),
                root + "/acceptance-transfer-done",
                timeout=10,
            )
        finally:
            ack.unlink(missing_ok=True)
        transfer_finished = time.time()

        payload = output / "exports" / entry["path"]
        from research.colab_cli.transfers import download_file, upload_file

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
                "time.sleep(120)\n"
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
                "150",
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
                self._wait_file(cli, root + "/client-start.json", started_file, began + 25)
                started = json.loads(started_file.read_text())
                client.terminate()
                client.wait(timeout=5)
                finished_file = directory / "finished.json"
                try:
                    cli.call(
                        "download",
                        "-s",
                        cli.session,
                        root + "/client-finished.json",
                        str(finished_file),
                        timeout=10,
                    )
                except supervisor.Refused as exc:
                    if "not found" not in str(exc).lower() and "404" not in str(exc):
                        raise
                else:
                    raise supervisor.Refused(
                        "kernel became idle before the client-loss observation"
                    )
                self._wait_file(cli, root + "/client-finished.json", finished_file, began + 155)
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
        # A CLI process is stateless; preserve the broker-owned accounting and
        # lease authority when using a fresh client object for the reconnect
        # proof.  No credentials are copied to the runtime.
        reconnected.accounting = cli.accounting
        reconnected.renew = cli.renew
        reconnected.release_credential = cli.release_credential
        context["cli"] = reconnected
        observed = self._state(context)
        after_identity = {key: observed["probe"][key] for key in identity}
        if after_identity != identity or reconnected.sessions(timeout=10) != before:
            raise supervisor.Refused("job identity changed after client loss/reconnect")
        self._record(
            "client-loss-reconnect",
            killed_client_pid=client.pid,
            job=identity,
            kernel_request=request,
            finished=finished,
            sessions=before,
        )
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
        reconnected.rpc(
            f"from pathlib import Path\nPath({root + '/acceptance-finish'!r}).touch()\nresult=True"
        )

    @staticmethod
    def _wait_file(cli: supervisor.CLI, remote: str, local: Path, deadline: float) -> None:
        while time.monotonic() < deadline:
            try:
                cli.call("download", "-s", cli.session, remote, str(local), timeout=10)
                return
            except supervisor.Refused:
                time.sleep(2)
        raise supervisor.Refused("kernel completion file not observed; refusing exec resubmission")

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
            timeout=20,
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
                timeout=20,
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
        if not required <= names:
            raise supervisor.Refused(
                f"acceptance export set incomplete: missing {required - names}"
            )
        for name in required:
            local = output / "exports" / name
            if not local.is_file() or not (local.with_name(local.name + ".complete")).is_file():
                raise supervisor.Refused(f"missing verified local export: {name}")
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
        cli.call(
            "upload", "-s", cli.session, str(checkpoint), fault_root + "/input.pkl", timeout=20
        )
        result = cli.rpc(
            "import hashlib,subprocess\nfrom pathlib import Path\n"
            f"p=Path({fault_root!r})\n"
            f"assert hashlib.sha256((p/'input.pkl').read_bytes()).hexdigest()=={digest!r}\n"
            f"args=[{root + '/.venv/bin/python'!r},"
            f"{root + '/repo/research/colab_cli/acceptance_probe.py'!r},"
            "'--resume-input',str(p/'input.pkl'),'--resume-output',str(p/'output'/'resume.json')]\n"
            "subprocess.run(args,check=True,timeout=10)\n"
            "payload=p/'output'/'resume.json'\n"
            "result={'path':'resume.json','size':payload.stat().st_size,"
            "'sha256':hashlib.sha256(payload.read_bytes()).hexdigest()}",
            timeout=20,
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
        # Suppress renewal longer than the lease. The runtime, not the Mac,
        # must checkpoint/attempt unassign. The next sessions call is evidence.
        self._state(context)
        lease_expiry = time.monotonic() + lease
        time.sleep(lease + 1)
        # Reconnect only to file transport during the bounded checkpoint grace.
        # It never renews the failed lease or submits a kernel execution.  The
        # runtime may need most of its 15-second signal/checkpoint grace before
        # the immutable checkpoint appears, so an early 404 is not evidence.
        checkpoint_deadline = time.monotonic() + WATCHDOG_CHECKPOINT_GRACE_SECONDS
        checkpoint_hash = None
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
                        timeout=3,
                    )
                    candidate = marker.read_text().split()[0]
                    cli.call(
                        "download",
                        "-s",
                        cli.session,
                        context["root"] + "/output/lease-checkpoint.pkl",
                        str(local),
                        timeout=3,
                    )
                    if supervisor.sha256(local) != candidate:
                        raise supervisor.Refused("watchdog checkpoint hash mismatch")
                    checkpoint_hash = candidate
                    break
                except supervisor.Refused:
                    time.sleep(1)
            if checkpoint_hash is None:
                raise supervisor.Refused("watchdog checkpoint was not exported during grace")
            destination = context["output"] / "watchdog-checkpoint.pkl"
            local.replace(destination)
            marker.replace(destination.with_suffix(".pkl.sha256"))
        observed_at = None
        release_wait_started = lease_expiry
        deadline = lease_expiry + WATCHDOG_DISAPPEARANCE_BOUND_SECONDS
        last_sessions: dict[str, str] = {}
        while time.monotonic() < deadline:
            # Read-only account polling is deliberately permitted after the
            # simulated Mac loss; it does not renew the lease or queue kernel
            # work.  A release only counts when this exact backend endpoint
            # vanishes: a `?` session label can otherwise hide a paid orphan.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            last_sessions = cli.sessions(timeout=min(20.0, remaining))
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
        self._record(
            "mac-loss-watchdog-unassign",
            lease_seconds=lease,
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
        path.write_text(json.dumps({"gates": self.records}, indent=2, sort_keys=True) + "\n")
