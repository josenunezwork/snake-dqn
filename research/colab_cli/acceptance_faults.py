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
        baseline = json.loads((context["output"] / "usage-allocated.json").read_text())
        usage = cli.usage()
        if baseline["balance"] - usage["balance"] >= context["manifest"]["budget"]["unit_cap"]:
            raise supervisor.Refused("acceptance observed debit cap reached")
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
                "COLAB_CLI_ASSIGN_RETRIES": "0",
                "COLAB_CLI_REQUEST_TIMEOUT": "30",
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
                transfer_started = time.time()
                entry = self._download_snapshot(context, probe)
                payload = output / "exports" / entry["path"]
                cli.call(
                    "upload", "-s", cli.session, str(payload), root + "/roundtrip.bin", timeout=40
                )
                cli.call(
                    "download",
                    "-s",
                    cli.session,
                    root + "/roundtrip.bin",
                    str(directory / "roundtrip.bin"),
                    timeout=20,
                )
                if supervisor.sha256(directory / "roundtrip.bin") != entry["sha256"]:
                    raise supervisor.Refused("full-size busy-kernel roundtrip hash mismatch")
                ack = directory / "transfer-done"
                ack.write_text("complete\n")
                cli.call(
                    "upload",
                    "-s",
                    cli.session,
                    str(ack),
                    root + "/acceptance-transfer-done",
                    timeout=10,
                )
                transfer_finished = time.time()
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
                    raise supervisor.Refused("kernel became idle before full snapshot roundtrip")
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
            kernel_started=started["started_at"],
            kernel_finished=finished["finished_at"],
            kernel_completion_marker_absent_after_full_roundtrip=True,
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
        # Suppress renewal longer than the lease. The runtime, not the Mac,
        # must checkpoint/attempt unassign. The next sessions call is evidence.
        self._state(context)
        time.sleep(lease + 1)
        # Reconnect only to the file transport during the checkpoint grace. It
        # does not renew the failed lease or submit a kernel execution.
        with tempfile.TemporaryDirectory(dir=context["output"], prefix=".watchdog-") as temp:
            marker = Path(temp) / "checkpoint.sha256"
            cli.call(
                "download",
                "-s",
                cli.session,
                context["root"] + "/output/lease-checkpoint.pkl.sha256",
                str(marker),
                timeout=3,
            )
            checkpoint_hash = marker.read_text().split()[0]
            local = Path(temp) / "checkpoint.pkl"
            cli.call(
                "download",
                "-s",
                cli.session,
                context["root"] + "/output/lease-checkpoint.pkl",
                str(local),
                timeout=3,
            )
            if supervisor.sha256(local) != checkpoint_hash:
                raise supervisor.Refused("watchdog checkpoint hash mismatch")
            destination = context["output"] / "watchdog-checkpoint.pkl"
            local.replace(destination)
            marker.replace(destination.with_suffix(".pkl.sha256"))
        time.sleep(20)
        sessions = cli.sessions(timeout=20)
        if cli.session in sessions.values():
            raise supervisor.Refused("watchdog lease expired but assignment remains")
        self._record(
            "mac-loss-watchdog-unassign",
            lease_seconds=lease,
            sessions_after=sessions,
            checkpoint_sha256=checkpoint_hash,
            renewal_suppressed=True,
        )

    def write(self, output: Path) -> None:
        path = output / "acceptance-observer.json"
        if path.exists():
            raise supervisor.Refused("acceptance observer receipt already exists")
        path.write_text(json.dumps({"gates": self.records}, indent=2, sort_keys=True) + "\n")
