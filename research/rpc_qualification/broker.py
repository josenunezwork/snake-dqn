"""Narrow C1 injection runner around the already-qualified Colab supervisor.

This module intentionally does not change ``research.colab_cli``.  The normal
six sessions use the unmodified supervisor.  The separate CPU injection
session installs a temporary, instance-local post-send response-loss adapter
at the supervisor's ``ready`` hook, then immediately restores it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from research.colab_cli import cli_guard, supervisor
from research.colab_cli.failures import ClassifiedFailure
from research.colab_cli.transfers import download_file


def _write_once(path: Path, value: dict[str, Any]) -> None:
    supervisor.write_once(path, value)


@contextlib.contextmanager
def _exact_profile_canary_adapter() -> Any:
    """Permit only the separately-reviewed C1 script past family-canary admission."""
    from research.colab_cli import canary
    from research.rpc_qualification.admission import SCRIPT, validate_manifest

    original = canary.check

    def checked(manifest: dict[str, Any], repo: Path) -> None:
        if manifest.get("script") != SCRIPT:
            original(manifest, repo)
            return
        validate_manifest(manifest, repo)

    canary.check = checked
    try:
        yield
    finally:
        canary.check = original


def validate(manifest: dict[str, Any], repo: Path | None = None) -> dict[str, Any]:
    """Apply ordinary broker validation plus the exact C1 profile fence."""
    with _exact_profile_canary_adapter():
        return supervisor.validate(manifest, repo)


def _remote_ready_code(root: str) -> str:
    """Read a published readiness triplet; it never waits in the kernel."""
    return (
        "import hashlib, json\nfrom pathlib import Path\n"
        f"root=Path({root!r})\n"
        "p=root/'output'/'injection-ready.json'\n"
        "s=p.with_name(p.name+'.sha256'); c=p.with_name(p.name+'.complete')\n"
        "if not (p.is_file() and s.is_file() and c.is_file() and c.read_text()=='complete\\n'):\n"
        "  result={'ready':False}\n"
        "else:\n"
        "  d=hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "  assert s.read_text()==f'{d}  {p.name}\\n'\n"
        "  result={'ready':True, 'value':json.loads(p.read_text())}\n"
    )


def _remote_evidence_code(root: str, descriptor: str, session: str, code_sha256: str) -> str:
    """Collect marker/result identity and hashes without exposing label outcomes."""
    return (
        "import hashlib, json\nfrom pathlib import Path\n"
        f"descriptor=Path({descriptor!r}); spec=json.loads(descriptor.read_text())\n"
        "execution=Path(spec['execution_dir']); index=int(spec['index'])\n"
        f"rpc=Path('/content/.snake-rpc-'+hashlib.sha256({session!r}.encode()).hexdigest()[:32])\n"
        "paths={'started':rpc/f\"{spec['request_id']}.started\","
        "'done':rpc/f\"{spec['request_id']}.done\","
        "'execution_started':execution/f'unit-{index:08d}.started',"
        "'execution_done':execution/f'unit-{index:08d}.done','result':Path(spec['unit_path'])}\n"
        "def meta(path):\n"
        "  assert path.is_file()\n"
        "  return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),"
        "'size':path.stat().st_size}\n"
        "payload=json.loads(paths['result'].read_text())\n"
        "assert payload.get('unit_id')==spec['unit_id'] and payload.get('execution_count')==1\n"
        "started=json.loads(paths['started'].read_text()); "
        "done=json.loads(paths['done'].read_text())\n"
        f"assert started=={{'request_id':spec['request_id'],'sha256':{code_sha256!r}}}\n"
        "assert done.get('request_id')==spec['request_id'] "
        "and done.get('sha256')==started['sha256'] and done.get('ok') is True\n"
        "execution_started=json.loads(paths['execution_started'].read_text()); "
        "execution_done=json.loads(paths['execution_done'].read_text())\n"
        "assert execution_started==execution_done "
        "and execution_started.get('request_id')==spec['request_id']\n"
        "result={'unit_id':spec['unit_id'],'request_id':spec['request_id'],"
        "'execution_count':1, **{key:meta(value) for key,value in paths.items()}}\n"
    )


def _remote_publish_proof_code(root: str, proof: dict[str, Any]) -> str:
    """Create the runner's completion triplet; completion is always published last."""
    return (
        "import hashlib, json, os, tempfile\nfrom pathlib import Path\n"
        f"root=Path({root!r}); value={proof!r}\n"
        "path=root/'output'/'injection-proof.json'; "
        "data=(json.dumps(value,sort_keys=True)+ '\\n').encode()\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "def create(target, contents):\n"
        "  fd, name=tempfile.mkstemp(dir=target.parent, prefix='.injection-')\n"
        "  try:\n"
        "    with os.fdopen(fd,'wb') as out: "
        "out.write(contents); out.flush(); os.fsync(out.fileno())\n"
        "    os.link(name,target)\n"
        "  finally:\n"
        "    Path(name).unlink(missing_ok=True)\n"
        "create(path,data)\n"
        "digest=hashlib.sha256(data).hexdigest()\n"
        "create(path.with_name(path.name+'.sha256'), f'{digest}  {path.name}\\n'.encode())\n"
        "create(path.with_name(path.name+'.complete'), b'complete\\n')\n"
        "result={'sha256':digest,'size':len(data)}\n"
    )


def _remote_unit_code(root: str, descriptor: str, request_id: str) -> str:
    """Execute exactly one frozen descriptor; output is operational evidence only."""
    return (
        "import json, os, subprocess\nfrom pathlib import Path\n"
        f"root=Path({root!r})\ndescriptor={descriptor!r}\n"
        "env=dict(os.environ, PYTHONPATH=str(root/'frozen')+os.pathsep+str(root/'repo'))\n"
        "py=str(root/'.venv'/'bin'/'python')\n"
        "subprocess.run([py,'-m','research.rpc_qualification.loop','unit',"
        f"'--descriptor',descriptor,'--request-id',{request_id!r}],"
        "env=env,cwd=root/'frozen',check=True)\n"
        "result=json.loads(Path(descriptor).read_text())\n"
    )


class InjectionObserver:
    """Runs the intentionally lossy, same-request-ID injection proof once."""

    def __init__(self, output: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.output = output
        self.clock = clock
        self.ran = False

    @staticmethod
    def _tag_injected_failure(cli: supervisor.CLI, unit_id: str, request_id: str) -> None:
        for record in reversed(cli.call_records):
            if record.get("command") == "exec" and record.get("request_id") == request_id:
                record.update(
                    injected=True,
                    injection_kind="post_send_response_loss",
                    unit_id=unit_id,
                    request_id=request_id,
                )
                return
        raise supervisor.Refused("injected response loss lacks its raw exec call record")

    @staticmethod
    def _idle_before_same_id_retry(cli: supervisor.CLI) -> None:
        timeout = cli.remaining_seconds(30)
        if not cli_guard.kernel_idle(cli.oauth_python, cli.config, cli.session, timeout=timeout):
            raise supervisor.Refused("kernel did not independently prove idle before same-ID retry")
        cli.exec_uncertain = False

    @staticmethod
    def _arm_post_send_loss(
        cli: supervisor.CLI, request_id: str
    ) -> tuple[Callable[[], None], dict[str, bool]]:
        """Raise only after a successful exact request has been sent once."""
        original = cli._call_once
        state = {"post_send_loss": False}

        def wrapped(*args: str, timeout: float) -> str:
            response = original(*args, timeout=timeout)
            if not state["post_send_loss"] and args and args[0] == "exec" and "-f" in args:
                code = Path(args[args.index("-f") + 1]).read_text()
                if f"# SNAKE_REQUEST_ID {request_id}" in code:
                    state["post_send_loss"] = True
                    raise subprocess.TimeoutExpired(args, timeout)
            return response

        cli._call_once = wrapped  # type: ignore[method-assign]

        def restore() -> None:
            cli._call_once = original  # type: ignore[method-assign]

        return restore, state

    def _rpc_with_loss(
        self,
        cli: supervisor.CLI,
        *,
        code: str | Callable[[str], str],
        unit_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_id = request_id or uuid.uuid4().hex
        code = code(request_id) if callable(code) else code
        restore, state = self._arm_post_send_loss(cli, request_id)
        try:
            try:
                cli.rpc(code, timeout=cli.remaining_seconds(330), request_id=request_id)
            except ClassifiedFailure:
                if not state["post_send_loss"]:
                    raise
                self._tag_injected_failure(cli, unit_id, request_id)
            else:
                raise supervisor.Refused(
                    "injection adapter did not drop the selected exec response"
                )
        finally:
            restore()
        self._idle_before_same_id_retry(cli)
        if cli.renew:
            cli.renew()
        recovered = cli.rpc(code, timeout=cli.remaining_seconds(330), request_id=request_id)
        # Reissuing a completed ID is an independent marker/idempotence proof.
        if cli.renew:
            cli.renew()
        completed_reissue = cli.rpc(code, timeout=cli.remaining_seconds(330), request_id=request_id)
        if recovered != completed_reissue:
            raise supervisor.Refused("completed request ID returned a different durable result")
        return {
            "unit_id": unit_id,
            "request_id": request_id,
            "code": code,
            "result": recovered,
        }

    def __call__(self, event: str, context: dict[str, Any]) -> None:
        if event != "ready" or self.ran:
            return
        manifest, cli, root = context["manifest"], context["cli"], context["root"]
        if manifest.get("env", {}).get("RQ_MODE") != "injection":
            return
        self.ran = True
        ready_deadline = self.clock() + cli.remaining_seconds(600)
        prepared: dict[str, Any] | None = None
        while self.clock() < ready_deadline:
            reply = cli.rpc(_remote_ready_code(root), timeout=cli.remaining_seconds(30))
            if isinstance(reply, dict) and reply.get("ready") is True:
                prepared = reply.get("value")
                break
            if cli.renew:
                cli.renew()
            time.sleep(2)
        descriptors = prepared.get("descriptors") if isinstance(prepared, dict) else None
        if not isinstance(descriptors, list) or len(descriptors) < 3:
            raise supervisor.Refused("injection preparation did not publish three descriptors")
        units = []
        for descriptor in descriptors[:3]:
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("unit_id"), str):
                raise supervisor.Refused("injection descriptor lacks unit identity")
            proof_unit = self._rpc_with_loss(
                cli,
                code=lambda request_id, path=str(descriptor["path"]): _remote_unit_code(
                    root, path, request_id
                ),
                unit_id=descriptor["unit_id"],
                request_id=(
                    descriptor["request_id"]
                    if isinstance(descriptor.get("request_id"), str)
                    else None
                ),
            )
            evidence = cli.rpc(
                _remote_evidence_code(
                    root,
                    str(descriptor["path"]),
                    cli.session,
                    hashlib.sha256(proof_unit["code"].encode()).hexdigest(),
                ),
                timeout=cli.remaining_seconds(30),
            )
            if (
                not isinstance(evidence, dict)
                or evidence.get("request_id") != proof_unit["request_id"]
            ):
                raise supervisor.Refused(
                    "injection unit marker evidence differs from request identity"
                )
            for kind in ("started", "done", "execution_started", "execution_done", "result"):
                metadata = evidence.get(kind)
                if not isinstance(metadata, dict):
                    raise supervisor.Refused(f"injection evidence lacks {kind} metadata")
                local = self.output / "injection-evidence" / proof_unit["request_id"] / kind
                download_file(
                    cli,
                    str(metadata["path"]),
                    local,
                    int(metadata["size"]),
                    cli.renew,
                    expected_sha256=str(metadata["sha256"]),
                )
                metadata["local_path"] = str(local)
            drop_index = next(
                (
                    index
                    for index, record in enumerate(cli.call_records)
                    if record.get("request_id") == proof_unit["request_id"]
                    and record.get("injected")
                ),
                None,
            )
            if drop_index is None:
                raise supervisor.Refused("injection proof lacks raw dropped-call index")
            units.append({**evidence, "drop_call_index": drop_index})
        proof = {
            "schema": "rpc-qualification-injection-proof/v1",
            "status": "COMPLETE",
            "mode": "injection",
            "injected_units": units,
            "injected_call_records": [r for r in cli.call_records if r.get("injected")],
            "completed_reissue": {"request_id": units[0]["request_id"], "same_result_sha256": True},
            "outcomes": "not read or exported",
        }
        if len(proof["injected_call_records"]) != 3:
            raise supervisor.Refused(
                "injection proof does not contain exactly three synthetic losses"
            )
        remote = cli.rpc(_remote_publish_proof_code(root, proof), timeout=cli.remaining_seconds(30))
        if not isinstance(remote, dict) or not isinstance(remote.get("sha256"), str):
            raise supervisor.Refused("remote injection completion triplet was not published")
        _write_once(self.output / "injection-proof.json", proof)


class QualificationBroker(supervisor.Broker):
    """Supervisor subclass that invokes the normal broker with one narrow observer."""

    def run(self, manifest: dict[str, Any], retained: dict[str, Any] | None = None) -> None:
        validated = validate(manifest)
        observer = InjectionObserver(Path(validated["output_dir"]))
        original = self.observer

        def observe(event: str, context: dict[str, Any]) -> None:
            if original:
                original(event, context)
            observer(event, context)

        self.observer = observe
        try:
            with _exact_profile_canary_adapter():
                super().run(validated, retained)
        finally:
            self.observer = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--colab", required=True)
    command = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command.add_parser(name).add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        if args.command == "validate":
            print(json.dumps(validate(manifest), indent=2, sort_keys=True))
        else:
            QualificationBroker(args.state_dir, args.colab).run(manifest)
    except (supervisor.Refused, ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
