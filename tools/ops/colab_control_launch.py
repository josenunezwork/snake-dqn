#!/usr/bin/env python3
"""Bind one qualified Colab-control root before execing its pinned launcher.

This is deliberately a tiny, exec-only wrapper.  It does not allocate, invoke
the provider CLI, change scheduling priority, or identify processes by text
search.  The dispatch caller has already checked the qualified source closure;
this wrapper records the actual root PID/birth/executable and refuses a
non-zero inherited nice before the broker launcher can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # This wrapper is launched as a script.  Its own repository supplies the
    # identity contract; it never imports control policy from the broker tree.
    sys.path.insert(0, str(ROOT))

from research.compute import colab_control  # noqa: E402
from research.compute.colab_control import POLICY_ID, RULING_SHA256  # noqa: E402
from research.compute.factory_companion_argv import native_argv  # noqa: E402

SCHEMA = "factory-colab-control-nice0-root/v1"


class Refused(ValueError):
    """The control root does not have the identity required by the ruling."""


def _nice(pid: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-o", "nice=", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    value = result.stdout.strip()
    if not re.fullmatch(r"-?[0-9]+", value):
        raise Refused("cannot read control root scheduling nice")
    return int(value)


def _identity(pid: int) -> dict[str, Any]:
    """Read the native PID-reuse identity already used by the factory guard."""
    try:
        from research.bl_labels.factory_persistence import _native_identity

        identity = _native_identity(pid)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise Refused("cannot read control root native identity") from exc
    birth, executable = identity.get("birth"), identity.get("executable")
    if not isinstance(birth, str) or not birth or not isinstance(executable, str) or not executable:
        raise Refused("control root native identity is incomplete")
    return identity


def _read_admission(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    required = (
        "policy_id",
        "ruling_sha256",
        "operational_pin",
        "qualified_broker_source_closure",
        "qualification_record",
        "qualification_record_sha256",
        "broker",
        "launcher",
        "module",
        "manifest",
        "manifest_sha256",
        "job_id",
        "state_path",
        "output_path",
        "overlap_status",
        "executable",
        "interpreter",
        "import_root",
        "cli_executable",
        "cli_interpreter",
        "cli_sha256",
        "oauth_interpreter",
        "mint_program_sha256",
    )
    if not isinstance(value, dict) or value.get("policy_id") != POLICY_ID:
        raise Refused("control admission policy is not admitted v1")
    if value.get("ruling_sha256") != RULING_SHA256:
        raise Refused("control admission ruling differs")
    if any(not isinstance(value.get(key), str) or not value[key] for key in required):
        raise Refused("control admission is incomplete")
    if not isinstance(value.get("source_closure"), dict) or not value["source_closure"]:
        raise Refused("control admission source closure is incomplete")
    if not isinstance(value.get("cli_package_closure"), dict) or not value["cli_package_closure"]:
        raise Refused("control admission CLI package closure is incomplete")
    if (
        not isinstance(value.get("dispatch_source_closure"), dict)
        or not value["dispatch_source_closure"]
    ):
        raise Refused("control admission dispatch closure is incomplete")
    if not isinstance(value.get("argv"), list) or not value["argv"]:
        raise Refused("control admission expected argv is incomplete")
    if value["module"] != "research.colab_cli.supervisor":
        raise Refused("control admission module is not the qualified supervisor")
    if value["overlap_status"] not in {
        "PREPARED_NOT_QUALIFIED_FOR_OVERLAP",
        "QUALIFIED_FOR_OVERLAP",
    }:
        raise Refused("control admission overlap state is not recognized")
    if not re.fullmatch(r"[0-9a-f]{64}", value["ruling_sha256"]):
        raise Refused("control admission ruling digest is malformed")
    if not re.fullmatch(r"[0-9a-f]{40}", value["operational_pin"]):
        raise Refused("control admission operational pin is malformed")
    if not re.fullmatch(r"[0-9a-f]{40}", value["qualified_broker_source_closure"]):
        raise Refused("control admission source closure is malformed")
    if not re.fullmatch(r"[0-9a-f]{64}", value["manifest_sha256"]):
        raise Refused("control admission manifest digest is malformed")
    return value


def bind_root(admission_path: Path, broker: Path, launcher: Path, argv: list[str]) -> Path:
    """Create the root record, binding only this exact future exec chain."""
    admission = _read_admission(admission_path)
    if broker.resolve() != Path(admission["broker"]).resolve():
        raise Refused("control broker differs from admitted broker")
    if launcher.resolve() != Path(admission["launcher"]).resolve():
        raise Refused("control launcher differs from admitted launcher")
    if not argv or Path(argv[0]).resolve() != launcher.resolve():
        raise Refused("control launcher argv differs from admitted launcher")
    if len(argv) < 2 or argv[1] != "--control-nice0":
        raise Refused("control launcher is missing its exact nice-0 mode")
    expected = [str(launcher), "--control-nice0", *admission["argv"][5:]]
    if argv != expected:
        raise Refused("control launcher arguments differ from the admitted exact chain")
    manifest = Path(admission["manifest"])
    if (
        not manifest.is_file()
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != admission["manifest_sha256"]
    ):
        raise Refused("control manifest bytes differ from admission")
    if _nice(os.getpid()) != 0:
        raise Refused("qualified Colab control root requires absolute nice 0")
    identity = _identity(os.getpid())
    bootstrap_argv = list(native_argv(os.getpid()))
    if bootstrap_argv != admission.get("bootstrap_argv"):
        raise Refused("bootstrap native argv differs from the admitted dispatch command")
    pane_pid = os.getppid()
    pane_identity = _identity(pane_pid)
    pane_argv = list(native_argv(pane_pid))
    pane_cwd = colab_control.native_cwd(pane_pid)
    if (
        Path(pane_identity["executable"]).resolve() != Path("/bin/bash").resolve()
        or pane_argv != admission.get("pane_argv")
        or Path(pane_cwd).resolve() != broker.resolve()
        or _nice(pane_pid) != 0
    ):
        raise Refused("persistent pane identity differs from the admitted dispatch chain")
    root = admission_path.with_suffix(".root.json")
    record = {
        **admission,
        "schema": SCHEMA,
        "admission_path": str(admission_path.resolve()),
        "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "pid": os.getpid(),
        "birth": identity["birth"],
        "birth_precision": identity.get("birth_precision"),
        "role": "wrapper",
        "bootstrap": {
            "pid": os.getpid(),
            "birth": identity["birth"],
            "executable": identity["executable"],
            "cwd": str(Path.cwd().resolve()),
            "argv": bootstrap_argv,
        },
        "pane": {
            "pid": pane_pid,
            "birth": pane_identity["birth"],
            "executable": "/bin/bash",
            "cwd": pane_cwd,
            "argv": pane_argv,
        },
    }
    parsed = colab_control._record(root, record)
    if not colab_control._source_matches(parsed, str(broker.resolve())):
        raise Refused("control admission sources no longer match the qualified closure")
    fd, temporary = tempfile.mkstemp(prefix=f".{root.name}.", dir=root.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, root)
        os.unlink(temporary)
    except FileExistsError as exc:
        raise Refused("create-only control root record already exists") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return root


def verify_launch(path: Path, pid: int, argv: list[str]) -> None:
    """Revalidate the exact Bash exec phase before the broker can allocate."""
    records = colab_control.load_admissions(path.parent)
    identity = _identity(pid)
    record = records.get((pid, identity["birth"]))
    if record is None or record.path != path.resolve():
        raise Refused("launch phase has no exact admitted PID/birth")
    colab_control.verify_witness(record)
    admission = json.loads(Path(record.admission_path).read_text())
    if (
        Path(identity["executable"]).resolve() != Path("/bin/bash").resolve()
        or list(native_argv(pid)) != ["/bin/bash", admission["launcher"], "--control-nice0", *argv]
        or tuple(argv) != record.argv[5:]
        or _nice(pid) != 0
        or colab_control.native_cwd(pid) != record.import_root
        or os.environ.get("SNAKE_DQN_PYTHON") != record.interpreter
        or os.environ.get("SNAKE_COLAB_OAUTH_PYTHON") != record.oauth_interpreter
        or not colab_control._source_matches(record, record.import_root)
    ):
        raise Refused("launch phase identity or source differs from admission")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-launch", type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if args.verify_launch is not None:
            if args.pid is None:
                raise Refused("launch verification requires the actual shell PID")
            verify_launch(args.verify_launch, args.pid, argv)
            return 0
        if args.admission is None or args.broker is None or args.launcher is None:
            raise Refused("root binding requires admission, broker and launcher")
        root_record = bind_root(args.admission, args.broker, args.launcher, argv)
        os.environ["SNAKE_COLAB_CONTROL_ROOT_RECORD"] = str(root_record.resolve())
        os.environ["SNAKE_COLAB_CONTROL_HELPER"] = str(Path(__file__).resolve())
        # Freeze the interpreter of the script phase as well as its arguments.
        os.execv("/bin/bash", ["/bin/bash", *argv])
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE: COLAB_CONTROL_NICE0 {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
