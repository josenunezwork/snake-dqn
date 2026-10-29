#!/usr/bin/env python3
"""Dispatch a qualified supervisor with bounded account allocation exclusion."""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

if __package__:
    from .qualification import OPS, tree
else:
    from qualification import OPS, tree

HERE = Path(__file__).resolve().parent
_guard_spec = importlib.util.spec_from_file_location(
    "_colab_control_imports", Path(__file__).with_name("control_imports.py")
)
assert _guard_spec is not None and _guard_spec.loader is not None
_guard_module = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(_guard_module)
_guard_module.verify(HERE.parents[1])
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from research.compute.colab_control import POLICY_ID as CONTROL_POLICY_ID  # noqa: E402
from research.compute.colab_control import RULING_SHA256 as CONTROL_RULING_SHA256  # noqa: E402
from research.compute.gpu_preference import FALLBACK_SUFFIX, sibling_id, variants  # noqa: E402

EXACT_ERROR = "AccountingError('assignment endpoints changed during allocation')"
BLOCK = re.compile(r"^\s*(?:BLOCK|HOLD)\b|RUNTIME BLOCKED|do not run this|PREPARED_NOT_DISPATCHED")
NICE_FLOOR = 15
# Sandbox admission executes its sealed focused test suite before invoking the
# unchanged qualified supervisor.  Keep ordinary host subprocesses at 60s while
# giving only that offline admission step a bounded window large enough for the
# suite on a busy Mac.
COMMAND_TIMEOUT_SECONDS = 60.0
SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS = 180.0
SANDBOX_OUTPUT_ROOTS = {
    "sandbox-b-heads-v1": HERE.parents[1].parent / "snake-dqn-artifacts/a100-sandbox-b-20261026",
    "sandbox-b-heads-v2": HERE.parents[1].parent / "snake-dqn-artifacts/a100-sandbox-b2-20261026",
    "sandbox-b-heads-v3": HERE.parents[1].parent / "snake-dqn-artifacts/a100-sandbox-b3-20261026",
    "sandbox-a": HERE.parents[1].parent / "snake-dqn-artifacts/sandbox-a-resident",
    "sandbox-a-train01": HERE.parents[1].parent / "snake-dqn-artifacts/sandbox-a-resident",
}
P1_ATTEMPT2_RULING = "55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13"
CONTROL_INTERPRETER = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
CONTROL_OAUTH_INTERPRETER = Path("/Users/josenunez/.venvs/colab-cli/bin/python")
F5_PLAN_ADMISSION_SCHEMA = "bl-f5-plan-operational-admission/v1"
F5_TIMING_ADMISSION_SCHEMA = "bl-f5-timing-operational-admission/v1"
F5_TIMING_RECOVERY_SCHEMA = "bl-f5-timing-recovery/v1"
F5_TIMING_RECOVERY_RULING_SHA256 = (
    "4f10f264fda3c7190e07f486803fa02cc10645741fb95ef215b1d1b0cc123439"
)
F5_TIMING_RECOVERY_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "F5_DEVICE": "cuda:0",
    "F5_MODE": "qualification",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
F5_PLAN_COMMIT = "c989f6738f6aad143eb97bc47f6bd9e6687057ce"
# Named successor of held, unexecuted plan attempt02 after transport requalification.
# This published source preserves c989f673's F5 bytes and contains CLI tree 6e8c1982.
F5_PLAN_REQUAL_COMMIT = "75b1442768f37ce83188a2990c9ef0026ee806b5"
F5_PLAN_REQUAL_BINDINGS = {
    "transport_repair_ruling": "cf283cf1f4ebd0e6575275621594b2c96b5b59e7d8f0da640cd553db0291ab1b",
    "capture05_postpublication_verification": (
        "8a06712f6700420562016e048f7983b0c633eb7d5cbc291b79e91c65d0fcfa1b"
    ),
}
F5_TIMING_REQUAL_BINDINGS = {
    **F5_PLAN_REQUAL_BINDINGS,
    "capture05_qualification": "27439f187c0c80f5b3afbb0afd863cf6ac2761c34d5da1a34fd83065d1deaa43",
    "plan_successor_ruling": "11bf59dd4d6c7d585f71970a02769439d927aa5dd7c75373c9d250f27632f610",
}
F5_PLAN_RULINGS = {
    "acceptance": "ebef9f754d0ea4c143226332f53be8a2705ad5fb050291047c94a35944d8f25a",
    "capture_qualification": "8f376c53a2150db018c529b2d75796f1ed4bbd564b31630d082fb1012780539f",
    "parallel_clarification": "71c5b924c4b5ae9caa85870acf047aaea2a9c04bc78206dc6ee46381e1dc8a2e",
    "owner_ruling": "48df0ca878dea69a6e0608946f8770cd56e7d017d79890b9566ab31debba767d",
}
F5_PLAN_REQUIRED_EXPORTS = [
    "prepare/prepare-receipt.json",
    "prepare/f5-batch-plan.pt",
    "paired/assignment-environment.json",
    "paired/fit-receipt.json",
    "paired/fit-C.pth",
    "paired/fit-W.pth",
    "paired/f5-state-C.pt",
    "paired/f5-state-W.pt",
    "qualification/qualification.json",
]
F5_PLAN_INPUT_DESTINATIONS = {
    "selected-training.tar.xz",
    "selected-training.tar.xz.sha256",
    "selected-training.tar.xz.complete",
    "b8.pth",
    "timing-fixtures.json",
    "control.json",
    "verified-ledger.json",
}
F5_B8_URL = (
    "https://github.com/josenunezwork/snake-dqn/releases/download/"
    "s1-inputs-20261019/B8_s3_final.pth"
)
F5_B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
F5_SELECTED_TRAINING_SHA256 = "2c033750b0d25568b19ba81aec7bd0e9b0583bdd32a994a9a1984a409e45c46c"
F5_PLAN_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "F5_DEVICE": "cuda:0",
    "F5_MODE": "prepare-qualification",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "SNAKE_COLAB_CANARY": "1",
    "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
    "VECLIB_MAXIMUM_THREADS": "1",
}
F5_TIMING_ENV = {**F5_PLAN_ENV, "F5_MODE": "qualification"}
F5_TIMING_REQUIRED_EXPORTS = [
    "paired/assignment-environment.json",
    "paired/fit-receipt.json",
    "paired/fit-C.pth",
    "paired/fit-W.pth",
    "paired/f5-state-C.pt",
    "paired/f5-state-W.pt",
    "qualification/qualification.json",
]
F5_TIMING_INPUT_SHA256 = {
    "b8.pth": F5_B8_SHA256,
    "timing-fixtures.json": "737b2c61eabeb570456c8d7c287a5e615ac642a71d320db44925c58c252201aa",
    "champion_a5_freespace_20260621.pth": (
        "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
    ),
    "best_apex_fs.pth": "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195",
    "best_apex_stage1_fs.pth": ("0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d"),
    "best_apex_pre_fs.pth": "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
    "frp3_m3_s12_u60000_20261005.pth": (
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
    ),
}
F5_TIMING_INPUT_DESTINATIONS = set(F5_TIMING_INPUT_SHA256) | {
    "control.json",
    "verified-ledger.json",
}


class Refused(ValueError):
    """The wrapper cannot prove that the next operational step is allowed."""


def write_create_only_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a complete immutable record without a partial readable final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except FileExistsError as exc:
        raise Refused(f"create-only record already exists: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def persistent_shell(
    command_text: str, *, p1_window: dict | None = None, control_nice0: bool = False
) -> str:
    """Return a tmux command whose long-lived shell has an effective nice floor.

    Tmux first invokes its own command shell.  Its shell immediately execs this
    checked launcher, which in turn execs a floor-adjusted Bash before starting
    the supplied command.  This prevents a normal-priority wrapper from waiting
    around a low-priority supervisor or agent child.
    """
    if p1_window is not None:
        # The prospective P1 ruling overrides the ordinary nice-15 policy only
        # for this explicitly reviewed maintenance window. Never try to raise
        # an inherited low priority: refuse before starting the supervisor.
        registration = shlex.join(
            [
                sys.executable,
                "-m",
                "research.bl_ppo.p1_attempt2_window",
                "register-control",
                "--arm-path",
                p1_window["arm_path"],
                "--owner-token",
                p1_window["owner_token"],
                "--root-record",
                p1_window["root_record"],
                "--evidence",
                p1_window["registration_evidence"],
            ]
        )
        quiet = (
            "set -euo pipefail; "
            'test "$(/bin/ps -o nice= -p "$$" | tr -d " ")" = 0 || '
            '{ echo "HOLD: P1 control root requires absolute nice 0" >&2; exit 2; }; '
            f'{registration} --root-pid "$$"; set +e; {command_text}'
        )
        return "exec " + shlex.join(["/bin/bash", "-c", quiet, "p1_attempt2_control"])
    if control_nice0:
        # ``nice -n 0`` preserves the inherited value, and a non-root child cannot
        # reliably raise it.  Refuse before the launcher can reach allocation.
        checked = (
            "set -euo pipefail; "
            'test "$(/bin/ps -o nice= -p "$$" | tr -d " ")" = 0 || '
            '{ echo "REFUSE: Colab control requires absolute nice 0" >&2; exit 2; }; '
            f"set +e; {command_text}"
        )
        return "exec " + shlex.join(["/bin/bash", "-c", checked, "colab-control-nice0"])
    floor = f"""current_nice=$(/bin/ps -o nice= -p \"$$\")
if ! [[ \"$current_nice\" =~ ^[[:space:]]*-?[0-9]+[[:space:]]*$ ]]; then
  echo \"REFUSE: cannot read persistent shell scheduling nice value\" >&2
  exit 2
fi
if [ \"$current_nice\" -lt {NICE_FLOOR} ]; then
  exec /usr/bin/nice -n \"$(({NICE_FLOOR} - current_nice))\" /bin/bash -c \"$1\"
fi
exec /bin/bash -c \"$1\""""
    return "exec " + shlex.join(["/bin/bash", "-c", floor, "persistent-shell", command_text])


def command(
    args: list[str],
    *,
    deadline: float | None = None,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Bound every subprocess while the account lock is held."""
    remaining = (
        timeout_seconds if deadline is None else min(timeout_seconds, deadline - time.monotonic())
    )
    if remaining <= 0:
        raise TimeoutError("allocation lock deadline expired")
    return subprocess.run(args, timeout=remaining, check=True, **kwargs)


def note(log: Path, value: str) -> None:
    """Append an ops observation without replacing supervisor evidence."""
    with log.open("a") as stream:
        stream.write(value + "\n")


@contextlib.contextmanager
def allocation_lock(path: Path, seconds: float) -> Iterator[float]:
    """Own the flock here; termination, including SIGKILL, closes the descriptor."""
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused(f"account allocation lock is held: {path}") from exc
        try:
            yield time.monotonic() + seconds
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def qualified(broker: Path, ops: Path, deadline: float | None = None, ref: str = "HEAD") -> None:
    """Check actual broker bytes both before validation and under allocation lock."""
    command(
        [
            sys.executable,
            str(HERE / "qualification.py"),
            "check",
            "--repo",
            str(broker),
            "--record",
            str(ops / "colab-cli-qualified.json"),
            "--ref",
            ref,
        ],
        deadline=deadline,
        stdout=subprocess.DEVNULL,
    )


def create_control_admission(
    *,
    ops: Path,
    name: str,
    broker: Path,
    commit: str,
    manifest: Path,
    state: Path,
    output: str,
    pane_argv: list[str],
    bootstrap_argv: list[str],
) -> Path:
    """Create one immutable admission for the exact refreshed allocation input."""
    value = read_json(manifest)
    payload = manifest.read_bytes()
    path = ops / "colab-control-admissions" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    closure = {
        str(candidate.relative_to(broker)): hashlib.sha256(candidate.read_bytes()).hexdigest()
        for candidate in sorted((broker / "research" / "colab_cli").rglob("*"))
        if candidate.is_file() and "__pycache__" not in candidate.parts
    }
    cli = Path.home() / ".local/bin/colab"
    if not cli.is_file():
        raise Refused("qualified Colab CLI executable is missing")
    first_line = cli.read_text(encoding="utf-8", errors="strict").splitlines()[0]
    if not first_line.startswith("#!") or not Path(first_line[2:]).is_absolute():
        raise Refused("qualified Colab CLI has no absolute shebang interpreter")
    cli_interpreter = Path(first_line[2:])
    if not cli_interpreter.is_file():
        raise Refused("qualified Colab CLI shebang interpreter is missing")
    package_root = subprocess.check_output(
        [
            str(cli_interpreter),
            "-I",
            "-c",
            "import colab_cli; print(next(iter(colab_cli.__path__)))",
        ],
        text=True,
        timeout=10,
    ).strip()
    package = Path(package_root)
    if not package.is_dir():
        raise Refused("qualified Colab CLI package root is missing")
    cli_package_closure = {
        str(candidate): hashlib.sha256(candidate.read_bytes()).hexdigest()
        for candidate in sorted(package.rglob("*"))
        if candidate.is_file() and "__pycache__" not in candidate.parts
    }
    if not cli_package_closure:
        raise Refused("qualified Colab CLI package closure is empty")
    # The broker's release helper is imported only to bind its fixed local
    # credential-mint payload; this function never calls the provider.
    from research.compute import colab_control
    from research.compute.colab_control import release_identity

    mint_program, _ = release_identity(broker)

    interpreter = CONTROL_INTERPRETER
    if not interpreter.is_file() or not CONTROL_OAUTH_INTERPRETER.is_file():
        raise Refused("qualified Colab control interpreter is missing")
    qualification_snapshot = path.with_name(f"{name}.qualification.json")
    qualification_value = read_json(ops / "colab-cli-qualified.json")
    if qualification_value.get("tree_hash") != tree(broker):
        raise Refused("qualification registry changed before control admission")
    write_create_only_json(qualification_snapshot, qualification_value)
    git_executable = Path(
        subprocess.check_output(["/usr/bin/xcrun", "--find", "git"], text=True, timeout=5).strip()
    ).resolve()
    if not git_executable.is_file():
        raise Refused("qualified source Git executable is missing")
    record = {
        "interpreter_native": colab_control.interpreter_native(CONTROL_INTERPRETER),
        "cli_interpreter_native": colab_control.interpreter_native(cli_interpreter),
        "oauth_interpreter_native": colab_control.interpreter_native(CONTROL_OAUTH_INTERPRETER),
        "git_executable": str(git_executable),
        "git_sha256": hashlib.sha256(git_executable.read_bytes()).hexdigest(),
        "schema": "factory-colab-control-nice0-admission/v1",
        "policy_id": CONTROL_POLICY_ID,
        "ruling_sha256": CONTROL_RULING_SHA256,
        "operational_pin": subprocess.check_output(
            ["git", "-C", str(HERE.parents[1]), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip(),
        "manifest_commit": commit,
        "operational_root": str(HERE.parents[1]),
        "pane_argv": pane_argv,
        "bootstrap_argv": bootstrap_argv,
        "qualified_broker_source_closure": tree(broker),
        "qualification_record": str(qualification_snapshot.resolve()),
        "qualification_record_sha256": hashlib.sha256(
            qualification_snapshot.read_bytes()
        ).hexdigest(),
        "dispatch_source_closure": {
            str(candidate.resolve()): hashlib.sha256(candidate.read_bytes()).hexdigest()
            for candidate in (
                HERE / "dispatch.py",
                HERE / "dispatch.sh",
                HERE / "colab_control_launch.py",
                HERE / "control_imports.py",
                HERE.parents[1] / "research/bl_labels/__init__.py",
                HERE.parents[1] / "research/bl_labels/exports.py",
                HERE / "control_overlap.py",
                HERE.parents[1] / "research/compute/colab_control.py",
                HERE.parents[1] / "research/compute/factory_companion_argv.py",
                HERE.parents[1] / "research/compute/factory_companion_native.py",
                HERE.parents[1] / "research/bl_labels/factory_persistence.py",
            )
        },
        "source_closure": closure,
        "broker": str(broker),
        "launcher": str(broker / "research/colab_cli/launch.sh"),
        "module": "research.colab_cli.supervisor",
        # The registered PID execs into this exact caffeinate wrapper.  Its
        # native birth identity is retained across exec and checked at sample.
        "executable": str(Path("/usr/bin/caffeinate").resolve()),
        "interpreter": str(interpreter),
        "import_root": str(broker),
        "argv": [
            "/usr/bin/caffeinate",
            "-is",
            str(interpreter),
            "-m",
            "research.colab_cli.supervisor",
            "--state-dir",
            str(state),
            "--colab",
            str(cli),
            "run",
            str(manifest),
        ],
        "cli_executable": str(cli),
        "cli_interpreter": str(cli_interpreter),
        "cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "cli_package_closure": cli_package_closure,
        "oauth_interpreter": str(CONTROL_OAUTH_INTERPRETER),
        "mint_program_sha256": hashlib.sha256(mint_program.encode()).hexdigest(),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "job_id": value["job_id"],
        "state_path": str(state),
        "output_path": output,
        "overlap_status": value["colab_control_nice0"]["overlap_status"],
    }
    write_create_only_json(path, record)
    return path


def require_overlap_canary(
    value: dict[str, Any], original_bytes: bytes, *, broker: Path | None = None
) -> None:
    """Allow the prepared policy only for its already-authorized exact canary.

    A normal paid main cannot turn a self-declared ``PREPARED`` state into an
    overlap admission.  This wrapper verifies only mechanical source binding;
    the qualified supervisor retains its independent science/budget admission.
    """
    control = value["colab_control_nice0"]
    if control.get("overlap_status") == "QUALIFIED_FOR_OVERLAP":
        from tools.ops.control_overlap import require_proof

        if broker is None or not isinstance(control.get("overlap_receipt"), dict):
            raise Refused("qualified overlap requires its immutable independent receipt")
        require_proof(control["overlap_receipt"], operational_root=HERE.parents[1], broker=broker)
        return
    if control.get("overlap_status") != "PREPARED_NOT_QUALIFIED_FOR_OVERLAP":
        raise Refused("control overlap proof is absent or not in the prepared state")
    canary = control.get("authorized_overlap_canary")
    if not isinstance(canary, dict):
        raise Refused("prepared Colab control admits only an authorized overlap canary")
    authority = canary.get("authority_path")
    authority_sha = canary.get("authority_sha256")
    if not isinstance(authority, str) or not Path(authority).is_absolute():
        raise Refused("overlap canary authority path is not absolute")
    source = Path(authority)
    if not source.is_file() or not isinstance(authority_sha, str):
        raise Refused("overlap canary authority evidence is unavailable")
    if hashlib.sha256(source.read_bytes()).hexdigest() != authority_sha:
        raise Refused("overlap canary authority evidence digest differs")
    # The field which stores this digest cannot be included in its own hash.
    # Bind all other parsed manifest bytes through a deterministic canonical
    # representation, with only that one self-reference removed.
    normalized = json.loads(json.dumps(value))
    normalized["colab_control_nice0"]["authorized_overlap_canary"].pop("manifest_sha256", None)
    manifest_identity = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if canary.get("manifest_sha256") != manifest_identity:
        raise Refused("overlap canary manifest is not source-bound")
    if canary.get("quiet_host_required") is not False:
        raise Refused("quiet-host study cannot use the overlap control exception")


def require_no_control_priority_drift() -> None:
    """Refuse a new paid admission if an existing admitted tree has drifted."""
    from research.bl_labels.factory_persistence import PersistenceTracker

    def read(command: list[str]) -> str:
        return subprocess.check_output(command, text=True, errors="surrogateescape", timeout=10)

    tracker = PersistenceTracker("colab-control-paid-admission")
    tracker.sample(read, set())
    rows = tracker.known.values()
    drifted = [row["pid"] for row in rows if row.get("control_identity") and row.get("nice") != 0]
    if drifted:
        raise Refused(f"qualified Colab control priority drift: {drifted}")


def supervisor_launcher(broker: Path, manifest: dict[str, Any]) -> Path:
    """Use the explicit host admission adapter only for declared sandbox jobs."""
    collector = sandbox_collector_directory(manifest)
    if "sandbox_admission" in manifest:
        if broker.resolve() != HERE.parents[1]:
            raise Refused("sandbox adapter must run from the selected broker checkout")
        return broker / (
            "tools/sandbox/launch_v2.sh" if collector is not None else "tools/sandbox/launch.sh"
        )
    return broker / "research/colab_cli/launch.sh"


def sandbox_collector_directory(manifest: dict[str, Any]) -> Path | None:
    """Resolve an explicit host-only opt-in, never a remote environment control.

    Derive a sibling of the fresh output directory so allocation retries get
    distinct journals automatically and collection cannot pre-create output_dir.
    The original packet/preflight digest binds this field like every other byte.
    """
    if "sandbox_supervision" not in manifest:
        return None
    if (
        "sandbox_admission" not in manifest
        or manifest["sandbox_supervision"] != "collector-v2"
        or not isinstance(manifest.get("output_dir"), str)
        or not Path(manifest["output_dir"]).is_absolute()
    ):
        raise Refused("sandbox_supervision requires sandbox admission and collector-v2")
    raw_output = manifest["output_dir"]
    output = Path(raw_output)
    if not output.name or str(output.resolve()) != raw_output:
        raise Refused("collector output must be a canonical absolute path")
    return output.with_name(output.name + ".supervision")


def require_sandbox_admission(manifest: dict[str, Any], broker: Path) -> None:
    """Substitute adopted sandbox preflight for a family timing canary only.

    The ten-line committed manifest binds the original packet by SHA. The
    qualified supervisor may add its documented defaults and dispatch may refresh
    account prices or make its one exact accounting retry; nothing else varies.
    This function never allocates, reads registered outcomes, or edits transport.
    """
    from tools.sandbox import preflight
    from tools.sandbox.manifest import _forbidden

    if broker.resolve() != HERE.parents[1]:
        raise Refused("sandbox admission broker differs from adapter root")
    admission = manifest.get("sandbox_admission")
    if (
        not isinstance(admission, dict)
        or set(admission) != {"schema", "manifest_path"}
        or admission["schema"] != "standing-a100-development-sandbox-admission/v1"
        or not isinstance(admission["manifest_path"], str)
        or not Path(admission["manifest_path"]).is_absolute()
    ):
        raise Refused("sandbox requires an explicit ten-line manifest admission")
    allowed_keys = {
        "job_id",
        "commit",
        "script",
        "device",
        "source_paths",
        "requirements",
        "timeout_minutes",
        "output_dir",
        "max_export_bytes",
        "lease_seconds",
        "inputs",
        "env",
        "required_exports",
        "budget",
        "resume",
        "sandbox_admission",
        "sandbox_execution",
        "requirements_file",
        "technical_envelope",
        "sandbox",
        "sandbox_a",
        "sandbox_source_delivery",
        "completion_attempt",
        "sandbox_supervision",
    }
    if set(manifest) - allowed_keys:
        raise Refused("sandbox forbids canary, registered-study or unknown controls")
    sandbox_collector_directory(manifest)
    packages = {
        "research/sandbox_b_heads/run_job.sh": "sandbox-b-heads-v1",
        "research/sandbox_b_heads/run_b2.sh": "sandbox-b-heads-v2",
        "research/sandbox_b_heads/run_b3.sh": "sandbox-b-heads-v3",
        "research/sandbox_a_resident/job.sh": "sandbox-a",
        "research/sandbox_a_resident/train_job.sh": "sandbox-a-train01",
    }
    study = packages.get(manifest.get("script"))
    if study is None:
        raise Refused("sandbox script is outside adopted development packages")
    if "completion_attempt" in manifest and study != "sandbox-a-train01":
        raise Refused("training completion metadata is exclusive to TRAIN-01")
    output = preflight._assert_safe_path(manifest.get("output_dir", ""), "sandbox output")
    if not output.is_absolute() or not output.resolve().is_relative_to(SANDBOX_OUTPUT_ROOTS[study]):
        raise Refused("sandbox output must stay in its isolated development artifact root")
    _forbidden(
        {
            key: manifest.get(key)
            for key in ("script", "env", "source_paths", "output_dir", "required_exports", "inputs")
        }
    )
    for item in manifest.get("inputs", []):
        preflight._assert_safe_path(item.get("path", ""), "sandbox input")
    env = manifest.get("env", {})
    if any(key.startswith("SNAKE_COLAB_CANARY") for key in env):
        raise Refused("sandbox cannot enable canary instrumentation")
    allowed_env = {"SNAKE_DQN_DEVICE", "COLAB_CLI_REQUIREMENTS"}
    if study in {"sandbox-b-heads-v1", "sandbox-b-heads-v2", "sandbox-b-heads-v3"}:
        allowed_env.add("SANDBOX_B_SALT")
        prefixes = ("src/", "tools/sandbox/", "research/sandbox_b_heads/")
    else:
        allowed_env.update({"SANDBOX_A_CONTROL_JSON", "SANDBOX_A_SOURCE_COMMIT", "MAX_JOBS"})
        prefixes = (
            "src/",
            "tools/sandbox/",
            "research/sandbox_a_resident/",
            "research/gpu_sim_spike/",
        )
    if study == "sandbox-a-train01":
        allowed_env.add("SANDBOX_A_STALL_TIMEOUT_SECONDS")
        stall = env.get("SANDBOX_A_STALL_TIMEOUT_SECONDS", "900")
        try:
            seconds = float(stall) if isinstance(stall, str) else math.nan
        except ValueError:
            seconds = math.nan
        if not math.isfinite(seconds) or not 0 < seconds <= 900:
            raise Refused("TRAIN-01 stall timeout must be finite, positive and at most 900 seconds")
    elif "sandbox_source_delivery" in manifest:
        raise Refused("background source delivery is exclusive to TRAIN-01")
    delivery = manifest.get("sandbox_source_delivery")
    if "sandbox_source_delivery" in manifest and delivery != "background-sparse-v1":
        raise Refused("sandbox source delivery must select background-sparse-v1")
    if set(env) - allowed_env:
        raise Refused("sandbox environment contains non-package controls")
    shared = {"research/a100_a2/model.py", "research/bl_residual/data.py"}
    for source in manifest.get("source_paths", []):
        parts = Path(source).parts
        if (
            Path(source).is_absolute()
            or ".." in parts
            or (source not in shared and not source.startswith(prefixes))
        ):
            raise Refused("sandbox source is outside its engineering dependency closure")
    req = manifest.get("requirements", {})
    if req.get("gpu") not in {"H100", "A100"} or req.get("allowed_gpus", [req["gpu"]]) != [
        req["gpu"]
    ]:
        raise Refused("sandbox permits only an explicit H100 or A100 assignment")
    path = preflight._assert_safe_path(admission["manifest_path"], "sandbox manifest")
    payload = preflight.preflight(path, run_tests=True)
    if payload["study"] != study:
        raise Refused("sandbox study differs from entrypoint")
    original = read_json(Path(payload["outputs"]["broker_manifest"]["path"]))

    def normalized(value: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(value))
        req = result["requirements"]
        req.setdefault("allowed_gpus", [req["gpu"]])
        result["budget"].setdefault("reserve_units", 0)
        result["budget"]["other_assignments"] = {}  # Fresh broker prices only.
        result.setdefault("env", {})
        result.setdefault("lease_seconds", 180)
        result.setdefault("max_export_bytes", 1024**3)
        result["output_dir"] = str(Path(result["output_dir"]).resolve())
        for item in result.get("inputs", []):
            item["path"] = str(Path(item["path"]).resolve())
        return result

    expected, current = normalized(original), normalized(manifest)
    if original.get("requirements", {}).get("prefer") is not None:
        expected = next(
            (
                normalized(item)
                for item in variants(original)
                if item["job_id"] == current["job_id"]
            ),
            expected,
        )
    retry_id = original["job_id"] + "-retry-1"
    if len(retry_id) > 64:
        digest = hashlib.sha256(original["job_id"].encode()).hexdigest()[:8]
        retry_id = f'{original["job_id"][:47]}-{digest}-retry-1'
    if current["job_id"] == retry_id:
        expected.update(job_id=retry_id, output_dir=expected["output_dir"] + "-retry-1")
    if current != expected:
        raise Refused("sandbox runtime packet differs from committed preflight binding")
    # Retain all non-receipt checks that ordinary canary.check performs.
    from research.colab_cli import canary, transfer_policy

    transfer_policy.transfer_limit(manifest)
    canary.job_device(manifest, broker)
    canary._exports(manifest)
    if study in {"sandbox-a", "sandbox-a-train01"}:
        # Job A owns its control/seed recipe checker and focused tests.
        module = (
            "research.sandbox_a_resident.training_preflight"
            if study == "sandbox-a-train01"
            else "research.sandbox_a_resident.preflight"
        )
        command(
            [sys.executable, "-m", module, str(path.parent)],
            cwd=broker,
        )


def validate(
    broker: Path, cwd: Path, state: Path, manifest: Path, log: Path, deadline: float | None = None
) -> None:
    """Run real validation; launch.sh --check would only print and return."""
    packet = read_json(manifest)
    colab = Path.home() / ".local/bin/colab"
    if sandbox_collector_directory(packet) is not None:
        # The supervision registry binds exact argv AND canonical CLI source.
        # Keep the venv shebang script, but resolve the ~/.local/bin alias.
        colab = colab.resolve()
    timeout_seconds = (
        SANDBOX_ADMISSION_VALIDATE_TIMEOUT_SECONDS
        if "sandbox_admission" in packet
        else COMMAND_TIMEOUT_SECONDS
    )
    with log.open("x") as stream:
        command(
            [
                str(supervisor_launcher(broker, packet)),
                "--state-dir",
                str(state),
                "--colab",
                str(colab),
                "validate",
                str(manifest),
            ],
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
        )


def register(ops: Path, log: Path, name: str) -> None:
    """Register each attempt separately, including its eventual EXIT marker."""
    with (ops / "watch-logs.txt").open("a") as stream:
        stream.write(f"{log} {name}\n")


def start(
    broker: Path,
    cwd: Path,
    state: Path,
    manifest: Path,
    log: Path,
    name: str,
    deadline: float,
    *,
    control_nice0: bool = False,
    control_admission: Path | None = None,
    small_bench_config: Path | None = None,
    gpu_receipt: Path | None = None,
) -> None:
    """Submit one shell command with preserved argument boundaries and exit evidence."""
    packet = read_json(manifest)
    collector = sandbox_collector_directory(packet)
    if collector is not None and (collector.exists() or collector.is_symlink()):
        raise Refused("collector evidence path already exists")
    if collector is not None and (control_nice0 or small_bench_config is not None):
        raise Refused("collector-v2 cannot be combined with another host launch adapter")
    colab = Path.home() / ".local/bin/colab"
    if collector is not None:
        colab = colab.resolve()
    log.touch(exist_ok=False)
    register(log.parent.parent, log, name)
    launcher = [
        str(supervisor_launcher(broker, packet)),
        "--state-dir",
        str(state),
        "--colab",
        str(colab),
        "run",
        str(manifest),
    ]
    if small_bench_config is not None:
        # A host-only CLI front door checks capture age at the actual `new`
        # request. The qualified broker/runtime and original manifest stay intact.
        guard = HERE.parent / "small_bench_cli.py"
        launcher[launcher.index("--colab") + 1] = str(guard)
    if control_nice0:
        if control_admission is None:
            raise Refused("missing create-only Colab control admission")
        launcher.insert(1, "--control-nice0")
        launcher = [
            str(CONTROL_INTERPRETER),
            str(HERE / "colab_control_launch.py"),
            "--admission",
            str(control_admission),
            "--broker",
            str(broker),
            "--launcher",
            str(broker / "research/colab_cli/launch.sh"),
            "--",
            *launcher,
        ]
    environment_items = ["env", f"HOME={Path.home()}", f'PATH={os.environ["PATH"]}']
    if collector is not None:
        environment_items.append(f"SNAKE_SANDBOX_COLLECTOR_DIR={collector}")
    if small_bench_config is not None:
        environment_items.append(f"SNAKE_SMALL_BENCH_CONFIG={small_bench_config}")
    if control_nice0:
        environment_items = [
            "env",
            "-u",
            "PYTHONPATH",
            "-u",
            "PYTHONHOME",
            "-u",
            "PYTHONSTARTUP",
            "-u",
            "PYTHONINSPECT",
            "-u",
            "PYTHONSAFEPATH",
            "PYTHONNOUSERSITE=1",
            f"HOME={Path.home()}",
            f'PATH={os.environ["PATH"]}',
        ]
        # Bind launch.sh's interpreter choice to the one recorded in the
        # admission instead of inheriting an arbitrary shell override.
        environment_items.extend(
            [
                f"SNAKE_DQN_PYTHON={CONTROL_INTERPRETER}",
                f"SNAKE_COLAB_OAUTH_PYTHON={CONTROL_OAUTH_INTERPRETER}",
                f"SNAKE_COLAB_CONTROL_ROOT_RECORD={control_admission.with_suffix('.root.json')}",
            ]
        )
    environment = shlex.join(environment_items)
    finalize = ""
    if gpu_receipt is not None:
        finalize = (
            shlex.join(
                [sys.executable, str(HERE / "gpu_receipt.py"), str(manifest), str(gpu_receipt)]
            )
            + '; proof_rc=$?; if [ "$proof_rc" != 0 ]; then rc="$proof_rc"; fi; '
        )
    body = (
        f"{{ cd {shlex.quote(str(cwd))} && {environment} {shlex.join(launcher)}; }} "
        f">> {shlex.quote(str(log))} 2>&1; rc=$?; "
        f"{finalize}"
        f'printf "EXIT=%s\\n" "$rc" >> {shlex.quote(str(log))}; exit "$rc"'
    )
    if control_nice0:
        if cwd.resolve() != broker.resolve():
            raise Refused("control working/import root must equal the pinned broker")
        create_control_admission(
            ops=log.parent.parent,
            name=name,
            broker=broker,
            commit=read_json(manifest)["commit"],
            manifest=manifest,
            state=state,
            output=read_json(manifest)["output_dir"],
            pane_argv=shlex.split(persistent_shell(body, control_nice0=True))[1:],
            bootstrap_argv=launcher,
        )
    value = read_json(manifest)
    p1_window = None
    if value.get("p1_attempt2", {}).get("ruling_sha256") == P1_ATTEMPT2_RULING:
        p1_window = value.get("p1_attempt2_control_window")
        required = ("arm_path", "owner_token", "root_record", "registration_evidence")
        if not isinstance(p1_window, dict) or any(
            not isinstance(p1_window.get(key), str) or not p1_window[key].strip()
            for key in required
        ):
            raise Refused("P1 attempt-2 requires its reviewed owned control window")
    try:
        command(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                f"job-{name}",
                persistent_shell(body, p1_window=p1_window, control_nice0=control_nice0),
            ],
            deadline=deadline,
        )
    except (OSError, TimeoutError, subprocess.SubprocessError):
        note(log, "OPS_ALLOCATION_FAILURE launch submission uncertain")
        raise


def _chain_values(manifests: list[dict[str, Any]], minutes: float, units: float) -> None:
    """Fence the narrow A100 retained-main route before it reaches the broker."""
    if len(manifests) < 2:
        raise Refused("retained dispatch requires two or more packets")
    if not all(
        item.get("requirements", {}).get("gpu") == "A100"
        and item.get("requirements", {}).get("allowed_gpus") == ["A100"]
        for item in manifests
    ):
        raise Refused("retained dispatch accepts explicit A100-only packets")
    if any(item.get("canary", {}).get("mode") == "probe" for item in manifests):
        raise Refused("retained canaries cannot promote or enter an A100 main chain")
    if len({item.get("job_id") for item in manifests}) != len(manifests):
        raise Refused("retained dispatch has duplicate job identifiers")
    if len({item.get("output_dir") for item in manifests}) != len(manifests):
        raise Refused("retained dispatch requires separate output roots")
    if not all(math.isfinite(value) and value > 0 for value in (minutes, units)):
        raise Refused("retention minutes and units must be finite positive values")
    if minutes > 240 or units > 40:
        raise Refused("retained ready-work chain exceeds the ruled 4 h / 40 u ceiling")
    if any(
        item.get("resume_exemption", {}).get("kind") == "acceptance"
        and item.get("timeout_minutes", math.inf) > 60
        for item in manifests
    ):
        raise Refused("retained chain packets require individually bounded <=60 minute manifests")
    # Validate ordinary native-canary evidence independently for every main
    # packet.  The existing canary checker enforces release proof and all
    # source/input/transport pins; retained jobs never supply their own proof.
    sys.path.insert(0, str(HERE.parents[1]))
    from research.colab_cli import canary, retention

    broker = Path(manifests[0]["_broker"]).resolve()
    if broker != HERE.parents[1]:
        raise Refused("use the retained-chain dispatch wrapper from the selected broker checkout")
    try:
        retention.validate_a100_chain(manifests, minutes, units)
        for item in manifests:
            retention.runtime_cache(item, broker)
            retention.require_native_qualification(item["retention_chain"]["qualification"], broker)
    except (retention.RetentionError, KeyError, OSError, ValueError) as exc:
        raise Refused(f"retained chain lacks source-bound A100 qualification: {exc}") from exc

    for item in manifests:
        try:
            canary.check(item, Path(item["_broker"]))
        except (canary.CanaryError, KeyError, OSError, ValueError) as exc:
            raise Refused(f"retained chain requires accepted native main canary: {exc}") from exc


def start_chain(
    broker: Path,
    cwd: Path,
    state: Path,
    manifests: list[Path],
    log: Path,
    name: str,
    minutes: float,
    units: float,
    deadline: float,
) -> None:
    """Submit one qualified serial chain; the broker retains/release-checks it."""
    log.touch(exist_ok=False)
    register(log.parent.parent, log, name)
    launcher = [
        str(broker / "research/colab_cli/launch.sh"),
        "--state-dir",
        str(state),
        "--colab",
        str(Path.home() / ".local/bin/colab"),
        "chain",
        *[str(path) for path in manifests],
        "--retention-minutes",
        str(minutes),
        "--retention-units",
        str(units),
    ]
    environment = shlex.join(["env", f"HOME={Path.home()}", f'PATH={os.environ["PATH"]}'])
    body = (
        f"{{ cd {shlex.quote(str(cwd))} && {environment} {shlex.join(launcher)}; }} "
        f">> {shlex.quote(str(log))} 2>&1; rc=$?; "
        f'printf "EXIT=%s\\n" "$rc" >> {shlex.quote(str(log))}; exit "$rc"'
    )
    command(
        ["tmux", "new-session", "-d", "-s", f"job-{name}", persistent_shell(body)],
        deadline=deadline,
    )


def dispatch_chain(args: argparse.Namespace) -> str:
    """Validate and submit a bounded retained A100 chain exactly once.

    This wrapper intentionally has no allocation retry.  A chain handoff is a
    mutating operation whose ambiguity must be resolved from broker evidence,
    never submitted again by the local wrapper.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        raise Refused("NAME must contain only letters, digits, dot, underscore, hyphen")
    ops = Path(os.environ.get("OPS_DIR", OPS)).resolve()
    broker, cwd = args.broker.resolve(), args.cwd.resolve()
    state = (args.state_dir or ops / "state" / args.name).resolve()
    paths = [args.manifest.resolve(), *[path.resolve() for path in args.chain_manifest]]
    values = [read_json(path) for path in paths]
    for value in values:
        value["_broker"] = str(broker)
    _chain_values(values, args.retention_minutes, args.retention_units)
    for value in values:
        value.pop("_broker")
        require_resume_policy(value)
        if value.get("scientific_admission") is False:
            raise Refused("manifest scientific_admission=false")
        if not isinstance(value.get("budget", {}).get("other_assignments", {}), dict):
            raise Refused("budget.other_assignments must be a map")
        commit = value.get("commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise Refused("manifest commit must be a full published SHA")
    for subdir in ("logs", "state", "manifests", "dispatch"):
        (ops / subdir).mkdir(parents=True, exist_ok=True)
    qualified(broker, ops)
    for value in values:
        qualified(broker, ops, ref=value["commit"])
    reservation = ops / "dispatch" / args.name
    reservation.mkdir()
    copied: list[Path] = []
    for index, (path, value) in enumerate(zip(paths, values)):
        copied_path = ops / "manifests" / f"{args.name}-{index}.json"
        with copied_path.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        copied.append(copied_path)
    for index, path in enumerate(copied):
        validate(
            broker,
            cwd,
            state / f"packet-{index}",
            path,
            ops / "logs" / f"{args.name}-{index}.validate",
        )
    seconds = float(os.environ.get("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "600"))
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise Refused("allocation timeout must be positive and at most 600 seconds")
    with allocation_lock(ops / "alloc.lock", seconds) as deadline:
        qualified(broker, ops, deadline)
        for value in values:
            qualified(broker, ops, deadline, ref=value["commit"])
        # The supervisor itself snapshots/charges one continuous assignment.
        # Refreshing each packet here would create divergent other-assignment
        # evidence and undermine that single accounting boundary.
        log = ops / "logs" / f"{args.name}.log"
        start_chain(
            broker,
            cwd,
            state,
            copied,
            log,
            args.name,
            args.retention_minutes,
            args.retention_units,
            deadline,
        )
    ensure_watcher(ops, broker)
    return f"DISPATCHED_CHAIN {args.name} packets={len(copied)} log={log}"


def dispatch_retention_acceptance(args: argparse.Namespace) -> str:
    """Start only the separately reviewed A100 retention acceptance boundary."""
    if args.name is not None or args.manifest is not None:
        raise Refused("retention acceptance does not accept ordinary NAME/MANIFEST arguments")
    broker, cwd = args.broker.resolve(), args.cwd.resolve()
    plan, admission = args.retention_acceptance.resolve(), args.boundary_admission.resolve()
    if not plan.is_file() or not admission.is_file():
        raise Refused("retention acceptance requires existing plan and boundary admission")
    if args.review_token != "A100_RETENTION_ACCEPTANCE_REVIEWED":
        raise Refused("retention acceptance requires the exact independent review token")
    log_root = Path(os.environ.get("OPS_DIR", OPS)).resolve() / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()[:12]
    log = log_root / f"a100-retention-acceptance-{digest}.log"
    if log.exists():
        raise Refused(f"retention acceptance log already exists: {log}")
    launcher = HERE / "retention_acceptance.sh"
    arguments = [
        str(launcher),
        str(broker),
        "--plan",
        str(plan),
        "--boundary-admission",
        str(admission),
        "--review-token",
        args.review_token,
    ]
    body = (
        f"{{ cd {shlex.quote(str(cwd))} && {shlex.join(arguments)}; }} "
        f">> {shlex.quote(str(log))} 2>&1; rc=$?; "
        f'printf "EXIT=%s\\n" "$rc" >> {shlex.quote(str(log))}; exit "$rc"'
    )
    # This is intentionally outside `qualified()`: it is the candidate's
    # independent requalification itself.  The execute command validates the
    # committed inventory and final-only regression before native allocation.
    command(
        ["tmux", "new-session", "-d", "-s", f"job-a100-retain-{digest}", persistent_shell(body)]
    )
    return f"DISPATCHED_RETENTION_ACCEPTANCE plan={plan} log={log}"


def read_json(path: Path) -> dict:
    """Require an object; malformed local evidence never authorizes a retry."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Refused(f"evidence is not an object: {path}")
    return value


def session_rows(stdout: str) -> list[tuple[str, str, str]]:
    """Parse complete CLI inventory, refusing ambiguous or contradictory rows."""
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("[") or line.startswith("[colab]"):
            continue
        match = re.fullmatch(r"\[([^\]]+)\]\s+(\S+)\s+\| Hardware:\s*([^|]+?)(?:\s+\|.*)?", line)
        if match is None or not match[3].strip():
            raise Refused("unrecognized account session row")
        rows.append(match.groups())
    if not rows and "No active sessions found on server." not in stdout:
        raise Refused("unrecognized account sessions output")
    if rows and "No active sessions found on server." in stdout:
        raise Refused("contradictory account sessions output")
    if len({endpoint for _, endpoint, _ in rows}) != len(rows):
        raise Refused("duplicate endpoint in account sessions output")
    return rows


def refresh_assignments(manifest: Path, evidence: Path, deadline: float) -> Path:
    """Snapshot live broker-owned prices into a create-only sibling under the lock."""
    result = command(
        [str(Path.home() / ".local/bin/colab"), "--auth", "oauth2", "sessions"],
        deadline=deadline,
        capture_output=True,
        text=True,
    )
    rows = session_rows(result.stdout)
    sessions = {endpoint: hardware.strip() for _, endpoint, hardware in rows}
    if (
        sessions
        and read_json(manifest).get("p1_attempt2", {}).get("ruling_sha256") == P1_ATTEMPT2_RULING
    ):
        raise Refused("P1 attempt-2 requires zero other assignments immediately before allocation")
    if len(sessions) != len(rows):
        raise Refused("duplicate endpoint in account sessions output")
    records: dict[str, dict[str, Any]] = {}
    sources = []
    account = Path.home() / ".config/colab-cli/snake"
    for path in sorted(account.glob("active-*.json")):
        raw = path.read_bytes()
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise Refused(f"broker record is not an object: {path}")
        endpoint, session = record.get("endpoint"), record.get("session")
        if (
            not isinstance(endpoint, str)
            or not endpoint
            or not isinstance(session, str)
            or not session
            or path.name != f"active-{session}.json"
            or endpoint in records
        ):
            raise Refused(f"ambiguous broker active identity: {path}")
        records[endpoint] = record
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    foreign = sorted(set(sessions) - set(records))
    if foreign:
        raise Refused("live endpoints not owned by our broker: " + ", ".join(foreign))
    if "sandbox_admission" in read_json(manifest):
        from tools.sandbox.manifest import protected_name

        for endpoint in sessions:
            record = records[endpoint]
            if any(protected_name(str(record.get(key, ""))) for key in ("output", "session")):
                # Do not open a protected running package to infer permission.
                # No active-contract coexistence proof is implemented here.
                raise Refused("sandbox waits for protected F5/study assignment to stop")
    rates = {}
    for endpoint, hardware in sessions.items():
        record = records[endpoint]
        # Match qualified accounting: a reserved ceiling is not a measured price.
        field = (
            "observed_rate_units_per_hour"
            if "priced_rate_units_per_hour" in record
            else "rate_units_per_hour"
        )
        rate = record.get(field)
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise Refused(f"broker endpoint lacks a calibrated positive rate: {endpoint}")
        registered = {"L4": 1.54, "CPU": 0.08}.get(hardware.upper())
        if registered is not None and not math.isclose(rate, registered, rel_tol=0, abs_tol=1e-9):
            raise Refused(f"broker rate differs from registered {hardware} rate: {endpoint}")
        if hardware.upper() not in ("L4", "CPU", "A100", "H100"):
            raise Refused(f"unregistered live hardware: {hardware}")
        rates[endpoint] = rate
    original = manifest.read_bytes()
    value = json.loads(original)
    require_f5_plan_live_assignment_match(value, {"budget": {"other_assignments": rates}})
    value["budget"]["other_assignments"] = rates
    refreshed = manifest.with_name(f"{manifest.stem}.live.json")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with refreshed.open("xb") as stream:
        stream.write(payload)
    proof = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "manifest": str(refreshed),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "sessions_stdout": result.stdout,
        "broker_records": sources,
        "other_assignments": rates,
    }
    with evidence.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return refreshed


# Owner's 2026-10-22 A0 work order explicitly selects PLAN_A100 §3's
# A1 acceptance envelope over the earlier generic 60-minute exemption ceiling.
A1_AUTHORITY_SHA256 = "1c87898be554d8e66788da6f0a09ee7b842a514e530c1d0e1c624bcf58c3962c"


def a1_acceptance_limit(manifest: dict[str, Any], exemption: dict[str, Any]) -> bool:
    """Recognize only the source-bound A1 ruler's authorized 2 h / 20 u envelope."""
    return (
        exemption.get("kind") == "acceptance"
        and exemption.get("authority_sha256") == A1_AUTHORITY_SHA256
        and manifest.get("script") == "research/gpu_sim_spike/colab.sh"
        and manifest.get("env", {}).get("GPU_SIM_SPIKE_ENTRYPOINT") == "a1-ruler"
        and manifest.get("requirements", {}).get("gpu") == "A100"
        and manifest.get("requirements", {}).get("allowed_gpus") == ["A100"]
        and manifest.get("budget", {}).get("unit_cap") == 20
        and manifest.get("a1", {}).get("shapes") == [6, 256, 1024]
        and manifest.get("a1", {}).get("mixes") == ["scripted", "frozen", "mixed"]
        and manifest.get("a1", {}).get("repeats") == 3
        and manifest.get("a1", {}).get("ticks_per_repeat") == 8
    )


def require_resume_policy(manifest: dict[str, Any]) -> None:
    """Require a resume contract or a declared, bounded operational exemption."""
    if "resume" in manifest and "resume_exemption" in manifest:
        raise Refused("declare resume or resume_exemption, never both")
    if "resume_exemption" in manifest:
        exemption = manifest["resume_exemption"]
        if not isinstance(exemption, dict) or exemption.get("kind") not in (
            "short-canary",
            "acceptance",
        ):
            raise Refused("resume_exemption.kind must be short-canary or acceptance")
        reason = exemption.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise Refused("resume_exemption.reason must be nonempty")
        timeout = manifest.get("timeout_minutes")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise Refused("resume exemption requires a finite positive timeout_minutes")
        # Owner's 2x rule: ruling's 20-minute canary; acceptance's 30-minute maximum.
        limit = 40 if exemption["kind"] == "short-canary" else 60
        if a1_acceptance_limit(manifest, exemption):
            limit = 120
        if timeout > limit:
            raise Refused(
                f"{exemption['kind']} resume exemption requires timeout_minutes <= {limit}"
            )
        return
    resume = manifest.get("resume")
    if not isinstance(resume, dict):
        raise Refused("manifest requires a resume block (or declared resume_exemption)")
    for field in ("unit_kind", "ledger_path"):
        if not isinstance(resume.get(field), str) or not resume[field].strip():
            raise Refused(f"resume.{field} must be nonempty")
    if resume.get("skip_verified") is not True:
        raise Refused("resume.skip_verified must be true")
    retries = resume.get("max_outcome_blind_retries")
    # Amendment 06 narrows the generic maximum for this exact ruled family.
    # This declaration never launches an automatic research completion retry.
    ruling = "fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc"
    env = manifest.get("env")
    phase = env.get("BL_P1_MODE") if isinstance(env, dict) else None
    ruled = manifest.get("p1_finalisation_ruling_sha256") == ruling
    maximum = {"y3-source": 1, "y3-replacement": 1, "qual-1m": 0}.get(phase)
    attempt2 = manifest.get("p1_attempt2", {})
    if ruled and phase == "y3-source" and attempt2.get("ruling_sha256") == P1_ATTEMPT2_RULING:
        if (
            type(retries) is not int
            or retries != 2
            or attempt2.get("canary_attempts") != 1
            or attempt2.get("repair_attempts") != 1
            or attempt2.get("canary_retries") != 0
            or attempt2.get("further_repair_retries") != 0
        ):
            raise Refused("P1 attempt-2 requires wrapper maximum 2 and no further research retries")
    elif ruled and maximum is not None:
        if type(retries) is not int or retries != maximum:
            raise Refused(f"resume.max_outcome_blind_retries must be the ruled integer {maximum}")
    elif type(retries) is not int or retries != 2:
        raise Refused("resume.max_outcome_blind_retries must be the integer 2")


def require_timing_calibration_retry_policy(manifest: dict[str, Any]) -> bool:
    """Return whether this manifest refuses the wrapper allocation retry.

    This opt-in is deliberately confined to timing-calibration manifests. The
    wrapper's historical accounting retry remains unchanged for every manifest
    without the block.
    """
    if "timing_calibration" not in manifest:
        return False
    timing_calibration = manifest["timing_calibration"]
    if not isinstance(timing_calibration, dict):
        raise Refused("timing_calibration must be an object")
    if "no_automatic_retry" not in timing_calibration:
        return False
    no_automatic_retry = timing_calibration["no_automatic_retry"]
    if type(no_automatic_retry) is not bool:
        raise Refused("timing_calibration.no_automatic_retry must be a boolean")
    return no_automatic_retry


def require_scientific_or_f5_plan_admission(manifest: dict[str, Any]) -> None:
    """Permit only sealed non-scientific F5 operational qualifications.

    ``scientific_admission: false`` normally stops the ordinary dispatch path.
    This narrow operational exception is intentionally self-contained: it has
    no bearing on retained chains, fits, timing qualification, or other false
    scientific admissions.
    """
    if manifest.get("scientific_admission") is not False:
        return
    admission = manifest.get("operational_admission")
    if not isinstance(admission, dict):
        raise Refused("manifest scientific_admission=false")
    if admission.get("schema") == F5_TIMING_ADMISSION_SCHEMA:
        if admission.get("status") in {
            "ACCEPTED_F5_TIMING_RECOVERY_ONLY",
            "ACCEPTED_F5_TIMING_RECOVERY_CANARY_ONLY",
        }:
            require_f5_timing_recovery_admission(manifest, admission)
            return
        require_f5_timing_operational_admission(manifest, admission)
        return
    if admission.get("schema") != F5_PLAN_ADMISSION_SCHEMA:
        raise Refused("manifest scientific_admission=false")
    if manifest.get("draft_only") is not None:
        raise Refused("F5 plan admission forbids draft_only")
    if manifest.get("commit") == F5_PLAN_REQUAL_COMMIT:
        for key, expected_sha in F5_PLAN_REQUAL_BINDINGS.items():
            reference = admission.get(key)
            if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                raise Refused(f"F5 requalified plan requires its exact evidence binding: {key}")
            path = Path(reference["path"])
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise Refused(
                    f"F5 requalified plan evidence must be a regular absolute file: {key}"
                )
            payload = path.read_bytes()
            if (
                reference.get("sha256") != expected_sha
                or reference.get("size") != len(payload)
                or hashlib.sha256(payload).hexdigest() != expected_sha
            ):
                raise Refused(f"F5 requalified plan evidence bytes differ: {key}")
    elif manifest.get("commit") != F5_PLAN_COMMIT:
        raise Refused("F5 plan admission requires the accepted implementation commit")
    if manifest.get("script") != "research/bl_residual/f5_colab.sh":
        raise Refused("F5 plan admission requires the F5 Colab entrypoint")
    if manifest.get("device") != "cuda:0":
        raise Refused("F5 plan admission requires device cuda:0")
    if manifest.get("env") != F5_PLAN_ENV:
        raise Refused("F5 plan admission requires the exact prepared environment")
    if manifest.get("substeps") != [
        "CPU_PLAN_MATERIALIZATION",
        "DISJOINT_CUDA_TRANSPORT_QUALIFICATION",
    ]:
        raise Refused("F5 plan admission has unauthorized substeps")
    if manifest.get("requirements") != {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 6,
        "est_minutes": 15,
    }:
        raise Refused("F5 plan admission requires the exact L4 placement")
    if manifest.get("timeout_minutes") != 30:
        raise Refused("F5 plan admission requires the 30-minute effective timeout")
    exemption = manifest.get("resume_exemption")
    if not isinstance(exemption, dict) or exemption.get("kind") != "short-canary":
        raise Refused("F5 plan admission requires a short-canary exemption")
    if exemption.get("timeout_minutes") != 30 or "resume" in manifest:
        raise Refused("F5 plan admission has an invalid resume declaration")
    if manifest.get("required_exports") != F5_PLAN_REQUIRED_EXPORTS:
        raise Refused("F5 plan admission has an unauthorized export inventory")
    if manifest.get("canary") != {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/bl_residual/f5_colab.sh",
    }:
        raise Refused("F5 plan admission requires the exact probe canary")
    max_export_bytes = manifest.get("max_export_bytes")
    if type(max_export_bytes) is not int or not 0 < max_export_bytes <= 2 * 1024**3:
        raise Refused("F5 plan admission requires a bounded export cap")
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, list)
        or {item.get("dest") for item in inputs if isinstance(item, dict)}
        != (F5_PLAN_INPUT_DESTINATIONS)
        or len(inputs) != len(F5_PLAN_INPUT_DESTINATIONS)
    ):
        raise Refused("F5 plan admission has an unauthorized input inventory")
    input_by_dest = {item["dest"]: item for item in inputs}
    b8 = input_by_dest["b8.pth"]
    if (
        b8.get("url") != F5_B8_URL
        or b8.get("sha256") != F5_B8_SHA256
        or b8.get("size_bytes") != 6562005
        or "path" in b8
    ):
        raise Refused("F5 plan admission requires the pinned HTTPS B8 input")
    if input_by_dest["selected-training.tar.xz"].get("sha256") != F5_SELECTED_TRAINING_SHA256:
        raise Refused("F5 plan admission requires the selected-training binding")
    f5 = manifest.get("f5")
    if not isinstance(f5, dict) or (
        f5.get("base_timeout_minutes"),
        f5.get("effective_timeout_minutes"),
        f5.get("automatic_retries"),
    ) != (15, 30, False):
        raise Refused("F5 plan admission requires bounded no-retry timing controls")
    if (
        f5.get("private_upload_cap_bytes") != 45397643
        or f5.get("private_upload_bytes_max") != 45397643
        or f5.get("transfer_bar_seconds") != 150
        or f5.get("staging_bar_seconds") != 300
    ):
        raise Refused("F5 plan admission requires the accepted transport bounds")
    timing = manifest.get("timing_calibration")
    retry_fields = {
        "no_automatic_retry",
        "no_allocation_retry",
        "no_job_retry",
        "no_completion_retry",
    }
    if not isinstance(timing, dict) or any(timing.get(field) is not True for field in retry_fields):
        raise Refused("F5 plan admission requires all timing retry fences")
    if (
        any(
            admission.get(field) is not False
            for field in (
                "scientific_fits_authorized",
                "numerical_looks_authorized",
                "release_authorized",
            )
        )
        or admission.get("status") != "ACCEPTED_PLAN_MATERIALIZATION_AND_TRANSPORT_ONLY"
    ):
        raise Refused("F5 plan admission grants unauthorized scientific authority")
    rulings = admission.get("rulings")
    if not isinstance(rulings, dict) or set(rulings) != set(F5_PLAN_RULINGS):
        raise Refused("F5 plan admission has incomplete ruling bindings")
    for key, expected_sha in F5_PLAN_RULINGS.items():
        reference = rulings[key]
        if not isinstance(reference, dict):
            raise Refused(f"F5 plan admission ruling is malformed: {key}")
        path_value = reference.get("path")
        if not isinstance(path_value, str):
            raise Refused(f"F5 plan admission ruling path is malformed: {key}")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise Refused(f"F5 plan admission ruling must be a regular file: {key}")
        payload = path.read_bytes()
        if (
            reference.get("sha256") != expected_sha
            or reference.get("size") != len(payload)
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise Refused(f"F5 plan admission ruling bytes differ: {key}")


def require_f5_timing_operational_admission(
    manifest: dict[str, Any], admission: dict[str, Any]
) -> None:
    """Permit exactly the timing/parity/calibration successor accepted for F5.

    This admission binds source-native qualification only.  It excludes the
    common-plan training archive and grants no fit, numerical-look, or release
    authority.  The successor's control and ledger are deliberately allowed to
    receive fresh package-specific digests after their allocations are sealed.
    """
    if manifest.get("draft_only") is not None:
        raise Refused("F5 timing admission forbids draft_only")
    if manifest.get("commit") == F5_PLAN_REQUAL_COMMIT:
        for key, expected_sha in F5_TIMING_REQUAL_BINDINGS.items():
            reference = admission.get(key)
            if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                raise Refused(f"F5 requalified timing requires its exact evidence binding: {key}")
            path = Path(reference["path"])
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise Refused(
                    f"F5 requalified timing evidence must be a regular absolute file: {key}"
                )
            payload = path.read_bytes()
            if (
                reference.get("sha256") != expected_sha
                or reference.get("size") != len(payload)
                or hashlib.sha256(payload).hexdigest() != expected_sha
            ):
                raise Refused(f"F5 requalified timing evidence bytes differ: {key}")
    elif manifest.get("commit") != F5_PLAN_COMMIT:
        raise Refused("F5 timing admission requires the accepted implementation commit")
    if manifest.get("script") != "research/bl_residual/f5_colab.sh":
        raise Refused("F5 timing admission requires the F5 Colab entrypoint")
    if manifest.get("device") != "cuda:0":
        raise Refused("F5 timing admission requires device cuda:0")
    if manifest.get("env") != F5_TIMING_ENV:
        raise Refused("F5 timing admission requires the exact qualification environment")
    if manifest.get("substeps") is not None:
        raise Refused("F5 timing admission has unauthorized substeps")
    if manifest.get("requirements") != {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 6,
        "est_minutes": 15,
    }:
        raise Refused("F5 timing admission requires the exact L4 placement")
    if manifest.get("timeout_minutes") != 30:
        raise Refused("F5 timing admission requires the 30-minute effective timeout")
    exemption = manifest.get("resume_exemption")
    if not isinstance(exemption, dict) or exemption.get("kind") != "short-canary":
        raise Refused("F5 timing admission requires a short-canary exemption")
    if exemption.get("timeout_minutes") != 30 or "resume" in manifest:
        raise Refused("F5 timing admission has an invalid resume declaration")
    if manifest.get("required_exports") != F5_TIMING_REQUIRED_EXPORTS:
        raise Refused("F5 timing admission has an unauthorized export inventory")
    if manifest.get("canary") != {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/bl_residual/f5_colab.sh",
    }:
        raise Refused("F5 timing admission requires the exact probe canary")
    max_export_bytes = manifest.get("max_export_bytes")
    if type(max_export_bytes) is not int or not 0 < max_export_bytes <= 2 * 1024**3:
        raise Refused("F5 timing admission requires a bounded export cap")
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, list)
        or {item.get("dest") for item in inputs if isinstance(item, dict)}
        != F5_TIMING_INPUT_DESTINATIONS
        or len(inputs) != len(F5_TIMING_INPUT_DESTINATIONS)
    ):
        raise Refused("F5 timing admission has an unauthorized input inventory")
    input_by_dest = {item["dest"]: item for item in inputs}
    if any(
        input_by_dest[dest].get("sha256") != digest
        for dest, digest in F5_TIMING_INPUT_SHA256.items()
    ):
        raise Refused("F5 timing admission requires the pinned timing and opponent inputs")
    b8 = input_by_dest["b8.pth"]
    if b8.get("url") != F5_B8_URL or b8.get("size_bytes") != 6562005 or "path" in b8:
        raise Refused("F5 timing admission requires the pinned HTTPS B8 input")
    for dest in ("control.json", "verified-ledger.json"):
        digest = input_by_dest[dest].get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise Refused(f"F5 timing admission requires a sealed {dest} digest")
    f5 = manifest.get("f5")
    if not isinstance(f5, dict) or (
        f5.get("base_timeout_minutes"),
        f5.get("effective_timeout_minutes"),
        f5.get("automatic_retries"),
        f5.get("private_upload_cap_bytes"),
        f5.get("private_upload_bytes_max"),
        f5.get("transfer_bar_seconds"),
        f5.get("staging_bar_seconds"),
    ) != (15, 30, False, 45397643, 45397643, 150, 300):
        raise Refused("F5 timing admission requires bounded no-retry timing controls")
    timing = manifest.get("timing_calibration")
    retry_fields = {
        "no_automatic_retry",
        "no_allocation_retry",
        "no_job_retry",
        "no_completion_retry",
    }
    if not isinstance(timing, dict) or any(timing.get(field) is not True for field in retry_fields):
        raise Refused("F5 timing admission requires all timing retry fences")
    if (
        any(
            admission.get(field) is not False
            for field in (
                "scientific_fits_authorized",
                "numerical_looks_authorized",
                "release_authorized",
            )
        )
        or admission.get("status") != "ACCEPTED_FULL_TIMING_PARITY_AND_CALIBRATION_ONLY"
    ):
        raise Refused("F5 timing admission grants unauthorized scientific authority")
    rulings = admission.get("rulings")
    if not isinstance(rulings, dict) or set(rulings) != set(F5_PLAN_RULINGS):
        raise Refused("F5 timing admission has incomplete ruling bindings")
    for key, expected_sha in F5_PLAN_RULINGS.items():
        reference = rulings[key]
        if not isinstance(reference, dict):
            raise Refused(f"F5 timing admission ruling is malformed: {key}")
        path_value = reference.get("path")
        if not isinstance(path_value, str):
            raise Refused(f"F5 timing admission ruling path is malformed: {key}")
        path = Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise Refused(f"F5 timing admission ruling must be a regular file: {key}")
        payload = path.read_bytes()
        if (
            reference.get("sha256") != expected_sha
            or reference.get("size") != len(payload)
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise Refused(f"F5 timing admission ruling bytes differ: {key}")


def _require_recovery_authority(manifest: dict[str, Any], admission: dict[str, Any]) -> None:
    """Bind the F5-only 300-second exception to its sealed inventory bytes."""
    reference = admission.get("timing_recovery_authority")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "size"}:
        raise Refused("F5 timing recovery requires its exact authority reference")
    path_value, expected_sha, expected_size = (
        reference.get("path"),
        reference.get("sha256"),
        reference.get("size"),
    )
    if (
        not isinstance(path_value, str)
        or not isinstance(expected_sha, str)
        or type(expected_size) is not int
    ):
        raise Refused("F5 timing recovery authority reference is malformed")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise Refused("F5 timing recovery authority must be a regular absolute file")
    payload = path.read_bytes()
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha:
        raise Refused("F5 timing recovery authority bytes differ")
    try:
        authority = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise Refused("F5 timing recovery authority is not JSON") from exc
    if authority != manifest.get("f5_timing_synthetic_inventory"):
        raise Refused("F5 timing recovery inventory differs from its sealed authority")
    if not isinstance(authority, dict) or authority.get("schema") != (
        "f5-timing-recovery-synthetic-input-inventory/v1"
    ):
        raise Refused("F5 timing recovery authority schema differs")
    if authority.get("ruling_sha256") != F5_TIMING_RECOVERY_RULING_SHA256:
        raise Refused("F5 timing recovery ruling differs")
    if authority.get("source_commit") != manifest.get("commit"):
        raise Refused("F5 timing recovery source commit differs")
    if (
        not isinstance(authority.get("source_tree"), str)
        or re.fullmatch(r"[0-9a-f]{40}", authority["source_tree"]) is None
    ):
        raise Refused("F5 timing recovery source tree differs")
    try:
        source_tree = subprocess.check_output(
            ["git", "-C", str(HERE.parents[1]), "rev-parse", f"{manifest['commit']}^{{tree}}"],
            text=True,
            timeout=5,
        ).strip()
    except (KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise Refused("F5 timing recovery source commit is unavailable") from exc
    if authority["source_tree"] != source_tree:
        raise Refused("F5 timing recovery source tree is not the committed source closure")
    if manifest.get("f5_timing_source_tree") != source_tree:
        raise Refused("F5 timing recovery manifest source tree differs")
    try:
        from research.colab_cli.transfer_policy import transfer_limit

        limit = transfer_limit(manifest)
    except (TypeError, ValueError) as exc:
        raise Refused("F5 timing recovery transfer authority is refused") from exc
    if limit != 300.0:
        raise Refused("F5 timing recovery requires the scoped 300-second transfer allowance")


def _require_f5_recovery_common(
    manifest: dict[str, Any], admission: dict[str, Any], *, canary: bool
) -> None:
    """Validate source, L4 placement, inventory and non-scientific recovery scope."""
    if manifest.get("draft_only") is not None or manifest.get("substeps") is not None:
        raise Refused("F5 timing recovery has unauthorized draft or substeps")
    if manifest.get("script") != "research/bl_residual/f5_colab.sh":
        raise Refused("F5 timing recovery requires the F5 Colab entrypoint")
    if manifest.get("device") != "cuda:0":
        raise Refused("F5 timing recovery requires device cuda:0")
    expected_env = dict(F5_TIMING_RECOVERY_ENV)
    if canary:
        expected_env.update(
            {
                "F5_MODE": "qualification-canary",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_residual",
            }
        )
    if manifest.get("env") != expected_env:
        raise Refused("F5 timing recovery requires its exact runtime environment")
    if (
        manifest.get("requirements", {}).get("gpu") != "L4"
        or manifest.get("requirements", {}).get("allowed_gpus") != ["L4"]
        or manifest.get("requirements", {}).get("min_vcpu") != 6
    ):
        raise Refused("F5 timing recovery requires the exact L4 placement")
    if any(
        admission.get(field) is not False
        for field in (
            "scientific_fits_authorized",
            "numerical_looks_authorized",
            "release_authorized",
        )
    ):
        raise Refused("F5 timing recovery grants unauthorized scientific authority")
    recovery = admission.get("timing_recovery")
    if not isinstance(recovery, dict) or recovery.get("schema") != F5_TIMING_RECOVERY_SCHEMA:
        raise Refused("F5 timing recovery requires its scoped admission")
    if recovery.get("ruling_sha256") != F5_TIMING_RECOVERY_RULING_SHA256:
        raise Refused("F5 timing recovery ruling differs")
    _require_recovery_authority(manifest, admission)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise Refused("F5 timing recovery inputs are malformed")
    ledgers = [
        item
        for item in inputs
        if isinstance(item, dict) and item.get("dest") == "verified-ledger.json"
    ]
    if len(ledgers) != 1:
        raise Refused("F5 timing recovery requires one hash-bound verified ledger input")
    ledger = ledgers[0]
    ledger_path = Path(str(ledger.get("path", "")))
    if (
        not ledger_path.is_absolute()
        or ledger_path.is_symlink()
        or not ledger_path.is_file()
        or not isinstance(ledger.get("sha256"), str)
        or type(ledger.get("size_bytes")) is not int
        or ledger_path.stat().st_size != ledger["size_bytes"]
        or hashlib.sha256(ledger_path.read_bytes()).hexdigest() != ledger["sha256"]
    ):
        raise Refused("F5 timing recovery verified ledger input differs")
    try:
        from research.bl_ppo.exports import verify_export

        if verify_export(ledger_path) != ledger["sha256"]:
            raise Refused("F5 timing recovery verified ledger digest differs")
    except ValueError as exc:
        raise Refused("F5 timing recovery verified ledger lacks immutable sidecars") from exc
    f5 = manifest.get("f5")
    if not isinstance(f5, dict) or (
        f5.get("automatic_retries"),
        f5.get("private_upload_bytes_max"),
        f5.get("transfer_seconds"),
        f5.get("staging_seconds"),
    ) != (False, 45397643, 300, 300):
        raise Refused("F5 timing recovery requires bounded 300-second transport controls")
    timing = manifest.get("timing_calibration")
    retry_fields = (
        "no_automatic_retry",
        "no_allocation_retry",
        "no_job_retry",
        "no_completion_retry",
    )
    if not isinstance(timing, dict) or any(timing.get(field) is not True for field in retry_fields):
        raise Refused("F5 timing recovery requires all timing retry fences")


def require_f5_timing_recovery_admission(
    manifest: dict[str, Any], admission: dict[str, Any]
) -> None:
    """Admit only the ruling's sealed recovery operation or its CANARY_ONLY probe.

    This is deliberately distinct from the historical F5 timing admission.  It
    leaves the former 15/30 probe, all generic admissions, and the D1024
    exception untouched while making the timing-specific transport authority
    machine-checkable at the ops boundary.
    """
    status = admission.get("status")
    if status not in {
        "ACCEPTED_F5_TIMING_RECOVERY_ONLY",
        "ACCEPTED_F5_TIMING_RECOVERY_CANARY_ONLY",
    }:
        raise Refused("F5 timing recovery admission status differs")
    _require_f5_recovery_common(
        manifest, admission, canary=status == "ACCEPTED_F5_TIMING_RECOVERY_CANARY_ONLY"
    )
    recovery = admission["timing_recovery"]
    f5 = manifest["f5"]
    requirements = manifest["requirements"]
    if status == "ACCEPTED_F5_TIMING_RECOVERY_ONLY":
        if recovery.get("scope") != "F5_W_TIMING04_FULL_RESUMABLE_OPERATION_ONLY":
            raise Refused("F5 timing recovery scope differs")
        if requirements.get("est_minutes") != 120 or manifest.get("timeout_minutes") != 240:
            raise Refused("F5 timing recovery requires the 120/240-minute envelope")
        if (
            f5.get("base_minutes"),
            f5.get("effective_minutes"),
            f5.get("sizing_basis"),
        ) != (120, 240, "UNMEASURED_ENGINEERING_PLANNING_ALLOWANCE"):
            raise Refused("F5 timing recovery requires labelled unmeasured 120/240 sizing")
        family_receipt = manifest.get("canary")
        if family_receipt is not None and (
            not isinstance(family_receipt, dict)
            or set(family_receipt) != {"receipt"}
            or not isinstance(family_receipt["receipt"], dict)
            or set(family_receipt["receipt"]) != {"path", "sha256"}
        ):
            raise Refused("F5 timing recovery permits only a bound family receipt")
        if "resume_exemption" in manifest:
            raise Refused("F5 timing recovery long operation forbids exemption or canary controls")
        resume = manifest.get("resume")
        if resume != {
            "unit_kind": "f5-timing-unit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        }:
            raise Refused("F5 timing recovery requires its exact resumable unit declaration")
        if (
            type(resume["skip_verified"]) is not bool
            or type(resume["max_outcome_blind_retries"]) is not int
        ):
            raise Refused("F5 timing recovery resume types differ")
        # A completion attempt deliberately skips units already sealed in the
        # input ledger.  Requiring every unit path here would force a producer
        # to duplicate those immutable exports.  The coverage proof instead
        # binds the input ledger, skipped IDs, newly exported IDs, and final
        # coverage for this particular attempt.
        expected_exports = [
            "qualification/qualification.json",
            "resume/verified-skip-proof.json",
            "environment/before.json",
            "environment/after.json",
        ]
        if manifest.get("required_exports") != expected_exports:
            raise Refused("F5 timing recovery requires its sealed unit coverage proof")
        return
    if recovery.get("scope") != "F5_W_TIMING04_FAMILY_CANARY_ONLY":
        raise Refused("F5 timing recovery scope differs")
    if requirements.get("est_minutes") != 15 or manifest.get("timeout_minutes") != 30:
        raise Refused("F5 timing recovery canary requires the 15/30-minute envelope")
    if (f5.get("base_minutes"), f5.get("effective_minutes")) != (15, 30):
        raise Refused("F5 timing recovery canary sizing differs")
    canary = manifest.get("canary")
    if canary != {
        "schema": "colab-family-canary/v1",
        "mode": "probe",
        "family": "research/bl_residual/f5_colab.sh",
        "record_class": "CANARY_ONLY",
        "tiny_workload": True,
    }:
        raise Refused("F5 timing recovery requires the exact CANARY_ONLY tiny-workload controls")
    if manifest.get("resume") != {
        "unit_kind": "f5-timing-unit",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise Refused("F5 timing recovery canary requires the real resume declaration")
    resume = manifest["resume"]
    if (
        type(resume["skip_verified"]) is not bool
        or type(resume["max_outcome_blind_retries"]) is not int
    ):
        raise Refused("F5 timing recovery resume types differ")
    if manifest.get("required_exports") != [
        "qualification/qualification.json",
        "resume/verified-skip-proof.json",
        "environment/before.json",
        "environment/after.json",
    ]:
        raise Refused("F5 timing recovery canary requires its CANARY_ONLY coverage proof")


def require_f5_plan_live_assignment_match(original: dict[str, Any], live: dict[str, Any]) -> None:
    """Reject a changed account snapshot instead of mutating a sealed F5 package."""
    admission = original.get("operational_admission")
    if isinstance(admission, dict) and admission.get("schema") in {
        F5_PLAN_ADMISSION_SCHEMA,
        F5_TIMING_ADMISSION_SCHEMA,
    }:
        if live.get("budget", {}).get("other_assignments") != original.get("budget", {}).get(
            "other_assignments"
        ):
            raise Refused("F5 live assignments differ from sealed admission; seal a successor")


def wait_allocation(output: Path, log: Path, deadline: float, poll: float) -> str:
    """An assignment record is too early: accounting calibration must finish."""
    while time.monotonic() < deadline:
        calibration = output / "rate-calibration.json"
        if calibration.is_file() and read_json(calibration).get("status") == "PASS":
            note(log, f"OPS_ALLOCATION_SUCCESS calibration={calibration}")
            return "success"
        receipt = output / "receipt.json"
        if receipt.is_file():
            value = read_json(receipt)
            retry = (
                value.get("status") == "INCOMPLETE"
                and value.get("failure_class") == "ACCOUNTING"
                and value.get("error") == EXACT_ERROR
                and value.get("cleanup_error") is None
                and value.get("stop_verified") is True
            )
            # Receipt publication precedes shell exit; wait before a retry.
            exited = bool(re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE))
            if retry and not exited:
                time.sleep(min(poll, max(0, deadline - time.monotonic())))
                continue
            note(log, "OPS_ALLOCATION_FAILURE_RETRYABLE" if retry else "OPS_ALLOCATION_FAILURE")
            return "retry" if retry else "failed"
        if re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE):
            note(log, "OPS_ALLOCATION_FAILURE no-terminal-receipt")
            return "failed"
        time.sleep(min(poll, max(0, deadline - time.monotonic())))
    note(log, "OPS_ALLOCATION_WAIT_TIMEOUT")
    return "timeout"


def ensure_watcher(ops: Path, broker: Path) -> None:
    """Start the watcher outside the allocation critical section."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", "watchd"],
        timeout=10,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        command(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                "watchd",
                persistent_shell(
                    shlex.join(
                        ["env", f"OPS_DIR={ops}", f"OPS_REPO={broker}", str(HERE / "watchd.sh")]
                    )
                ),
            ]
        )


ALLOCATION_UNAVAILABLE = (
    "[colab] Allocation refused (precondition failed). This can mean "
    "too many active sessions, or a temporary usage or capacity "
    "limit for the requested runtime. Run `colab stop` to free up a "
    "session, wait and retry, or try a different accelerator."
)


def allocation_unavailable(output: Path, log: Path) -> bool:
    """Recognize only the installed 0.7.4 CLI's explicit allocation rejection.

    This is only a candidate: reconciliation below must independently prove
    no assignment. Timeout, entitlement, accounting and post-endpoint errors
    never qualify. The qualified supervisor intentionally retains uncertainty.
    """
    if not re.search(r"^EXIT=[1-9]\d*$", log.read_text(), re.MULTILINE):
        return False
    path = output / "receipt.json"
    if not path.is_file():
        return False
    receipt = read_json(path)
    error = receipt.get("error", "")
    if not isinstance(error, str):
        return False
    try:
        detail = (
            ast.literal_eval(error[len("Refused(") : -1]) if error.startswith("Refused(") else ""
        )
    except (ValueError, SyntaxError):
        return False
    session = receipt.get("session")
    expected = (
        f"colab new failed (1): [colab] Creating session '{session}'...\n"
        + ALLOCATION_UNAVAILABLE
        + "\n"
    )
    calls = receipt.get("cli_calls")
    return (
        detail == expected
        and receipt.get("status") == "INCOMPLETE"
        and receipt.get("failure_class") == "SUPERVISOR"
        and receipt.get("endpoint") is None
        and receipt.get("terminal") is None
        and receipt.get("cleanup_error") == "ALLOCATION_UNRESOLVED: retained pending record"
        and receipt.get("stop_verified") is False
        and all(receipt.get(key) == [] for key in ("exports", "transfers", "contents_transfers"))
        and isinstance(calls, list)
        and all(isinstance(call, dict) for call in calls)
        and sum(call.get("command") == "new" for call in calls) == 1
        and all(call.get("command") in {"version", "sessions", "usage", "new"} for call in calls)
        and next(call for call in calls if call["command"] == "new").get("status") == "failed"
        and not any(
            (output / name).exists()
            for name in ("hardware.json", "terminal.json", "rate-calibration.json")
        )
    )


def reconcile_unavailable(
    broker: Path, manifest: Path, state: Path, log: Path, proof: Path, deadline: float
) -> None:
    """Archive and retire only the rejected request's proven empty pending intent.

    Hold the same account mutation lock as the supervisor while issuing GET-only
    recovery/inventory. Any endpoint, missing evidence, changed baseline, helper
    failure or unrelated pending record leaves the account fenced for an operator.
    """
    packet = read_json(manifest)
    output = Path(packet["output_dir"])
    if not allocation_unavailable(output, log):
        raise Refused("H100 allocation unavailability is not proven")
    receipt_path = output / "receipt.json"
    receipt = read_json(receipt_path)
    session = receipt.get("session", "")
    if not re.fullmatch(r"snake-[a-z0-9_-]+-[a-f0-9]{12}", session):
        raise Refused("unavailable allocation lacks an exact supervisor session")
    executed = output / "manifest.json"
    if (
        not executed.is_file()
        or hashlib.sha256(executed.read_bytes()).hexdigest() != receipt.get("manifest_sha256")
        or read_json(executed).get("job_id") != packet["job_id"]
        or read_json(executed).get("requirements") != packet["requirements"]
        or receipt.get("job_id") != packet["job_id"]
        or receipt.get("commit") != packet["commit"]
    ):
        raise Refused("unavailable allocation receipt/manifest identity differs")
    account = Path.home() / ".config/colab-cli/snake"
    with allocation_lock(account / "broker.lock", max(0.001, deadline - time.monotonic())):
        pending = account / f"pending-{session}.json"
        if set(account.glob("pending-*.json")) != {pending}:
            raise Refused("unavailable allocation has ambiguous pending ownership")
        raw = pending.read_bytes()
        intent = read_json(pending)
        lane = state / packet["job_id"]
        config = lane / "sessions.json"
        assignment = config.with_suffix(".assignment.json")
        # The durable identity filename comes from the supervisor's CLI class.
        if intent.get("assignment_record") != str(assignment):
            raise Refused("unavailable allocation assignment record path differs")
        if (
            intent.get("session") != session
            or intent.get("output") != str(output)
            or intent.get("config") != str(config)
            or "endpoint" in intent
            or pending.is_symlink()
            or config.is_symlink()
            or assignment.exists()
            or assignment.is_symlink()
            or (lane / "assignment.json").exists()
            or (account / f"active-{session}.json").exists()
            or (config.exists() and read_json(config).get(session))
        ):
            raise Refused("an allocation identity exists or pending binding differs")
        notebook = intent.get("notebook_hash", "")
        if not re.fullmatch(r"[a-f0-9]{32}", notebook):
            raise Refused("unavailable allocation notebook identity missing")
        recovered = command(
            [
                str(CONTROL_OAUTH_INTERPRETER),
                str(broker / "research/colab_cli/cli_guard.py"),
                "--recover-assignment",
                notebook,
            ],
            deadline=deadline,
            capture_output=True,
            text=True,
        )
        if json.loads(recovered.stdout) != {"endpoint": None}:
            raise Refused("H100 endpoint recovered; fallback forbidden")
        inventory = command(
            [str(Path.home() / ".local/bin/colab"), "--auth", "oauth2", "sessions"],
            deadline=deadline,
            capture_output=True,
            text=True,
        ).stdout
        rows = session_rows(inventory)
        before = intent.get("before")
        if (
            not isinstance(before, list)
            or len(before) >= 3
            or set(before) != {endpoint for _, endpoint, _ in rows}
            or len(rows) != len(before)
            or any(name == session for name, _, _ in rows)
        ):
            raise Refused("account membership changed; fallback forbidden")
        if raw != pending.read_bytes():
            raise Refused("pending allocation changed during reconciliation")
        write_create_only_json(
            proof,
            {
                "schema": "colab-allocation-unavailable/v1",
                "classification": "ALLOCATION_UNAVAILABLE",
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                },
                "pending": {
                    "path": str(pending),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "record": intent,
                },
                "recovery": {"endpoint": None},
                "sessions_stdout": inventory,
                "endpoint_ever_assigned": False,
                "vm_compute_started": False,
            },
        )
        pending.unlink()


def dispatch_preferred(
    args: argparse.Namespace, value: dict[str, Any], ops: Path, broker: Path, cwd: Path, state: Path
) -> str:
    """Seal/admit both variants, then permit one proven pre-assignment fallback."""
    if getattr(args, "colab_control_nice0", False) or getattr(args, "small_bench_config", None):
        raise Refused("GPU preference requires ordinary control admission")
    if (
        value.get("timing_calibration")
        or value.get("operational_admission")
        or value.get("p1_attempt2")
    ):
        raise Refused(
            "hardware-specific/timing admissions require separate preference ratification"
        )
    packets = variants(value)
    names = [args.name, sibling_id(args.name)]
    states = [state, Path(str(state) + FALLBACK_SUFFIX)]
    paths = [ops / "manifests" / f"{name}.json" for name in names]
    logs = [ops / "logs" / f"{name}.log" for name in names]
    for name, packet, lane, path, log in zip(names, packets, states, paths, logs):
        require_resume_policy(packet)
        require_scientific_or_f5_plan_admission(packet)
        candidates = [
            ops / "dispatch" / name,
            log,
            path,
            path.with_suffix(".live.json"),
            lane,
            Path(packet["output_dir"]),
        ]
        collector = sandbox_collector_directory(packet)
        if collector is not None:
            candidates.append(collector)
        for candidate in candidates:
            if candidate.exists() or candidate.is_symlink():
                raise Refused(f"dispatch identity already used: {candidate}")
    if getattr(args, "dry_run", False):
        with tempfile.TemporaryDirectory(prefix="gpu-preference-validate-") as temporary:
            root = Path(temporary)
            for index, packet in enumerate(packets):
                path = root / f"{index}.json"
                write_create_only_json(path, packet)
                validate(broker, cwd, root / f"state-{index}", path, root / f"log-{index}")
                if read_json(path) != packet:
                    raise Refused("GPU preference manifest changed during validation")
        return (
            f"PASS_DRY_RUN {args.name} variants=H100,A100 allocations=0 live_accounting=NOT_CHECKED"
        )
    for subdir in ("logs", "state", "manifests", "dispatch"):
        (ops / subdir).mkdir(parents=True, exist_ok=True)
    # Reserve both identities before validation, and validate both before first submit.
    for name in names:
        (ops / "dispatch" / name).mkdir()
    for packet, path in zip(packets, paths):
        write_create_only_json(path, packet)
    seals = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    for path, lane, name in zip(paths, states, names):
        validate(broker, cwd, lane, path, ops / "logs" / f"{name}.validate")
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != seal for p, seal in zip(paths, seals)):
            raise Refused("GPU preference manifest changed during validation")
    write_create_only_json(
        ops / "dispatch" / args.name / "gpu-preference.json",
        {
            "schema": "colab-gpu-preference/v1",
            "preference": ["H100", "A100"],
            "variants": [
                {
                    "manifest": str(path),
                    "sha256": seal,
                    "name": name,
                    "gpu": packet["requirements"]["gpu"],
                    "budget": packet["budget"],
                    "receipt": str(ops / "dispatch" / name / "gpu-receipt.json"),
                }
                for name, path, seal, packet in zip(names, paths, seals, packets)
            ],
        },
    )
    seconds = float(os.environ.get("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "600"))
    poll = float(os.environ.get("OPS_ALLOC_POLL_INTERVAL_SECONDS", "1"))
    if not math.isfinite(seconds) or not 0 < seconds <= 600 or not math.isfinite(poll) or poll <= 0:
        raise Refused("invalid allocation wait bounds")
    with allocation_lock(ops / "alloc.lock", seconds) as deadline:
        for index, (packet, path, lane, log, name) in enumerate(
            zip(packets, paths, states, logs, names)
        ):
            if any(
                hashlib.sha256(p.read_bytes()).hexdigest() != seal for p, seal in zip(paths, seals)
            ):
                raise Refused("GPU preference manifest changed after admission")
            qualified(broker, ops, deadline)
            qualified(broker, ops, deadline, ref=packet["commit"])
            evidence = ops / "dispatch" / name / f"{name}.accounting.json"
            live = refresh_assignments(path, evidence, deadline)
            validate(broker, cwd, lane, live, ops / "logs" / f"{name}.live.validate", deadline)
            if (
                hashlib.sha256(live.read_bytes()).hexdigest()
                != read_json(evidence)["manifest_sha256"]
            ):
                raise Refused("refreshed preference manifest changed after validation")
            start(
                broker,
                cwd,
                lane,
                live,
                log,
                name,
                deadline,
                gpu_receipt=ops / "dispatch" / name / "gpu-receipt.json",
            )
            outcome = wait_allocation(Path(packet["output_dir"]), log, deadline, poll)
            if outcome == "failed":
                while time.monotonic() < deadline and not re.search(
                    r"^EXIT=\d+$", log.read_text(), re.MULTILINE
                ):
                    time.sleep(min(poll, max(0, deadline - time.monotonic())))
            # Never use the legacy accounting retry for a preference packet.
            if outcome == "retry":
                note(log, "OPS_ALLOCATION_RETRY_REFUSED GPU preference post-endpoint failure")
                outcome = "failed"
            if (
                index == 0
                and outcome == "failed"
                and allocation_unavailable(Path(packet["output_dir"]), log)
            ):
                reconcile_unavailable(
                    broker, live, lane, log, ops / "dispatch" / name / "unavailable.json", deadline
                )
                note(log, f"OPS_GPU_FALLBACK preadmitted={paths[1]} allocation_unavailable=true")
                continue
            break
    ensure_watcher(ops, broker)
    return (
        f"DISPATCHED {args.name} requested_gpu={packet['requirements']['gpu']} "
        f"allocation={outcome} log={log}"
    )


def dispatch(args: argparse.Namespace) -> str:
    """Validate first, then serialize one allocation and at most one exact retry."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.name):
        raise Refused("NAME must contain only letters, digits, dot, underscore, hyphen")
    ops = Path(os.environ.get("OPS_DIR", OPS)).resolve()
    manifest, broker, cwd = args.manifest.resolve(), args.broker.resolve(), args.cwd.resolve()
    state = (args.state_dir or ops / "state" / args.name).resolve()
    dispatch_txt = args.dispatch_txt or manifest.parent / "DISPATCH.txt"
    if dispatch_txt.is_file():
        blocked = [
            line
            for line in dispatch_txt.read_text().splitlines()
            if BLOCK.search(line) and "live inventory" not in line.lower()
        ]
        if blocked:
            raise Refused(f"{dispatch_txt} has BLOCK/HOLD instructions: {blocked}")
    original_bytes = manifest.read_bytes()
    value = json.loads(original_bytes)
    small_config = getattr(args, "small_bench_config", None)
    if small_config is not None:
        from tools.small_bench_operator import configuration

        admitted = configuration(small_config)
        if (
            admitted["job_id"] != args.name
            or admitted["manifest"]["sha256"] != hashlib.sha256(original_bytes).hexdigest()
            or Path(admitted["cwd"]).resolve() != broker
        ):
            raise Refused("small benchmark package identity differs")
    if not isinstance(value, dict):
        raise Refused("manifest must be an object")
    require_resume_policy(value)
    no_automatic_retry = require_timing_calibration_retry_policy(value)
    control_nice0 = bool(getattr(args, "colab_control_nice0", False))
    control_admission: Path | None = None
    if control_nice0:
        control = value.get("colab_control_nice0")
        if not isinstance(control, dict):
            raise Refused("--colab-control-nice0 requires manifest colab_control_nice0 admission")
        if control.get("policy_id") != CONTROL_POLICY_ID:
            raise Refused("Colab control policy_id is not the admitted v1 identity")
        if control.get("ruling_sha256") != CONTROL_RULING_SHA256:
            raise Refused("Colab control ruling SHA256 is not the binding ruling")
        if control.get("overlap_status") not in {
            "PREPARED_NOT_QUALIFIED_FOR_OVERLAP",
            "QUALIFIED_FOR_OVERLAP",
        }:
            raise Refused("Colab control overlap state is not recognized")
        require_overlap_canary(value, original_bytes, broker=broker)
        require_no_control_priority_drift()
        if value.get("p1_attempt2", {}).get("ruling_sha256") == P1_ATTEMPT2_RULING:
            raise Refused("P1 attempt-2 remains a quiet-host control window")
    if (
        (
            value.get("requirements", {}).get("gpu") in {"A100", "H100"}
            or any(
                gpu in value.get("requirements", {}).get("allowed_gpus", [])
                for gpu in ("A100", "H100")
            )
        )
        and value.get("canary", {}).get("mode") == "probe"
        and str(value.get("script", "")).startswith("research/gpu_sim_spike/")
    ):
        sys.path.insert(0, str(HERE.parents[1]))
        from research.gpu_sim_spike.native_compile import require_for_a100_canary

        try:
            require_for_a100_canary(value, broker)
        except (ValueError, OSError, KeyError) as exc:
            raise Refused(f"native CUDA compile gate: {exc}") from exc
    require_scientific_or_f5_plan_admission(value)
    if value.get("operational_admission", {}).get("schema") in {
        F5_PLAN_ADMISSION_SCHEMA,
        F5_TIMING_ADMISSION_SCHEMA,
    }:
        if args.name != value.get("job_id"):
            raise Refused("F5 dispatch name must equal the sealed job_id")
        declared_state = value.get("supervisor_state_dir")
        if not isinstance(declared_state, str):
            raise Refused("F5 admission requires a sealed supervisor_state_dir")
        declared_state_path = Path(declared_state).resolve()
        if state != declared_state_path:
            raise Refused("F5 state directory differs from the sealed admission")
        if manifest.parent not in declared_state_path.parents:
            raise Refused("F5 state directory must remain inside its package")
    if not isinstance(value.get("budget", {}).get("other_assignments", {}), dict):
        raise Refused("budget.other_assignments must be a map")
    output = value.get("output_dir")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise Refused("manifest output_dir must be absolute")
    output = str(Path(output).resolve())
    collector = sandbox_collector_directory(value)
    if collector is not None and (collector.exists() or collector.is_symlink()):
        raise Refused("collector evidence path already exists")
    if not isinstance(value.get("job_id"), str) or not value["job_id"]:
        raise Refused("manifest job_id must be nonempty")
    commit = value.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("manifest commit must be a full published SHA")
    qualified(broker, ops)
    qualified(broker, ops, ref=commit)
    if "prefer" in value.get("requirements", {}):
        return dispatch_preferred(args, value, ops, broker, cwd, state)
    if getattr(args, "dry_run", False):
        for candidate in (
            ops / "dispatch" / args.name,
            ops / "logs" / f"{args.name}.log",
            ops / "manifests" / f"{args.name}.json",
            state,
            Path(output),
        ):
            if candidate.exists():
                raise Refused(f"dispatch identity already used: {candidate}")
        with tempfile.TemporaryDirectory(prefix="dispatch-validate-") as temporary:
            validate(broker, cwd, Path(temporary) / "state", manifest, Path(temporary) / "log")
        return f"PASS_DRY_RUN {args.name} allocations=0 live_accounting=NOT_CHECKED"
    for subdir in ("logs", "state", "manifests", "dispatch"):
        (ops / subdir).mkdir(parents=True, exist_ok=True)
    # Create-only name reservation also fences two clients dispatched concurrently.
    (ops / "dispatch" / args.name).mkdir()
    log = ops / "logs" / f"{args.name}.log"
    if log.exists():
        raise Refused(f"log already exists (name reused): {log}")
    manifest = ops / "manifests" / f"{args.name}.json"
    with manifest.open("xb") as stream:
        stream.write(original_bytes)
    validate(broker, cwd, state, manifest, ops / "logs" / f"{args.name}.validate")
    seconds = float(os.environ.get("OPS_ALLOC_LOCK_TIMEOUT_SECONDS", "600"))
    poll = float(os.environ.get("OPS_ALLOC_POLL_INTERVAL_SECONDS", "1"))
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise Refused("allocation timeout must be positive and at most 600 seconds")
    if not math.isfinite(poll) or poll <= 0:
        raise Refused("allocation poll interval must be positive and finite")
    expected = value
    with allocation_lock(ops / "alloc.lock", seconds) as deadline:
        for attempt in range(2):
            name = args.name if attempt == 0 else f"{args.name}-retry-1"
            if read_json(manifest) != expected:
                raise Refused("manifest changed after validation")
            qualified(broker, ops, deadline)
            qualified(broker, ops, deadline, ref=commit)
            evidence = ops / "dispatch" / args.name / f"{name}.accounting.json"
            live_manifest = refresh_assignments(manifest, evidence, deadline)
            validate(
                broker, cwd, state, live_manifest, ops / "logs" / f"{name}.live.validate", deadline
            )
            if (
                hashlib.sha256(live_manifest.read_bytes()).hexdigest()
                != read_json(evidence)["manifest_sha256"]
            ):
                raise Refused("refreshed manifest changed after validation")
            if control_nice0:
                require_no_control_priority_drift()
                control_admission = ops / "colab-control-admissions" / f"{name}.json"
            start(
                broker,
                cwd,
                state,
                live_manifest,
                log,
                name,
                deadline,
                control_nice0=control_nice0,
                control_admission=control_admission,
                **({"small_bench_config": small_config} if small_config is not None else {}),
            )
            outcome = wait_allocation(Path(output), log, deadline, poll)
            if outcome == "retry" and no_automatic_retry:
                note(log, "OPS_ALLOCATION_RETRY_REFUSED timing_calibration.no_automatic_retry=true")
                raise Refused(
                    "timing_calibration.no_automatic_retry forbids retry after accounting failure"
                )
            if attempt == 1 and outcome == "retry":
                note(log, "OPS_ALLOCATION_RETRY_EXHAUSTED")
                outcome = "failed"
            if outcome != "retry":
                break
            retry_id = value["job_id"] + "-retry-1"
            if len(retry_id) > 64:
                digest = hashlib.sha256(value["job_id"].encode()).hexdigest()[:8]
                retry_id = f'{value["job_id"][:47]}-{digest}-retry-1'
            retry = dict(value, output_dir=f"{output}-retry-1", job_id=retry_id)
            if value.get("p1_attempt2", {}).get("ruling_sha256") == P1_ATTEMPT2_RULING:
                window = dict(value["p1_attempt2_control_window"])
                for field in ("root_record", "registration_evidence"):
                    path = Path(window[field])
                    window[field] = str(
                        path.with_name(path.stem + "-allocation-retry-1" + path.suffix)
                    )
                retry["p1_attempt2_control_window"] = window
            expected = retry
            manifest = ops / "manifests" / f"{name}-retry-1.json"
            if Path(retry["output_dir"]).exists():
                raise Refused("retry output_dir already exists")
            with manifest.open("x") as stream:
                json.dump(retry, stream, indent=2, sort_keys=True)
                stream.write("\n")
            output = retry["output_dir"]
            state = Path(f"{state}-retry-1")
            log = ops / "logs" / f"{args.name}-retry-1.log"
            note(
                ops / "logs" / f"{args.name}.log",
                f"OPS_ALLOCATION_RETRY manifest={manifest} output_dir={output} log={log}",
            )
            validate(
                broker,
                cwd,
                state,
                manifest,
                ops / "logs" / f"{args.name}-retry-1.validate",
                deadline,
            )
    ensure_watcher(ops, broker)
    return f"DISPATCHED {args.name} allocation={outcome} log={log}"


def main() -> int:
    """Keep CLI compatible with dispatch.sh's existing interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?")
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--dispatch-txt", type=Path)
    parser.add_argument(
        "--broker", type=Path, default=Path("/Users/josenunez/Projects/ml/snake-dqn")
    )
    parser.add_argument("--cwd", type=Path, default=Path("/Users/josenunez/Projects/ml/snake-dqn"))
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--colab-control-nice0",
        action="store_true",
        help="start only an admitted qualified Colab control tree at inherited absolute nice 0",
    )
    parser.add_argument(
        "--chain-manifest",
        type=Path,
        action="append",
        default=[],
        help="additional already-qualified A100 main packet for one retained chain",
    )
    parser.add_argument("--retention-minutes", type=float)
    parser.add_argument("--retention-units", type=float)
    parser.add_argument("--retention-acceptance", type=Path)
    parser.add_argument("--boundary-admission", type=Path)
    parser.add_argument("--review-token")
    parser.add_argument("--small-bench-config", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate without reserving or allocating"
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    try:
        if args.dry_run and (args.chain_manifest or args.retention_acceptance is not None):
            raise Refused("dry-run supports only the ordinary single-job path")
        if args.colab_control_nice0 and (
            args.retention_acceptance is not None or args.chain_manifest
        ):
            raise Refused(
                "nice-0 control admission supports only the qualified single-job run path"
            )
        if args.retention_acceptance is not None:
            if args.boundary_admission is None or args.review_token is None:
                raise Refused("retention acceptance requires boundary admission and review token")
            print(dispatch_retention_acceptance(args))
        elif args.name is None or args.manifest is None:
            raise Refused("ordinary dispatch requires NAME and MANIFEST")
        elif args.chain_manifest:
            if args.retention_minutes is None or args.retention_units is None:
                raise Refused("retained chain requires --retention-minutes and --retention-units")
            print(dispatch_chain(args))
        elif args.retention_minutes is not None or args.retention_units is not None:
            raise Refused("retention arguments require --chain-manifest")
        else:
            print(dispatch(args))
    except (OSError, ValueError, TypeError, TimeoutError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
