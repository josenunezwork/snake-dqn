#!/usr/bin/env python3
"""Dispatch a qualified supervisor with bounded account allocation exclusion."""

from __future__ import annotations

import argparse
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

EXACT_ERROR = "AccountingError('assignment endpoints changed during allocation')"
BLOCK = re.compile(r"^\s*(?:BLOCK|HOLD)\b|RUNTIME BLOCKED|do not run this|PREPARED_NOT_DISPATCHED")
NICE_FLOOR = 15
P1_ATTEMPT2_RULING = "55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13"
CONTROL_INTERPRETER = Path("/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python")
CONTROL_OAUTH_INTERPRETER = Path("/Users/josenunez/.venvs/colab-cli/bin/python")


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
    args: list[str], *, deadline: float | None = None, **kwargs: Any
) -> subprocess.CompletedProcess:
    """Bound every subprocess while the account lock is held."""
    remaining = 60.0 if deadline is None else min(60.0, deadline - time.monotonic())
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


def validate(
    broker: Path, cwd: Path, state: Path, manifest: Path, log: Path, deadline: float | None = None
) -> None:
    """Run real validation; launch.sh --check would only print and return."""
    with log.open("x") as stream:
        command(
            [
                str(broker / "research/colab_cli/launch.sh"),
                "--state-dir",
                str(state),
                "--colab",
                str(Path.home() / ".local/bin/colab"),
                "validate",
                str(manifest),
            ],
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            deadline=deadline,
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
) -> None:
    """Submit one shell command with preserved argument boundaries and exit evidence."""
    log.touch(exist_ok=False)
    register(log.parent.parent, log, name)
    launcher = [
        str(broker / "research/colab_cli/launch.sh"),
        "--state-dir",
        str(state),
        "--colab",
        str(Path.home() / ".local/bin/colab"),
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
    body = (
        f"{{ cd {shlex.quote(str(cwd))} && {environment} {shlex.join(launcher)}; }} "
        f">> {shlex.quote(str(log))} 2>&1; rc=$?; "
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


def refresh_assignments(manifest: Path, evidence: Path, deadline: float) -> Path:
    """Snapshot live broker-owned prices into a create-only sibling under the lock."""
    result = command(
        [str(Path.home() / ".local/bin/colab"), "--auth", "oauth2", "sessions"],
        deadline=deadline,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.startswith("[") or line.startswith("[colab]"):
            continue
        match = re.fullmatch(r"\[([^\]]+)\]\s+(\S+)\s+\| Hardware:\s*([^|]+?)(?:\s+\|.*)?", line)
        if match is None or not match[3].strip():
            raise Refused("unrecognized account session row")
        rows.append(match.groups())
    if not rows and "No active sessions found on server." not in result.stdout:
        raise Refused("unrecognized account sessions output")
    if rows and "No active sessions found on server." in result.stdout:
        raise Refused("contradictory account sessions output")
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
        if hardware.upper() not in ("L4", "CPU", "A100"):
            raise Refused(f"unregistered live hardware: {hardware}")
        rates[endpoint] = rate
    original = manifest.read_bytes()
    value = json.loads(original)
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
            value.get("requirements", {}).get("gpu") == "A100"
            or "A100" in value.get("requirements", {}).get("allowed_gpus", [])
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
    if value.get("scientific_admission") is False:
        raise Refused("manifest scientific_admission=false")
    if not isinstance(value.get("budget", {}).get("other_assignments", {}), dict):
        raise Refused("budget.other_assignments must be a map")
    output = value.get("output_dir")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise Refused("manifest output_dir must be absolute")
    output = str(Path(output).resolve())
    if not isinstance(value.get("job_id"), str) or not value["job_id"]:
        raise Refused("manifest job_id must be nonempty")
    commit = value.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("manifest commit must be a full published SHA")
    qualified(broker, ops)
    qualified(broker, ops, ref=commit)
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
