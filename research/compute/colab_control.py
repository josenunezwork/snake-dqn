"""Positive identity checks for the nice-0 qualified Colab control exception.

This is deliberately a classifier, not an admission mechanism and not a
process controller.  The factory and the operator renicer use the same result:
only a recorded broker root and its exact, bounded transport children are
exempt.  Every other descendant remains visible to the ordinary process policy.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from importlib.machinery import PathFinder
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .factory_companion_argv import native_argv

POLICY_ID = "factory-colab-control-nice0/v1"
RULING_SHA256 = "2ec03d85be07d4d9264e8aba105567ee23b40e364313c0da91fd57c215b0d64a"
SUPERVISOR_MODULE = "research.colab_cli.supervisor"
ADMISSION_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-control-admissions"
)


# Non-executing module resolution in the exact interpreter/site environment.
# -I plus the explicit broker path mirrors the scrubbed launch's -m sys.path.
IMPORT_PROBE_PROGRAM = """import importlib.machinery as m,json,sys
root=sys.argv[1]
sys.path.insert(0,root)
r=m.PathFinder.find_spec('research',sys.path)
p=list(r.submodule_search_locations or ()) if r else []
c=m.PathFinder.find_spec('research.colab_cli',p) if p else None
q=list(c.submodule_search_locations or ()) if c else []
s=m.PathFinder.find_spec('research.colab_cli.supervisor',q) if q else None
print(json.dumps({'research_origin':r.origin if r else 'missing',
'cli_origin':c.origin if c else None,'supervisor_origin':s.origin if s else None}))
"""


class IdentitySensorError(RuntimeError):
    """A live process which must be checked lacks a required native observation."""


class IdentityMismatch(ValueError):
    """Readable identity evidence differs; ordinary policy applies."""


@dataclass(frozen=True)
class AdmissionRecord:
    """The immutable facts which bind one launched control tree."""

    path: Path
    pid: int
    birth: str
    executable: str
    interpreter: str
    import_root: str
    module: str
    argv: tuple[str, ...]
    source_closure: object
    qualified_broker_source_closure: str
    cli_executable: str
    cli_sha256: str
    cli_interpreter: str
    cli_package_closure: object
    oauth_interpreter: str
    mint_program_sha256: str
    admission_path: str
    admission_sha256: str
    job_id: str
    state_path: str
    output_path: str
    pane_pid: int | None
    pane_birth: str | None
    pane_cwd: str | None
    pane_argv: tuple[str, ...] | None
    qualification_record: str
    qualification_record_sha256: str
    operational_root: str
    dispatch_source_closure: object
    operational_pin: str
    manifest: str
    manifest_sha256: str
    git_executable: str = ""
    git_sha256: str = ""
    interpreter_native: str = ""
    cli_interpreter_native: str = ""
    oauth_interpreter_native: str = ""


@dataclass(frozen=True)
class ControlIdentity:
    """A process positively identified as a permitted control role."""

    pid: int
    birth: str
    role: str
    root_pid: int
    admission_path: str


def _timeout(cap: float) -> float:
    """Share the factory's existing full-observation deadline without importing it."""
    observer = sys.modules.get("research.bl_labels.factory_persistence")
    remaining = getattr(observer, "observation_timeout", None)
    return min(cap, remaining()) if callable(remaining) else cap


def _sha256(path: Path) -> str:
    _timeout(10)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            _timeout(10)
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"control admission has no absolute {name}")
    return str(Path(value).resolve())


def _absolute_token(value: object, name: str) -> str:
    """Keep an absolute invocation token; resolving a venv Python loses its ABI."""
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"control admission has no absolute {name}")
    return value


def _record(path: Path, raw: Mapping[str, Any]) -> AdmissionRecord:
    if raw.get("policy_id") != POLICY_ID:
        raise ValueError("control admission policy differs")
    if raw.get("ruling_sha256") != RULING_SHA256:
        raise ValueError("control admission ruling digest is missing")
    if not isinstance(raw.get("operational_pin"), str) or not re.fullmatch(
        r"[0-9a-f]{40}", raw["operational_pin"]
    ):
        raise ValueError("control admission operational pin is missing")
    if (
        not isinstance(raw.get("qualified_broker_source_closure"), str)
        or len(raw["qualified_broker_source_closure"]) != 40
    ):
        raise ValueError("control admission broker tree is missing")
    pid = raw.get("pid")
    if type(pid) is not int or pid <= 0 or not raw.get("birth"):
        raise ValueError("control admission PID/birth is missing")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise ValueError("control admission argv is invalid")
    for name in ("manifest_sha256", "admission_sha256", "job_id"):
        if not isinstance(raw.get(name), str) or not raw[name]:
            raise ValueError(f"control admission {name} is missing")
    for name in ("manifest_sha256", "admission_sha256", "qualification_record_sha256"):
        if not isinstance(raw.get(name), str) or len(raw[name]) != 64:
            raise ValueError(f"control admission {name} is invalid")
    for name in ("cli_sha256", "mint_program_sha256"):
        if not isinstance(raw.get(name), str) or len(raw[name]) != 64:
            raise ValueError(f"control admission {name} is invalid")
    state_path = _absolute(raw.get("state_path"), "state_path")
    output_path = _absolute(raw.get("output_path"), "output_path")
    pane = raw.get("pane")
    if pane is not None and (
        not isinstance(pane, dict)
        or type(pane.get("pid")) is not int
        or not pane.get("birth")
        or pane.get("executable") != "/bin/bash"
        or not isinstance(pane.get("cwd"), str)
        or not isinstance(pane.get("argv"), list)
        or not all(isinstance(item, str) for item in pane["argv"])
    ):
        raise ValueError("control pane identity is invalid")
    return AdmissionRecord(
        path=path,
        pid=pid,
        birth=str(raw["birth"]),
        executable=_absolute(raw.get("executable"), "executable"),
        interpreter=_absolute_token(raw.get("interpreter"), "interpreter"),
        import_root=_absolute(raw.get("import_root"), "import_root"),
        module=str(raw.get("module") or ""),
        argv=tuple(argv),
        source_closure=raw.get("source_closure"),
        qualified_broker_source_closure=raw["qualified_broker_source_closure"],
        cli_executable=_absolute_token(raw.get("cli_executable"), "cli_executable"),
        cli_sha256=str(raw.get("cli_sha256") or ""),
        cli_interpreter=_absolute_token(raw.get("cli_interpreter"), "cli_interpreter"),
        cli_package_closure=raw.get("cli_package_closure"),
        oauth_interpreter=_absolute_token(raw.get("oauth_interpreter"), "oauth_interpreter"),
        mint_program_sha256=str(raw.get("mint_program_sha256") or ""),
        admission_path=_absolute(raw.get("admission_path"), "admission_path"),
        admission_sha256=raw["admission_sha256"],
        job_id=str(raw.get("job_id") or ""),
        state_path=state_path,
        output_path=output_path,
        pane_pid=None if pane is None else pane["pid"],
        pane_birth=None if pane is None else str(pane["birth"]),
        pane_cwd=None if pane is None else _absolute(pane["cwd"], "pane cwd"),
        pane_argv=None if pane is None else tuple(pane["argv"]),
        qualification_record=_absolute(raw.get("qualification_record"), "qualification record"),
        qualification_record_sha256=raw["qualification_record_sha256"],
        operational_root=_absolute(raw.get("operational_root"), "operational root"),
        dispatch_source_closure=raw.get("dispatch_source_closure"),
        operational_pin=raw["operational_pin"],
        manifest=_absolute(raw.get("manifest"), "manifest"),
        manifest_sha256=raw["manifest_sha256"],
        git_executable=str(raw.get("git_executable", "")),
        git_sha256=str(raw.get("git_sha256", "")),
        interpreter_native=str(raw.get("interpreter_native", raw["interpreter"])),
        cli_interpreter_native=str(raw.get("cli_interpreter_native", raw["cli_interpreter"])),
        oauth_interpreter_native=str(raw.get("oauth_interpreter_native", raw["oauth_interpreter"])),
    )


def load_admissions(directory: Path = ADMISSION_DIR) -> dict[tuple[int, str], AdmissionRecord]:
    """Load complete create-only admission records, refusing malformed records.

    A stale record remains useful as a *relevance seed*, but cannot admit a
    reused PID because the birth identity is always checked below.
    """
    if not directory.exists():
        return {}
    records: dict[tuple[int, str], AdmissionRecord] = {}
    # Dispatch first writes a pending request.  It is deliberately not a root
    # admission and must never turn a harmless preflight into a monitor fault.
    for path in sorted(directory.glob("*.root.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("record is not an object")
            record = _record(path.resolve(), raw)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise IdentitySensorError(f"unreadable control admission {path.name}") from error
        key = (record.pid, record.birth)
        if key in records:
            raise IdentitySensorError("duplicate control admission PID/birth")
        records[key] = record
    return records


def normalize_interpreter(argv: Iterable[str], invocation: str, native: str) -> tuple[str, ...]:
    """Preserve every argument while accepting Darwin's bound framework exec."""
    words = tuple(argv)
    if words and words[0] in {invocation, native}:
        return (invocation, *words[1:])
    return words


def interpreter_native(path: Path) -> str:
    """Observe the actual executable of this exact interpreter, without work."""
    program = (
        "import ctypes,os; l=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True); "
        "f=l.proc_pidpath; f.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32]; "
        "f.restype=ctypes.c_int; b=ctypes.create_string_buffer(4096); "
        "n=f(os.getpid(),b,len(b)); assert n>0; print(os.fsdecode(b.value))"
    )
    value = subprocess.check_output(
        [str(path), "-I", "-c", program], text=True, timeout=_timeout(5)
    ).strip()
    if not Path(value).is_absolute() or not Path(value).is_file():
        raise OSError("interpreter native executable unavailable")
    return str(Path(value).resolve())


def verify_witness(record: AdmissionRecord) -> None:
    """Verify a live matching root; dead-job evidence never blocks other jobs."""
    path = record.path
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_parent = path.with_name(path.name[: -len(".root.json")] + ".json").resolve()
    if (
        Path(record.admission_path) != expected_parent
        or _sha256(expected_parent) != record.admission_sha256
    ):
        raise IdentityMismatch("root record lacks its create-only dispatch admission witness")
    parent = json.loads(expected_parent.read_text(encoding="utf-8"))
    bound = (
        "policy_id",
        "ruling_sha256",
        "operational_pin",
        "qualified_broker_source_closure",
        "source_closure",
        "broker",
        "launcher",
        "module",
        "executable",
        "interpreter",
        "import_root",
        "argv",
        "cli_executable",
        "cli_interpreter",
        "cli_sha256",
        "cli_package_closure",
        "oauth_interpreter",
        "mint_program_sha256",
        "manifest",
        "manifest_sha256",
        "job_id",
        "state_path",
        "output_path",
        "qualification_record",
        "qualification_record_sha256",
        "operational_root",
        "dispatch_source_closure",
        "git_executable",
        "git_sha256",
        "interpreter_native",
        "cli_interpreter_native",
        "oauth_interpreter_native",
    )
    if not isinstance(parent, dict) or any(raw.get(key) != parent.get(key) for key in bound):
        raise IdentityMismatch("root record differs from its dispatch admission")
    pane = raw.get("pane")
    if pane is not None and pane.get("argv") != parent.get("pane_argv"):
        raise IdentityMismatch("root pane differs from its pending exact argv")
    if _sha256(Path(record.manifest)) != record.manifest_sha256:
        raise IdentityMismatch("root manifest bytes differ from dispatch admission")
    bootstrap = raw.get("bootstrap")
    expected_bootstrap = [
        record.interpreter,
        str(Path(record.operational_root) / "tools/ops/colab_control_launch.py"),
        "--admission",
        record.admission_path,
        "--broker",
        record.import_root,
        "--launcher",
        str(Path(record.import_root) / "research/colab_cli/launch.sh"),
        "--",
        str(Path(record.import_root) / "research/colab_cli/launch.sh"),
        "--control-nice0",
        *_expected_supervisor_argv(record)[3:],
    ]
    if (
        not isinstance(bootstrap, dict)
        or normalize_interpreter(
            bootstrap.get("argv", ()), record.interpreter, record.interpreter_native
        )
        != tuple(expected_bootstrap)
        or parent.get("bootstrap_argv") != expected_bootstrap
        or bootstrap.get("pid") != record.pid
        or bootstrap.get("birth") != record.birth
        or bootstrap.get("cwd") != record.import_root
        or not _matches_executable(
            str(bootstrap.get("executable", "")), record.interpreter, record.interpreter_native
        )
    ):
        raise IdentityMismatch("root bootstrap differs from the exact native dispatch witness")


def candidate_pids(
    rows: Iterable[Mapping[str, Any]], records: Mapping[tuple[int, str], AdmissionRecord]
) -> set[int]:
    """Return roots and all currently visible descendants needing native checks.

    The tree is expanded before identity classification.  Therefore a readable
    mismatch never hides a generic Python, shell, or native-tool descendant.
    """
    table = {int(row["pid"]): row for row in rows if isinstance(row.get("pid"), int)}
    result = {
        pid for pid, birth in records if pid in table and table[pid].get("birth", birth) == birth
    }
    result.update(
        record.pane_pid
        for record in records.values()
        if record.pid in result
        and record.pane_pid in table
        and table.get(record.pid, {}).get("ppid") == record.pane_pid
        and table[record.pane_pid].get("birth") == record.pane_birth
    )
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid not in result and row.get("ppid") in result:
                result.add(pid)
                changed = True
    return result


def native_cwd(pid: int) -> str:
    """Read a Darwin process cwd through lsof; this is not a record claim."""
    output = subprocess.check_output(
        ["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], text=True, timeout=_timeout(3)
    )
    paths = [line[1:] for line in output.splitlines() if line.startswith("n")]
    if len(paths) != 1 or not Path(paths[0]).is_absolute():
        raise OSError("native cwd unavailable")
    return str(Path(paths[0]).resolve())


def _import_resolution_matches(record: AdmissionRecord) -> bool:
    observed = json.loads(
        subprocess.check_output(
            [record.interpreter, "-I", "-c", IMPORT_PROBE_PROGRAM, record.import_root],
            text=True,
            timeout=_timeout(5),
        )
    )
    return observed == {
        "research_origin": None,
        "cli_origin": str(Path(record.import_root) / "research/colab_cli/__init__.py"),
        "supervisor_origin": str(Path(record.import_root) / "research/colab_cli/supervisor.py"),
    }


def _source_matches(record: AdmissionRecord, cwd: str) -> bool:
    """Check that the actual import root exposes the pinned supervisor bytes."""
    qualification_path = Path(record.qualification_record)
    if _sha256(qualification_path) != record.qualification_record_sha256:
        return False
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    if (
        not isinstance(qualification, dict)
        or qualification.get("schema") != "snake-dqn-colab-cli-qualified/v1"
        or qualification.get("tree_hash") != record.qualified_broker_source_closure
    ):
        return False
    dispatch = record.dispatch_source_closure
    if not isinstance(dispatch, dict) or not dispatch:
        return False
    for path, digest in dispatch.items():
        _timeout(10)
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            return False
        if _sha256(Path(path)) != digest:
            return False
        try:
            relative = Path(path).resolve().relative_to(record.operational_root)
        except ValueError:
            return False
        pinned = subprocess.check_output(
            [
                "/usr/bin/git",
                "-C",
                record.operational_root,
                "show",
                f"{record.operational_pin}:{relative.as_posix()}",
            ],
            timeout=_timeout(3),
        )
        if hashlib.sha256(pinned).hexdigest() != digest:
            return False
    root = Path(cwd).resolve()
    if root != Path(record.import_root):
        return False
    # The qualified broker uses a namespace parent. A new initializer/module
    # could execute outside the frozen transport subtree before the supervisor.
    parent = PathFinder.find_spec("research", [str(root)])
    if (
        parent is None
        or parent.origin is not None
        or list(parent.submodule_search_locations or ()) != [str(root / "research")]
    ):
        return False
    if not _import_resolution_matches(record):
        return False
    source = root / "research" / "colab_cli" / "supervisor.py"
    if not source.is_file():
        return False
    closure = record.source_closure
    if isinstance(closure, dict) and closure:
        actual_inventory = {
            str(candidate.relative_to(root))
            for candidate in (root / "research/colab_cli").rglob("*")
            if candidate.is_file() and "__pycache__" not in candidate.parts
        }
        if set(closure) != actual_inventory:
            return False
        # A manifest can bind just the qualified tree or a wider source closure.
        for relative, expected in closure.items():
            _timeout(10)
            if (
                not isinstance(relative, str)
                or not isinstance(expected, str)
                or len(expected) != 64
            ):
                return False
            candidate = (root / relative).resolve()
            if (
                root not in candidate.parents
                or not candidate.is_file()
                or _sha256(candidate) != expected
            ):
                return False
        required = {
            "research/colab_cli/supervisor.py",
            "research/colab_cli/source.py",
            "research/colab_cli/release.py",
        }
        if not required <= set(closure):
            return False
        try:
            tree = subprocess.check_output(
                ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD:research/colab_cli"],
                text=True,
                timeout=_timeout(3),
            ).strip()
            dirty = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "research/colab_cli",
                ],
                capture_output=True,
                text=True,
                timeout=_timeout(3),
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise OSError("cannot verify qualified broker tree") from error
        return tree == record.qualified_broker_source_closure and not dirty
    # A tree id is checked against the actual checkout *and* a clean worktree;
    # an object reachable from HEAD alone does not bind imported worktree bytes.
    if isinstance(closure, str) and len(closure) >= 12:
        try:
            observed = subprocess.check_output(
                ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD:research/colab_cli"],
                text=True,
                timeout=_timeout(3),
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return False
        try:
            dirty = subprocess.run(
                [
                    "/usr/bin/git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "research/colab_cli",
                ],
                capture_output=True,
                text=True,
                timeout=_timeout(3),
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise OSError("cannot verify qualified broker worktree") from error
        return observed == closure and not dirty
    return False


def _supervisor_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    return (
        normalize_interpreter(argv, record.interpreter, record.interpreter_native)
        == (
            record.interpreter,
            "-m",
            SUPERVISOR_MODULE,
            "--state-dir",
            record.state_path,
            "--colab",
            record.cli_executable,
            "run",
            record.manifest,
        )
        and record.module == SUPERVISOR_MODULE
    )


def _expected_supervisor_argv(record: AdmissionRecord) -> tuple[str, ...]:
    """Recover the full supervisor command from the exact admitted wrapper."""
    return record.argv[2:] if _wrapper_argv(record.argv, record) else record.argv


def _wrapper_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    return (
        len(argv) == 11
        and argv[0] == "/usr/bin/caffeinate"
        and argv[1] == "-is"
        and _supervisor_argv(argv[2:], record)
    )


def _cli_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    """Accept the broker's fixed colab CLI family, never a basename substring."""
    argv = normalize_interpreter(argv, record.cli_interpreter, record.cli_interpreter_native)
    if len(argv) >= 2 and argv[0] == record.cli_interpreter and argv[1] == record.cli_executable:
        argv = (record.cli_executable, *argv[2:])
    if len(argv) < 6 or argv[0] != record.cli_executable:
        return False
    if _sha256(Path(record.cli_executable)) != record.cli_sha256:
        return False
    closure = record.cli_package_closure
    if not isinstance(closure, dict) or not closure:
        return False
    for path, digest in closure.items():
        if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(digest, str):
            return False
        if len(digest) != 64 or _sha256(Path(path)) != digest:
            return False
    common = Path(os.path.commonpath(list(closure)))
    if common.is_dir():
        installed = {
            str(path)
            for path in common.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if installed != set(closure):
            return False
    config = str(Path(record.state_path) / record.job_id / "sessions.json")
    if argv[1:5] != ("--auth", "oauth2", "--config", config):
        return False
    action = argv[5]
    rest = argv[6:]

    def session(value: str) -> bool:
        return _session_record(value, record) is not None

    if action in {"version", "sessions", "usage"}:
        return not rest
    if action == "new":
        return (
            len(rest) in {2, 4}
            and rest[0] == "-s"
            and session(rest[1])
            and (len(rest) == 2 or (rest[2] == "--gpu" and bool(rest[3])))
        )
    if action == "stop":
        return len(rest) == 2 and rest[0] == "-s" and session(rest[1])
    if action == "exec":
        valid = (
            len(rest) == 6
            and rest[0] == "-s"
            and session(rest[1])
            and rest[2] == "-f"
            and rest[4] == "--timeout"
            and Path(rest[3]).is_absolute()
        )
        if not valid:
            return False
        try:
            seconds = float(rest[5])
            return math.isfinite(seconds) and seconds > 0
        except ValueError:
            return False
    if action in {"upload", "download"}:
        return (
            len(rest) == 4
            and rest[0] == "-s"
            and session(rest[1])
            and all(item.startswith("/") for item in rest[2:])
        )
    return False


def _session_record(session: str, record: AdmissionRecord) -> dict[str, Any] | None:
    """Bind session operands to the broker's durable intent/active record."""
    if not re.fullmatch(re.escape(f"snake-{record.job_id}-") + r"[0-9a-f]{12}", session):
        return None
    for kind in ("pending", "active"):
        path = Path.home() / ".config/colab-cli/snake" / f"{kind}-{session}.json"
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            continue
        if (
            isinstance(value, dict)
            and value.get("session") == session
            and value.get("config")
            == str(Path(record.state_path) / record.job_id / "sessions.json")
            and value.get("output") == record.output_path
        ):
            return value
    return None


def _transport_helper_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    """Recognize the bounded cli_guard helper by pinned script and operands."""
    if len(argv) not in {4, 5} or argv[:2] != (
        record.oauth_interpreter,
        str(Path(record.import_root) / "research/colab_cli/cli_guard.py"),
    ):
        return False
    action, args = argv[2], argv[3:]
    if action in {"--kernel-idle", "--runtime-ready"}:
        return (
            len(args) == 2
            and args[0] == str(Path(record.state_path) / record.job_id / "sessions.json")
            and _session_record(args[1], record) is not None
        )
    if action not in {"--recover-assignment", "--unassign-endpoint"} or len(args) != 1:
        return False
    account = Path.home() / ".config/colab-cli/snake"
    for path in (
        *account.glob(f"pending-snake-{record.job_id}-*.json"),
        *account.glob(f"active-snake-{record.job_id}-*.json"),
    ):
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            continue
        bound = _session_record(value.get("session", ""), record)
        field = "notebook_hash" if action == "--recover-assignment" else "endpoint"
        if bound and bound.get(field) == args[0]:
            return True
    return False


def _matches_executable(observed: str, *tokens: str) -> bool:
    try:
        return str(Path(observed).resolve()) in {str(Path(token).resolve()) for token in tokens}
    except OSError:
        return False


def _mint_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    argv = normalize_interpreter(argv, record.oauth_interpreter, record.oauth_interpreter_native)
    if len(argv) != 4 or argv[0] != record.oauth_interpreter or argv[1] != "-c":
        return False
    if hashlib.sha256(argv[2].encode()).hexdigest() != record.mint_program_sha256:
        return False
    try:
        payload, scopes = release_identity(Path(record.import_root))

        return (
            hashlib.sha256(argv[2].encode()).hexdigest() == record.mint_program_sha256
            and argv[2] == payload
            and json.loads(argv[3]) == list(scopes)
        )
    except (ImportError, json.JSONDecodeError):
        return False


def release_identity(root: Path) -> tuple[str, tuple[str, ...]]:
    """Read fixed literals from the pinned broker without executing its imports."""
    tree = ast.parse((root / "research/colab_cli/release.py").read_text())
    values = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    payload = ast.literal_eval(values["_MINT_PROGRAM"])
    scope_node = values["RELEASE_SCOPES"]
    if (
        not isinstance(scope_node, ast.Call)
        or not isinstance(scope_node.func, ast.Name)
        or scope_node.func.id != "frozenset"
        or len(scope_node.args) != 1
    ):
        raise ValueError("qualified release scopes are not fixed literals")
    scopes = ast.literal_eval(scope_node.args[0])
    if (
        not isinstance(payload, str)
        or not isinstance(scopes, set)
        or not all(isinstance(item, str) for item in scopes)
    ):
        raise ValueError("qualified release identity is malformed")
    return payload, tuple(sorted(scopes))


def _git_argv(argv: tuple[str, ...], record: AdmissionRecord, parent_role: str) -> bool:
    """Admit only existing bounded read-only source/identity Git operations."""
    if (
        not record.git_executable
        or not argv
        or argv[0] not in {"git", "/usr/bin/git", record.git_executable}
    ):
        return False
    if _sha256(Path(record.git_executable)) != record.git_sha256:
        return False
    args = argv[1:]
    if parent_role in {"bootstrap", "launch_verifier"}:
        checks = {
            ("-C", record.import_root, "rev-parse", "HEAD:research/colab_cli"),
            (
                "-C",
                record.import_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "research/colab_cli",
            ),
        }
        if args in checks:
            return True
        prefix = ("-C", record.operational_root, "show")
        if args[:3] != prefix or len(args) != 4:
            return False
        return args[3] in {
            f"{record.operational_pin}:{Path(path).relative_to(record.operational_root).as_posix()}"
            for path in record.dispatch_source_closure
        }
    if parent_role != "supervisor":
        return False
    manifest = json.loads(Path(record.manifest).read_text())
    commit = manifest.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    if args in {
        ("cat-file", "-t", commit),
        ("cat-file", "-t", f"{commit}:{manifest.get('script')}"),
        ("rev-parse", "--verify", commit + "^{commit}"),
        ("rev-parse", commit + "^{tree}"),
        ("ls-tree", "-r", "-z", commit),
    }:
        return True
    if len(args) == 3 and args[:2] == ("cat-file", "-s"):
        return bool(re.fullmatch(r"[0-9a-f]{40}", args[2]))
    if len(args) == 2 and args[0] == "show" and args[1].startswith(commit + ":"):
        relative = args[1][41:]
        return (
            bool(relative) and not Path(relative).is_absolute() and ".." not in Path(relative).parts
        )
    return False


def _backup_argv(argv: tuple[str, ...], record: AdmissionRecord) -> bool:
    """Bound the existing 60-second verified-export backup helper exactly."""
    argv = normalize_interpreter(argv, record.interpreter, record.interpreter_native)
    if (
        len(argv) != 5
        or argv[:3] != (record.interpreter, "-m", "research.colab_cli.backup")
        or argv[3] != str(Path(record.output_path) / "exports")
    ):
        return False
    manifest = json.loads(Path(record.manifest).read_text())
    backup = manifest.get("backup_dir")
    if not isinstance(backup, str) or not backup:
        return False
    assignment = json.loads(
        (Path(record.state_path) / record.job_id / "assignment.json").read_text()
    )
    generation = assignment.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
        return False
    return argv[4] == str(Path(backup).expanduser() / record.job_id / generation)


def annotate(
    rows: Iterable[Mapping[str, Any]],
    records: Mapping[tuple[int, str], AdmissionRecord],
    *,
    argv_reader: Callable[[int], tuple[str, ...]] = native_argv,
    cwd_reader: Callable[[int], str] = native_cwd,
    source_checker: Callable[[AdmissionRecord, str], bool] = _source_matches,
) -> dict[int, ControlIdentity]:
    """Classify exact currently admitted roles, failing closed for live candidates.

    ``rows`` must be the full process table.  Callers first use
    :func:`candidate_pids` when gathering native metadata, then pass those rows
    here.  A complete identity mismatch is ordinary classification; missing
    native argv/cwd/birth/executable for a control-tree candidate is a sensor
    fault.
    """
    table = {int(row["pid"]): row for row in rows if isinstance(row.get("pid"), int)}
    candidates = candidate_pids(table.values(), records)
    for pid in candidates:
        _timeout(10)
        row = table[pid]
        if not row.get("birth") or not row.get("executable"):
            raise IdentitySensorError(f"control candidate {pid} lacks birth/executable")
        if "argv_vector" not in row:
            try:
                row["argv_vector"] = argv_reader(pid)
            except OSError as error:
                raise IdentitySensorError(f"control candidate {pid} lacks native argv") from error
    result: dict[int, ControlIdentity] = {}
    roots: dict[int, AdmissionRecord] = {}
    for (pid, birth), record in records.items():
        row = table.get(pid)
        if row is None or str(row.get("birth")) != birth:
            continue  # stale/PID-reused record is a readable mismatch.
        argv = normalize_interpreter(
            row["argv_vector"], record.interpreter, record.interpreter_native
        )
        executable = str(Path(str(row["executable"])).resolve())
        raw = json.loads(record.path.read_text())
        if argv == normalize_interpreter(
            raw.get("bootstrap", {}).get("argv", ()), record.interpreter, record.interpreter_native
        ) and _matches_executable(executable, record.interpreter, record.interpreter_native):
            role = "bootstrap"
        elif (
            argv
            == (
                "/bin/bash",
                str(Path(record.import_root) / "research/colab_cli/launch.sh"),
                "--control-nice0",
                *_expected_supervisor_argv(record)[3:],
            )
            and executable == "/bin/bash"
        ):
            role = "launch_shell"
        elif executable != record.executable:
            # Darwin caffeinate forks its assertion helper and execs the utility
            # in the original PID. Preserve that exact native birth across exec.
            if (
                _matches_executable(executable, record.interpreter, record.interpreter_native)
                and argv == _expected_supervisor_argv(record)
                and _supervisor_argv(argv, record)
            ):
                role = "supervisor"
            else:
                continue
        elif argv == record.argv and _supervisor_argv(argv, record):
            role = "supervisor"
        elif argv == record.argv and _wrapper_argv(argv, record):
            role = "wrapper"
        else:
            continue
        try:
            verify_witness(record)
            cwd = cwd_reader(pid)
        except IdentityMismatch:
            continue
        except OSError as error:
            raise IdentitySensorError(f"control root {pid} lacks native cwd") from error
        try:
            source_matches = source_checker(record, cwd)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            raise IdentitySensorError(f"control root {pid} source identity unreadable") from error
        if not source_matches:
            continue
        roots[pid] = record
        result[pid] = ControlIdentity(pid, birth, role, pid, str(record.path))

    # The pane is an exact registered shell, never a generic shell exemption.
    # It is admitted only while its separately bound caffeinate root is live.
    for record in records.values():
        if record.pid not in result or record.pane_pid is None:
            continue
        row = table.get(record.pane_pid)
        if row is None or str(row.get("birth")) != record.pane_birth:
            continue
        if str(Path(str(row["executable"])).resolve()) != "/bin/bash":
            continue
        if tuple(row["argv_vector"]) != record.pane_argv:
            continue
        root_row = table.get(record.pid)
        if root_row is None or root_row.get("ppid") != record.pane_pid:
            continue
        try:
            if cwd_reader(record.pane_pid) != record.pane_cwd:
                continue
        except OSError as error:
            raise IdentitySensorError(f"control pane {record.pane_pid} lacks native cwd") from error
        result[record.pane_pid] = ControlIdentity(
            record.pane_pid,
            record.pane_birth,
            "pane",
            record.pid,
            str(record.path),
        )

    # The launch path uses exec, but accepting a directly recorded caffeinate
    # wrapper also allows the supervisor child.  No exited root expands a tree.
    changed = True
    while changed:
        changed = False
        for pid in sorted(candidates - set(result)):
            _timeout(10)
            row = table[pid]
            parent = result.get(row.get("ppid"))
            if parent is None:
                continue
            record = roots[parent.root_pid]
            argv = tuple(row["argv_vector"])
            executable = str(Path(str(row["executable"])).resolve())
            role: str | None = None
            if (
                parent.role in {"bootstrap", "launch_verifier"}
                and _matches_executable(executable, record.interpreter, record.interpreter_native)
                and normalize_interpreter(argv, record.interpreter, record.interpreter_native)
                == (record.interpreter, "-I", "-c", IMPORT_PROBE_PROGRAM, record.import_root)
            ):
                role = "import_probe"
            elif (
                parent.role == "supervisor"
                and executable == record.executable == "/usr/bin/caffeinate"
                and argv == record.argv
                and _wrapper_argv(argv, record)
            ):
                role = "wrapper"
            elif (
                parent.role in {"bootstrap", "launch_verifier", "supervisor"}
                and record.git_executable
                and _matches_executable(executable, record.git_executable)
                and _git_argv(argv, record, parent.role)
                and cwd_reader(pid) == record.import_root
            ):
                role = "source_git"
            elif (
                parent.role == "launch_shell"
                and _matches_executable(executable, record.interpreter, record.interpreter_native)
                and normalize_interpreter(argv, record.interpreter, record.interpreter_native)
                == (
                    record.interpreter,
                    str(Path(record.operational_root) / "tools/ops/colab_control_launch.py"),
                    "--verify-launch",
                    str(record.path),
                    "--pid",
                    str(record.pid),
                    "--",
                    *_expected_supervisor_argv(record)[3:],
                )
            ):
                role = "launch_verifier"
            elif (
                parent.role == "wrapper"
                and _matches_executable(executable, record.interpreter, record.interpreter_native)
                and normalize_interpreter(argv, record.interpreter, record.interpreter_native)
                == _expected_supervisor_argv(record)
            ):
                try:
                    if source_checker(record, cwd_reader(pid)):
                        role = "supervisor"
                except OSError as error:
                    raise IdentitySensorError(
                        f"control supervisor {pid} lacks native cwd"
                    ) from error
            elif parent.role == "supervisor":
                try:
                    cli_matches = _cli_argv(argv, record)
                except OSError as error:
                    raise IdentitySensorError(
                        f"control CLI {pid} installation identity unreadable"
                    ) from error
                if cli_matches and _matches_executable(
                    executable,
                    record.cli_executable,
                    record.cli_interpreter,
                    record.cli_interpreter_native,
                ):
                    # Native executable may be a shebang interpreter; the exact
                    # pinned entry script is already bound by argv0 plus its hash.
                    role = "cli"
                elif _mint_argv(argv, record) and _matches_executable(
                    executable, record.oauth_interpreter, record.oauth_interpreter_native
                ):
                    role = "mint"
                elif _transport_helper_argv(
                    normalize_interpreter(
                        argv, record.oauth_interpreter, record.oauth_interpreter_native
                    ),
                    record,
                ) and _matches_executable(
                    executable, record.oauth_interpreter, record.oauth_interpreter_native
                ):
                    role = "transport_helper"
                elif _backup_argv(argv, record) and _matches_executable(
                    executable, record.interpreter, record.interpreter_native
                ):
                    role = "backup"
            if role:
                result[pid] = ControlIdentity(
                    pid, str(row["birth"]), role, parent.root_pid, str(record.path)
                )
                changed = True
    return result
