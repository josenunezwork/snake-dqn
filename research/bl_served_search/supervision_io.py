"""Prospective supervision evidence bindings and deliberately closed live gate.

An inventory is a declaration, not proof of the source loaded by an existing
interpreter or complete operation accounting. No function attaches to a process,
patches it, sends it a signal, or changes its remote control path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path
from typing import Any

from research.bl_paced_identity.record import export_json, read_export

POLICY = "classified-supervised-v2"
RULING_SHA256 = "83e47ff2b3cfddba0d9ee4a6ec2bab48efb645259857d7e81ea5fd23459f7426"
RULING_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/" "astra-guard-supervisor/RULING.md"
)
BLOCKER = (
    "remote_supervision admission unavailable: the current supervisor has no "
    "prospectively installed complete launch/exec/exit, final per-child CPU and "
    "bounded application-payload observer. Darwin kqueue does not enumerate "
    "unseen fork children; proc_pid_rusage races the existing parent's reap. "
    "A sampled inventory or terminal CLI operation tags cannot replace this "
    "evidence. Do not attach, restart or instrument TRAIN-01 attempt-07. "
    "Future sandbox accounting primitives are NOT_ADMISSIBLE until integrated "
    "and independently qualified; research/colab_cli remains unchanged."
)


def digest(value: Any) -> str:
    """Hash finite canonical JSON, identically to create-only export payloads."""
    from research.bl_paced_identity.events import canonical_bytes

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    """Hash an explicitly declared non-secret source or manifest off exposure."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_registry(pid: int) -> dict[str, Any]:
    """Declare the exact current root and disk sources without admitting it.

    Only known non-secret assignment keys are read. Session credential stores
    and process environments are never exported. The declaration preserves the
    important difference between disk-source identity and loaded-source proof.
    """
    import psutil

    process = psutil.Process(pid)
    with process.oneshot():
        argv = process.cmdline()
        cwd = Path(process.cwd()).resolve(strict=True)
        interpreter = str(Path(process.exe()).resolve(strict=True))
        birth_ns = round(process.create_time() * 1_000_000) * 1000
        uid = process.uids().real
    if len(argv) != 9 or argv[1:3] not in (
        ["-m", "tools.sandbox.supervisor"],
        ["-m", "tools.sandbox.supervisor_v2"],
        ["-m", "research.colab_cli.supervisor"],
    ):
        raise ValueError("root must have the exact known supervisor run argv structure")
    if argv[3] != "--state-dir" or argv[5] != "--colab" or argv[7] != "run":
        raise ValueError("unexpected supervisor argv structure")
    return _registry_from_identity(pid, uid, birth_ns, interpreter, argv, cwd)


def _registry_from_identity(
    pid: int, uid: int, birth_ns: int, interpreter: str, argv: list[str], cwd: Path
) -> dict[str, Any]:
    import psutil

    capture_start = time.monotonic_ns()
    boot_id = "boot-" + str(psutil.boot_time())
    state_dir = Path(argv[4]).resolve(strict=True)
    manifest_path = Path(argv[-1]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    lane = state_dir / manifest["job_id"]
    assignment_path = lane / "assignment.json"
    assignment = json.loads(assignment_path.read_text())
    hook = lane / "cli-hook/sitecustomize.py"
    paths = [
        *sorted((cwd / "research/colab_cli").glob("*.py")),
        *sorted((cwd / "tools/sandbox").glob("*.py")),
        hook,
    ]
    if argv[2] == "tools.sandbox.supervisor_v2":
        # Globs bind the wider closure, but required files must also be named
        # explicitly so a missing collector source fails during capture.
        paths.extend(
            cwd / "tools/sandbox" / name
            for name in ("collector_v2.py", "supervision_accounting.py", "supervisor.py")
        )
        paths.extend(
            cwd / name
            for name in (
                "tools/sandbox/launch_v2.sh",
                "tools/ops/dispatch.py",
                "tools/ops/dispatch.sh",
            )
        )
    sources = {str(path.resolve(strict=True)): file_digest(path) for path in paths}
    colab = Path(argv[6]).resolve(strict=True)
    if argv[2] == "tools.sandbox.supervisor_v2":
        sources[str(colab)] = file_digest(colab)
    with colab.open("rb") as handle:
        shebang = handle.readline(4096)
    if not shebang.startswith(b"#!"):
        raise ValueError("CLI declaration requires an unambiguous Python shebang")
    command = shlex.split(shebang[2:].decode().strip())
    if len(command) != 1 or not Path(command[0]).is_absolute():
        raise ValueError("CLI declaration has ambiguous interpreter")
    guard = cwd / "research/colab_cli/cli_guard.py"
    root = {
        "id": f"pid-{pid}-birth-{birth_ns}",
        "uid": uid,
        "pid": pid,
        "birth_ns": birth_ns,
        "boot_id": boot_id,
        "interpreter": interpreter,
        "module": argv[2],
        "argv": argv,
        "repo": str(cwd),
        "cwd": str(cwd),
        "state_dir": str(state_dir),
        "source_hashes": sources,
        "startup_hook_hash": file_digest(hook),
        "manifest": str(manifest_path),
        "manifest_digest": file_digest(manifest_path),
        "colab": str(colab),
        "config": str(lane / "sessions.json"),
        "cli_interpreter": str(Path(command[0]).resolve(strict=True)),
        "cli_source_hashes": {
            str(path.resolve(strict=True)): file_digest(path) for path in (colab, hook, guard)
        },
        "cli_startup_hook_hash": file_digest(hook),
        "operation_timeout_seconds": 120,
        "assignment": assignment["endpoint"],
        "session": assignment["session"],
        "generation": assignment["generation"],
    }
    if file_digest(RULING_PATH) != RULING_SHA256:
        raise ValueError("Astra ruling source differs")
    # A second read catches PID reuse/restart during the inventory itself.
    verify_root(root)
    return {
        "schema": "served-supervisor-registry/v2",
        "policy": POLICY,
        "boot_id": boot_id,
        "capture_start_ns": capture_start,
        "capture_end_ns": max(time.monotonic_ns(), capture_start + 1),
        "policy_source_hash": digest(policy_sources()),
        "ruling_sha256": RULING_SHA256,
        "ruling_hash": RULING_SHA256,
        "roots": [root],
        "declaration_only": True,
        "live_accounting_capability": "NOT_ADMISSIBLE",
        "loaded_source_verified": False,
        "manifest_path": str(manifest_path),
        "assignment_path": str(assignment_path),
        "blocker": BLOCKER,
    }


def verify_root(root: dict[str, Any]) -> None:
    """Recheck high-resolution identity and exact argv/cwd plus source hashes."""
    import psutil

    process = psutil.Process(root["pid"])
    with process.oneshot():
        actual = {
            "uid": process.uids().real,
            "birth_ns": round(process.create_time() * 1_000_000) * 1000,
            "interpreter": str(Path(process.exe()).resolve(strict=True)),
            "argv": process.cmdline(),
            "cwd": str(Path(process.cwd()).resolve(strict=True)),
        }
    if any(root[key] != value for key, value in actual.items()):
        raise ValueError("supervisor identity changed: fresh prospective registry required")
    for name, expected in root["source_hashes"].items():
        path = Path(name)
        if str(path.resolve(strict=True)) != name or file_digest(path) != expected:
            raise ValueError("supervisor disk source differs from sealed registry")


def read_registry(path: Path, *, live: bool = False) -> dict[str, Any]:
    """Verify the immutable triplet and policy; never accept a v1 reinterpretation."""
    from research.bl_served_search.remote_supervision import validate_registry

    registry = read_export(path)
    if registry.get("policy") != POLICY or registry.get("ruling_sha256") != RULING_SHA256:
        raise ValueError("supervisor registry policy/ruling mismatch")
    checked = validate_registry(registry)
    if not checked["valid"]:
        raise ValueError(f"invalid supervisor registry: {checked['errors']}")
    if live:
        for root in registry["roots"]:
            verify_root(root)
        # This repository intentionally does not implement an attach mechanism.
        # No caller-supplied capability flag can open this missing evidence gate.
        raise RuntimeError(BLOCKER)
    return registry


def replay_host(host: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    """Independent replay of the source-bound prospective evidence protocol."""
    from research.bl_served_search.remote_supervision import replay

    if not isinstance(host, dict) or not isinstance(intent, dict):
        raise ValueError("host and supervision intent must be objects")
    registry = host["supervisor_registry"]
    if not isinstance(registry, dict):
        raise ValueError("supervisor registry must be an object")
    if (
        host.get("load_policy") != POLICY
        or intent.get("load_policy") != POLICY
        or registry.get("ruling_sha256") != RULING_SHA256
        or registry.get("ruling_hash") != RULING_SHA256
        or intent.get("ruling_sha256") != RULING_SHA256
        or intent.get("registry_sha256") != digest(registry)
        or intent.get("policy_source_sha256") != policy_sources()
        or registry.get("policy_source_hash") != digest(policy_sources())
    ):
        raise ValueError("supervision intent registry/ruling/policy/source mismatch")
    result = replay(
        registry,
        host["supervision_journal"],
        host["supervision_cpu_evidence"],
        observations=host["supervision_identity_observations"],
        point_screens=host["supervision_point_screens"],
        ordinary_intervals=host["cpu_samples"],
    )
    if not result["valid"]:
        raise ValueError(f"supervision replay invalid: {result['errors']}")
    if result != host.get("supervision_replay"):
        raise ValueError("reported supervision accounting differs from independent replay")
    # Even a self-consistent offline replay cannot establish that an unavailable
    # live observer captured every process, phase and application payload. Keep
    # the panel reducer closed as well as the command-line admission path.
    raise ValueError(BLOCKER)


def policy_sources() -> dict[str, str]:
    """Bind all prospective implementation and amendment bytes off the tick."""
    base = Path(__file__).parent
    names = (
        "remote_supervision.py",
        "supervision_io.py",
        "supervision_identity.py",
        "supervision_cpu.py",
        "supervision_report.py",
        "combined_policy.py",
        "supervision_dispatch.py",
        "supervision_qualification.py",
        "desktop_apps.py",
        "desktop_policy.py",
        "DESKTOP_APPS_AMENDMENT_20261027.md",
        "ASTRA_DESKTOP_RULING_20261027.md",
        "SUPV2_WIRING_20261028.md",
        "host_load.py",
        "background_report.py",
        "served_tick_probe.py",
        "tick_trace.py",
        "REMOTE_SUPERVISION_AMENDMENT_20261027.md",
    )
    return {
        **{name: file_digest(base / name) for name in names},
        **{
            f"tools/sandbox/{name}": file_digest(base.parents[1] / "tools/sandbox" / name)
            for name in ("collector_v2.py", "supervisor_v2.py", "supervisor.py")
        },
        "tools/sandbox/supervision_accounting.py": file_digest(
            base.parents[1] / "tools/sandbox/supervision_accounting.py"
        ),
        **{
            name: file_digest(base.parents[1] / name)
            for name in (
                "tools/sandbox/launch_v2.sh",
                "tools/sandbox/source_delivery.py",
                "tools/ops/dispatch.py",
                "tools/ops/dispatch.sh",
                "research/colab_cli/supervisor.py",
                "research/colab_cli/policy.py",
                "research/colab_cli/cli_guard.py",
            )
        },
        "research/bl_paced_identity/events.py": file_digest(
            base.parent / "bl_paced_identity/events.py"
        ),
        "research/bl_paced_identity/record.py": file_digest(
            base.parent / "bl_paced_identity/record.py"
        ),
        "research/bl_k1/m1_background.py": file_digest(base.parent / "bl_k1/m1_background.py"),
    }


def main(argv: list[str] | None = None) -> int:
    """Seal a read-only candidate declaration; this never admits a live panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-registry", type=int, metavar="ROOT_PID", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = capture_registry(args.seal_registry)
    export_json(args.output, value)
    print(json.dumps({"registry": str(args.output), "admission": "UNAVAILABLE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
