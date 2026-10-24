"""Frozen serial experiment bundles with per-cell immutable exports.

This is deliberately a small runtime primitive rather than a scheduler.  A
study freezes commands, source/toolchain identity, allocation identity, and the
complete cell matrix before it asks the qualified Colab supervisor to run one
packet.  The primitive shares only setup; every cell has its own namespace,
payload, checksum, completion marker, and resume-ledger row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "colab-experiment-bundle/v1"
LEDGER_SCHEMA = "colab-experiment-bundle-ledger/v1"


class BundleError(ValueError):
    """A bundle cannot safely continue with its frozen contract."""


def digest(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise BundleError(f"{field} must be a nonempty path-free string")
    return value


def _argv(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise BundleError(f"{field} must be a nonempty argv string list")
    return value


def _relative(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BundleError(f"{field} must be a relative path")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or value in {".", ".."}:
        raise BundleError(f"{field} must be a safe relative path")
    return path.as_posix()


def namespace(bundle: dict[str, Any], cell: dict[str, Any]) -> str:
    """Return the only permitted per-cell resume identity."""
    return "/".join(
        (
            _identifier(cell.get("experiment", bundle.get("experiment")), "cell.experiment"),
            _identifier(cell.get("variant"), "cell.variant"),
            _identifier(cell.get("cell"), "cell.cell"),
            _identifier(cell.get("repeat"), "cell.repeat"),
        )
    )


def validate(plan: dict[str, Any]) -> None:
    """Validate all frozen branches before any setup command can execute."""
    if plan.get("schema") != SCHEMA:
        raise BundleError("unsupported bundle schema")
    _identifier(plan.get("experiment"), "experiment")
    _identifier(plan.get("allocation_id"), "allocation_id")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise BundleError("bundle requires frozen source/toolchain/dependency SHA256 pins")
    hashes = ("source_sha256", "toolchain_sha256", "dependency_lock_sha256")
    valid_hashes = all(
        isinstance(source.get(key), str)
        and len(source[key]) == 64
        and all(char in "0123456789abcdef" for char in source[key])
        for key in hashes
    )
    if (
        not valid_hashes
        or not isinstance(source.get("commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    ):
        raise BundleError("bundle requires frozen source/toolchain/dependency SHA256 pins")
    setup = plan.get("setup")
    if not isinstance(setup, list):
        raise BundleError("setup must be an argv list")
    for index, command in enumerate(setup):
        _argv(command, f"setup[{index}]")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise BundleError("bundle requires one or more cells")
    seen: set[str] = set()
    names: set[str] = set()
    dependencies: dict[str, list[str]] = {}
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise BundleError(f"cells[{index}] must be an object")
        name = _identifier(cell.get("name"), f"cells[{index}].name")
        if name in names:
            raise BundleError("duplicate cell name")
        names.add(name)
        value = namespace(plan, cell)
        if value in seen:
            raise BundleError("duplicate experiment/variant/cell/repeat namespace")
        seen.add(value)
        _argv(cell.get("command"), f"cells[{index}].command")
        _relative(cell.get("payload"), f"cells[{index}].payload")
        acceptance = cell.get("acceptance")
        if acceptance is not None:
            _argv(acceptance, f"cells[{index}].acceptance")
            _relative(cell.get("acceptance_verdict"), f"cells[{index}].acceptance_verdict")
        restore = cell.get("restore", [])
        if not isinstance(restore, list):
            raise BundleError(f"cells[{index}].restore must be an argv list")
        for restore_index, command in enumerate(restore):
            _argv(command, f"cells[{index}].restore[{restore_index}]")
        after = cell.get("on_failure", "close")
        if after not in {"close", "continue_after_isolation"}:
            raise BundleError("cell on_failure must close or continue_after_isolation")
        if after == "continue_after_isolation" and not restore:
            raise BundleError("continuation after failure requires an explicit restore sequence")
        if after == "continue_after_isolation":
            _argv(cell.get("isolation"), f"cells[{index}].isolation")
        elif cell.get("isolation") is not None:
            raise BundleError("isolation verifier is only valid after a restorable failure")
        depends = cell.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(item, str) for item in depends):
            raise BundleError("depends_on must be a string list")
        dependencies[name] = depends
    if any(item not in names for items in dependencies.values() for item in items):
        raise BundleError("cell depends_on references an unknown cell")
    admissions = plan.get("dependency_admissions", {})
    if not isinstance(admissions, dict) or not all(isinstance(key, str) for key in admissions):
        raise BundleError("dependency_admissions must be a map")
    required = {item for items in dependencies.values() for item in items}
    if set(admissions) != required:
        raise BundleError("every conditional dependency requires one frozen independent admission")
    for binding in admissions.values():
        if not isinstance(binding, dict):
            raise BundleError("dependency admission must be a binding object")
        _relative(binding.get("dest"), "dependency admission dest")
        path, sha = binding.get("path"), binding.get("sha256")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise BundleError("dependency admission path must be absolute")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise BundleError("dependency admission must have SHA256")


def plan_digest(plan: dict[str, Any]) -> str:
    """Hash the complete frozen package, including every command and branch."""
    validate(plan)
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def empty_ledger(plan: dict[str, Any]) -> dict[str, Any]:
    """Create the deterministic, allocation-bound empty resume ledger."""
    validate(plan)
    return {
        "schema": LEDGER_SCHEMA,
        "experiment": plan["experiment"],
        "allocation_id": plan["allocation_id"],
        "plan_sha256": plan_digest(plan),
        "units": [],
    }


def _ledger(plan: dict[str, Any], value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected = empty_ledger(plan)
    for field in ("schema", "experiment", "allocation_id", "plan_sha256"):
        if value.get(field) != expected[field]:
            raise BundleError("foreign bundle resume ledger")
    rows = value.get("units")
    if not isinstance(rows, list):
        raise BundleError("bundle ledger units must be a list")
    allowed = {namespace(plan, cell) for cell in plan["cells"]}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("namespace"), str):
            raise BundleError("malformed bundle ledger unit")
        item = row["namespace"]
        if item not in allowed or item in result:
            raise BundleError("foreign or duplicate bundle ledger namespace")
        if not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise BundleError("bundle ledger unit lacks SHA256")
        if type(row.get("size")) is not int or row["size"] < 0:
            raise BundleError("bundle ledger unit lacks nonnegative size")
        result[item] = row
    return result


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _atomic_ledger(path: Path, plan: dict[str, Any], rows: dict[str, dict[str, Any]]) -> None:
    value = empty_ledger(plan)
    value["units"] = [rows[key] for key in sorted(rows)]
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _verified_output(root: Path, item: dict[str, Any], row: dict[str, Any]) -> bool:
    directory = root / "cells" / item["name"]
    payload = directory / _relative(item["payload"], "payload")
    marker = payload.with_name(payload.name + ".complete")
    sidecar = payload.with_name(payload.name + ".sha256")
    if not (payload.is_file() and sidecar.is_file() and marker.is_file()):
        return False
    actual = digest(payload)
    if actual != row["sha256"] or payload.stat().st_size != row["size"]:
        raise BundleError("verified ledger output differs from immutable payload")
    return sidecar.read_text().strip() == actual and json.loads(marker.read_text()) == row


def _payload_path(directory: Path, cell: dict[str, Any]) -> Path:
    """Resolve a payload under its owned cell directory, never through a link."""
    payload = directory / _relative(cell["payload"], "payload")
    resolved_directory, resolved_payload = directory.resolve(), payload.resolve(strict=False)
    if resolved_directory not in resolved_payload.parents or payload.is_symlink():
        raise BundleError("cell payload escapes its owned output directory")
    return payload


def _accepted_dependencies(plan: dict[str, Any], verified: dict[str, dict[str, Any]]) -> set[str]:
    """Open a dependent branch only with frozen independent Mac PASS proof."""
    parents = {cell["name"]: cell for cell in plan["cells"]}
    accepted: set[str] = set()
    for name, binding in plan.get("dependency_admissions", {}).items():
        path = (
            Path(os.environ["COLAB_INPUT_DIR"])
            / _relative(binding["dest"], "dependency admission dest")
            if os.environ.get("COLAB_INPUT_DIR")
            else Path(binding["path"])
        )
        if not path.is_file() or digest(path) != binding["sha256"]:
            raise BundleError("dependency admission is missing or changed")
        verdict = json.loads(path.read_text())
        identity = namespace(plan, parents[name])
        if not isinstance(verdict, dict) or verdict.get("status") != "PASS":
            raise BundleError("dependency admission is not a PASS verdict")
        if verdict.get("namespace") != identity or identity not in verified:
            raise BundleError("dependency admission does not accept a verified parent")
        if (
            verdict.get("payload_sha256") != verified[identity]["sha256"]
            or verdict.get("source") != plan["source"]
        ):
            raise BundleError("dependency admission payload or frozen source differs")
        accepted.add(name)
    return accepted


def verify_ledger(
    plan_path: Path, output: Path, input_ledger: Path, destination: Path
) -> dict[str, Any]:
    """Mac-side outcome-blind aggregation of immutable complete payloads."""
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, dict):
        raise BundleError("bundle plan must be an object")
    validate(plan)
    source = json.loads(input_ledger.read_text())
    if not isinstance(source, dict):
        raise BundleError("bundle input ledger must be an object")
    rows = _ledger(plan, source)
    allowed_markers = {
        _payload_path(output / "cells" / cell["name"], cell)
        .with_name(Path(cell["payload"]).name + ".complete")
        .resolve()
        for cell in plan["cells"]
    }
    allowed_markers.add((output / "completed-ledger.json.complete").resolve())
    if any(marker.resolve() not in allowed_markers for marker in output.rglob("*.complete")):
        raise BundleError("foreign completed bundle namespace")
    for cell in plan["cells"]:
        identity = namespace(plan, cell)
        if identity in rows:
            old_payload = _payload_path(output / "cells" / cell["name"], cell)
            if old_payload.exists() or Path(str(old_payload) + ".complete").exists():
                raise BundleError("duplicate completed bundle namespace across attempts")
            continue
        directory = output / "cells" / cell["name"]
        payload = _payload_path(directory, cell)
        marker = payload.with_name(payload.name + ".complete")
        sidecar = payload.with_name(payload.name + ".sha256")
        if not (payload.is_file() and marker.is_file() and sidecar.is_file()):
            continue
        actual = digest(payload)
        row = {"namespace": identity, "sha256": actual, "size": payload.stat().st_size}
        if sidecar.read_text().strip() != actual or json.loads(marker.read_text()) != row:
            raise BundleError("bundle completed payload checksum or final marker differs")
        rows[identity] = row
    if destination.exists():
        raise BundleError("verified ledger destination already exists")
    _atomic_ledger(destination, plan, rows)
    return json.loads(destination.read_text())


def run(plan_path: Path, output: Path, ledger_path: Path) -> dict[str, Any]:
    """Run frozen setup/cells serially and publish markers only after checksums."""
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, dict):
        raise BundleError("bundle plan must be an object")
    validate(plan)
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise BundleError("bundle output is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    source_ledger = json.loads(ledger_path.read_text())
    if not isinstance(source_ledger, dict):
        raise BundleError("bundle ledger must be an object")
    rows = _ledger(plan, source_ledger)
    # The supplied ledger was verified on the Mac and is immutable input.  A
    # runtime result cannot promote its own rows to that ledger; it writes a
    # separate candidate-completion ledger for Mac verification after export.
    completed_rows: dict[str, dict[str, Any]] = {}
    completion_ledger = output / "completed-ledger.json"
    if completion_ledger.exists():
        raise BundleError("bundle output already has a completion ledger")
    repo_root = Path.cwd()
    for command in plan["setup"]:
        subprocess.run(
            command, check=True, cwd=repo_root, timeout=plan.get("setup_timeout_seconds", 1800)
        )
    results = []
    # Resume rows are outcome-blind.  A dependency also needs its frozen,
    # independently issued Mac PASS binding before it may run a child.
    completed = _accepted_dependencies(plan, rows)
    for cell in plan["cells"]:
        identity = namespace(plan, cell)
        if identity in rows:
            results.append(
                {"name": cell["name"], "namespace": identity, "status": "SKIPPED_VERIFIED"}
            )
            continue
        if not set(cell.get("depends_on", [])).issubset(completed):
            results.append(
                {"name": cell["name"], "namespace": identity, "status": "CLOSED_DEPENDENCY"}
            )
            continue
        directory = output / "cells" / cell["name"]
        if directory.exists():
            raise BundleError("cell output already exists without a verified ledger row")
        directory.mkdir(parents=True)
        environment = {
            **os.environ,
            "COLAB_BUNDLE_NAMESPACE": identity,
            "COLAB_BUNDLE_CELL_DIR": str(directory),
        }
        started = time.monotonic()
        result = subprocess.run(
            cell["command"],
            cwd=repo_root,
            env=environment,
            timeout=cell.get("timeout_seconds", 3600),
        )
        if result.returncode:
            results.append({"name": cell["name"], "namespace": identity, "status": "FAILED"})
            if cell.get("on_failure", "close") == "close":
                break
            for command in cell["restore"]:
                subprocess.run(
                    command,
                    check=True,
                    cwd=output,
                    timeout=cell.get("restore_timeout_seconds", 600),
                )
            subprocess.run(
                cell["isolation"],
                check=True,
                cwd=repo_root,
                env=environment,
                timeout=cell.get("isolation_timeout_seconds", 600),
            )
            continue
        payload = _payload_path(directory, cell)
        if not payload.is_file() or payload.is_symlink():
            raise BundleError("cell succeeded without its frozen payload")
        row = {"namespace": identity, "sha256": digest(payload), "size": payload.stat().st_size}
        payload.with_name(payload.name + ".sha256").write_text(row["sha256"] + "\n")
        _write_json_once(payload.with_name(payload.name + ".complete"), row)  # marker is last
        completed_rows[identity] = row
        _atomic_ledger(completion_ledger, plan, completed_rows)
        status = "COMPLETED_PENDING_MAC_ACCEPTANCE"
        acceptance = cell.get("acceptance")
        if acceptance is not None:
            accepted = subprocess.run(
                acceptance,
                check=False,
                cwd=repo_root,
                env=environment,
                timeout=cell.get("acceptance_timeout_seconds", 600),
            )
            verdict = directory / _relative(cell["acceptance_verdict"], "acceptance_verdict")
            if (
                accepted.returncode == 0
                and verdict.is_file()
                and json.loads(verdict.read_text()).get("status") == "PASS"
            ):
                # This frozen command is an explicit local isolation/acceptance
                # verdict.  It has no authority to promote the remote result;
                # only a later Mac ledger may make it resumable.
                status = "ACCEPTED_PENDING_MAC_LEDGER"
        results.append(
            {
                "name": cell["name"],
                "namespace": identity,
                "status": status,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
    summary = {
        "schema": SCHEMA,
        "experiment": plan["experiment"],
        "allocation_id": plan["allocation_id"],
        "cells": results,
    }
    _write_json_once(output / "bundle-summary.json", summary)
    # The final aggregate completion ledger is itself an immutable export.  It
    # is published only after the last cell and can be Mac-verified before it
    # becomes a later allocation's input ledger.
    if completed_rows:
        completion_digest = digest(completion_ledger)
        completion_ledger.with_name(completion_ledger.name + ".sha256").write_text(
            completion_digest + "\n"
        )
        _write_json_once(
            completion_ledger.with_name(completion_ledger.name + ".complete"),
            {"sha256": completion_digest, "size": completion_ledger.stat().st_size},
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    """Expose deterministic validation, empty-ledger creation, and execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("plan", type=Path)
    ledger_parser = sub.add_parser("empty-ledger")
    ledger_parser.add_argument("plan", type=Path)
    ledger_parser.add_argument("output", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("plan", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--ledger", type=Path, required=True)
    verify_parser = sub.add_parser("verify-ledger")
    verify_parser.add_argument("plan", type=Path)
    verify_parser.add_argument("bundle_output", type=Path)
    verify_parser.add_argument("--input-ledger", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        if not isinstance(plan, dict):
            raise BundleError("bundle plan must be an object")
        if args.command == "validate":
            validate(plan)
        elif args.command == "empty-ledger":
            _write_json_once(args.output, empty_ledger(plan))
        elif args.command == "verify-ledger":
            print(
                json.dumps(
                    verify_ledger(args.plan, args.bundle_output, args.input_ledger, args.output),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(run(args.plan, args.output, args.ledger), indent=2, sort_keys=True))
    except (BundleError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"bundle refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
