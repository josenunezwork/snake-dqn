"""Mac-only timing union and path/size/hash verified resume-ledger preparation."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from research.sandbox_a_resident.units import (
    allocation,
    cell_specs,
    digest,
    unit_id,
    unit_stem,
    verified_descriptor,
)


def allowed_path(path: Path) -> Path:
    """Refuse protected inputs and symlinks before any metadata read."""
    absolute = path.absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError("symlink input path refused before read")
    resolved = path.resolve()
    tokens = re.findall(r"[a-z0-9]+", str(resolved).lower())
    protected_campaign = ("t34", "round0", "study", "01")
    if any(token in tokens for token in ("f5", "d4096")) or any(
        tokens[index : index + 4] == list(protected_campaign) for index in range(len(tokens) - 3)
    ):
        raise ValueError("protected input path refused before read")
    return resolved


def read_json(path: Path) -> dict:
    """Read JSON only after applying the immutable-input guard."""
    return json.loads(allowed_path(path).read_text())


def checked_descriptor(path: Path, identity: str) -> dict:
    """Verify every payload sidecar is a permitted regular input before opening it."""
    for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")):
        allowed_path(candidate)
    return verified_descriptor(path, identity)


def expected_stems(control: dict) -> set[str]:
    return {unit_stem(e, p, limit) for e, p, limit, _ in cell_specs(control)}


def export_stem(path: str) -> str | None:
    """Return the logical unit stem for a direct assay export path."""
    if not path.startswith("assay/") or "/" in path[len("assay/") :]:
        return None
    name = path[len("assay/") :]
    for suffix in (
        ".tar.sha256",
        ".json.sha256",
        ".tar.complete",
        ".json.complete",
        ".tar.partial",
        ".json.partial",
        ".tar",
        ".json",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def terminal_metadata(path: str) -> bool:
    """Recognize fixed metadata exports and their atomic-publication sidecars."""
    return path in {
        f"assay/{name}{suffix}"
        for name in ("summary.json", "parity.json")
        for suffix in ("", ".sha256", ".complete", ".partial")
    }


def checked_exports(
    receipt: dict, control: dict, *, allow_terminal_metadata: bool = False
) -> dict[str, dict]:
    """Require a unique, in-allocation export inventory before inspecting payloads."""
    rows = receipt.get("exports", [])
    if not isinstance(rows, list):
        raise ValueError("receipt exports must be an array")
    exports: dict[str, dict] = {}
    stems = expected_stems(control)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("malformed receipt export")
        path = row["path"]
        if path in exports:
            raise ValueError("duplicate receipt export path")
        exports[path] = row
        if (
            path.startswith("assay/")
            and export_stem(path) not in stems
            and not (allow_terminal_metadata and terminal_metadata(path))
        ):
            raise ValueError("foreign receipt export")
    return exports


def checked_assay_directory(directory: Path, control: dict) -> None:
    """Reject partial, foreign, and linked assay files before building a resume ledger."""
    directory = allowed_path(directory)
    if not directory.is_dir():
        raise ValueError("assay export directory missing")
    stems = expected_stems(control)
    for candidate in directory.iterdir():
        allowed_path(candidate)
        path = f"assay/{candidate.name}"
        if not candidate.is_file() or (
            export_stem(path) not in stems and not terminal_metadata(path)
        ):
            raise ValueError("foreign assay export")


def ledger(control: dict, receipts: list[Path]) -> dict:
    """Verify prior broker metadata, then hash payloads without opening outcomes."""
    found = {}
    for receipt_path in receipts:
        receipt_path = allowed_path(receipt_path)
        receipt = read_json(receipt_path)
        manifest_path = receipt_path.parent / "manifest.json"
        manifest = read_json(manifest_path)
        if (
            receipt.get("stop_verified") is not True
            or digest(allowed_path(manifest_path)) != receipt.get("manifest_sha256")
            or json.loads(manifest.get("env", {}).get("SANDBOX_A_CONTROL_JSON", "null")) != control
            or receipt.get("commit") != control.get("source_commit")
            or manifest.get("commit") != control.get("source_commit")
        ):
            raise ValueError("ledger source/allocation/release metadata differs")
        exports = checked_exports(receipt, control, allow_terminal_metadata=True)
        directory = allowed_path(receipt_path.parent / "exports" / "assay")
        checked_assay_directory(directory, control)
        for worlds, phase, limit, _ in cell_specs(control):
            path = directory / f"{unit_stem(worlds, phase, limit)}.tar"
            tar_complete = Path(str(path) + ".complete")
            json_path = path.with_suffix(".json")
            json_complete = Path(str(json_path) + ".complete")
            tar_complete_exists = allowed_path(tar_complete).exists()
            json_complete_exists = allowed_path(json_complete).exists()
            if json_complete_exists and not tar_complete_exists:
                raise ValueError("timing completion marker without checkpoint completion marker")
            if not tar_complete_exists:
                continue
            if not json_complete_exists:
                # A wall-clock stop can happen after TAR publication but before timing export.
                # It is incomplete and must be retried as a fresh receipt/output attempt.
                continue
            identity = unit_id(worlds, phase, limit)
            if identity in found:
                raise ValueError("duplicate completed unit")
            checkpoint = checked_descriptor(path, identity)
            timing = checked_descriptor(json_path, identity)
            for descriptor in (checkpoint, timing):
                relative = "assay/" + Path(descriptor["path"]).name
                entry = exports.get(relative)
                if entry is None or any(entry[k] != descriptor[k] for k in ("size", "sha256")):
                    raise ValueError("unit absent from verified broker export inventory")
            found[identity] = {**checkpoint, "timing": timing}
    return {
        "schema": "sandbox-a-ledger/v1",
        "namespace": "a100-dev/sandbox-a",
        "allocation": allocation(control),
        "units": list(found.values()),
    }


def aggregate(control: dict, receipts: list[Path]) -> dict:
    """The sole timing look: require verified terminal receipts and complete union."""
    expected = {unit_id(e, p, limit) for e, p, limit, _ in cell_specs(control)}
    found, seen_sessions = {}, set()
    elapsed = 0.0
    for receipt_path in receipts:
        receipt_path = allowed_path(receipt_path)
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("stop_verified") is not True
            or receipt.get("error")
            or receipt.get("cleanup_error")
        ):
            raise ValueError("terminal PASS with independently verified release required")
        session = receipt.get("endpoint")
        if not session or session in seen_sessions:
            raise ValueError("missing/duplicate allocation identity")
        seen_sessions.add(session)
        seconds = receipt.get("elapsed_seconds")
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("receipt elapsed time invalid")
        elapsed += seconds
        manifest_path = receipt_path.parent / "manifest.json"
        manifest = read_json(manifest_path)
        if digest(allowed_path(manifest_path)) != receipt.get("manifest_sha256"):
            raise ValueError("receipt manifest hash differs")
        exports = checked_exports(receipt, control, allow_terminal_metadata=True)
        root = allowed_path(receipt_path.parent / "exports" / "assay")
        for name in ("summary.json", "parity.json"):
            row = exports.get(f"assay/{name}")
            path = root / name
            if (
                row is None
                or digest(allowed_path(path)) != row["sha256"]
                or allowed_path(path).stat().st_size != row["size"]
            ):
                raise ValueError("unverified summary/parity export")
        summary = read_json(root / "summary.json")
        if (
            summary.get("allocation") != allocation(control)
            or summary.get("parity", {}).get("status") != "PASS"
            or summary.get("source_commit") != manifest["commit"]
        ):
            raise ValueError("foreign allocation/source or missing resume proof")
        if summary.get("status", "PASS") != "PASS" or any(
            result.get("status", "PASS") != "PASS" for result in summary["results"]
        ):
            raise ValueError("INVALID cell/sweep cannot qualify a timing comparison")
        for result in summary["results"]:
            identity = result["id"]
            if identity not in expected or identity in found:
                raise ValueError("duplicate/foreign result")
            stem = unit_stem(
                result["recipe"]["worlds"],
                result["phase"],
                result["recipe"]["episode_limit"] if "episode_limits" in control else None,
            )
            for suffix in ("tar", "json"):
                path = root / f"{stem}.{suffix}"
                row = exports.get(f"assay/{stem}.{suffix}")
                if (
                    row is None
                    or digest(allowed_path(path)) != row["sha256"]
                    or allowed_path(path).stat().st_size != row["size"]
                ):
                    raise ValueError("unverified result/checkpoint")
            found[identity] = result
    if set(found) != expected:
        raise ValueError("INCOMPLETE: missing complete fixed-cycle units")
    integrated_rows = sum(r["hero_rows"] for r in found.values() if r["phase"] == "integrated")
    return {
        "status": (
            "PASS" if len(seen_sessions) == 1 else "TIMING_UNION_COMPLETE_COMPARISON_INCOMPLETE"
        ),
        "same_allocation_comparison": len(seen_sessions) == 1,
        "label": "simulator-D",
        "results": list(found.values()),
        "allocation_seconds_including_bootstrap_transfer_release": elapsed,
        "integrated_hero_rows_per_allocation_second": integrated_rows / elapsed,
        "decision_scope": (
            "engineering feasibility only; no efficacy/candidate selection; "
            "compare an independently qualified matched L4 path before chartering PPO"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ledger", "aggregate"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    control = read_json(args.control)
    result = (
        ledger(control, args.inputs) if args.mode == "ledger" else aggregate(control, args.inputs)
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS_{args.mode.upper()} {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
