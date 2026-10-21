"""Create source-pinned validation-only manifests. Never contacts Colab."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_prefix.make_manifest import qualified_transport
from research.bl_served_search.exports import digest, export_json, verify_export
from research.colab_cli.supervisor import validate

ROOT = Path(__file__).resolve().parents[2]
B8 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def source_paths(repo: Path = ROOT) -> list[str]:
    """Ship the static import closure, config and supervisor, including local imports."""
    pending = list((repo / "research/bl_served_search").glob("*.py"))
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module, *(f"{node.module}.{item.name}" for item in node.names)]
            for name in names:
                base = repo.joinpath(*name.split("."))
                for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                    if candidate.is_file() and candidate not in visited:
                        pending.append(candidate)
    return sorted(
        {str(path.relative_to(repo)) for path in visited}
        | {
            "research/bl_served_search/run_l2.sh",
            "research/bl_served_search/requirements.txt",
            "research/colab_cli",
            "src",
            "configs",
            "pyproject.toml",
            "requirements.txt",
        }
    )


def forecast() -> dict[str, Any]:
    """Carry the T3 measurement as a floor; never pretend it measures frozen L4."""
    batches = 160 * 578
    worker_hours = batches / 0.757 / 3600
    floor = 20 + worker_hours / 12 * 60
    conservative = 20 + 3 * worker_hours / 12 * 60
    return {
        "schema": "bl-served-search-forecast/v1",
        "status": "NOT_DISPATCHABLE",
        "worlds": 160,
        "workers": 12,
        "batches_per_world_assumption": 578,
        "reference_batches_per_worker_second": 0.757,
        "reference_worker_hours": worker_hours,
        "setup_export_minutes": 20,
        "reference_floor_allocation_minutes": floor,
        "reference_floor_units_at_2_per_hour": floor / 30,
        "conservative_l4_cpu_slowdown": 3,
        "conservative_allocation_minutes": conservative,
        "conservative_units_at_2_per_hour": conservative / 30,
        "timeout_minutes": 384,
        "absolute_wall_cap_minutes": 480,
        "unit_cap": 16,
        "reserve_units": 3.2,
        "two_times_conservative_minutes": 2 * conservative,
        "frozen_mixed_extra_cost": "unmeasured; reference is reused scripted only",
        "reason": "conservative forecast exceeds 8h/16u and 384min usable cap; "
        "no L4 full-body all-mix rate measured; no allocation permitted",
    }


def portable_allocation(source: Path, out: Path) -> tuple[Path, Path]:
    """Strip Mac checkpoint paths without changing seeds, rosters or semantics."""
    bank_path, audit_path = source / "bank.json", source / "audit.json"
    verify_export(bank_path)
    verify_export(audit_path)
    bank = json.loads(bank_path.read_text())
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "CLEAR" or audit.get("bank_sha256") != digest(bank_path):
        raise ValueError("allocation not bound to a CLEAR audit")
    for row in bank["rows"] + bank["excluded_preflight_rows"]:
        for spec in row["plan"]["specs"]:
            if spec[0] == "checkpoint":
                spec[1] = Path(spec[1]).name
    export_json(out / "bank.json", bank)
    export_json(
        out / "audit.json",
        {
            **audit,
            "bank_sha256": digest(out / "bank.json"),
            "allocation_bank_sha256": digest(bank_path),
            "allocation_audit_sha256": digest(audit_path),
            "portable_transform": "checkpoint basenames only; identities/rotation unchanged",
        },
    )
    return out / "bank.json", out / "audit.json"


def build(commit: str, allocation: Path, out: Path, *, repo: Path = ROOT) -> dict:
    """Validate the envelope offline, retaining a hard NOT_DISPATCHABLE plan."""
    if out.exists():
        raise FileExistsError("fresh manifest directory required")
    authority = qualified_transport(repo, commit)
    # The script and every new executable must be present at the published pin.
    published = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "origin/main" not in published:
        raise ValueError("pin must be published on origin/main")
    out.mkdir(parents=True)
    bank_path, audit_path = portable_allocation(allocation, out)
    files: list[tuple[str, Path]] = [
        ("bank.json", bank_path),
        ("audit.json", audit_path),
        ("registration.md", repo / "research/bl_served_search/PREREGISTRATION_L2.md"),
    ]
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007.grid_h5000_identity import FRP3_S12

    checkpoint_root = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    for name, checksum in (*dev_screen.POOL, FRP3_S12):
        path = checkpoint_root / name
        if digest(path) != checksum:
            raise ValueError(f"checkpoint drift: {name}")
        files.append((f"checkpoints/{name}", path))
    b8 = checkpoint_root / "b8_s3_t7_20261016.pth"
    if digest(b8) != B8:
        raise ValueError("B8 s3 drift")
    files.append(("b8.pth", b8))
    inputs = []
    for dest, path in files:
        inputs.append({"dest": dest, "path": str(path.resolve()), "sha256": digest(path)})
        if dest in {"bank.json", "audit.json"}:
            for suffix in (".sha256", ".complete"):
                sidecar = Path(str(path) + suffix)
                inputs.append(
                    {
                        "dest": dest + suffix,
                        "path": str(sidecar.resolve()),
                        "sha256": digest(sidecar),
                    }
                )
    estimate = forecast()
    plan = {
        "schema": "bl-served-search-plan/v1",
        "dispatchable": False,
        "development_only": True,
        "excluded_from_registered_verdicts": True,
        "commit": commit,
        "forecast": estimate,
        "input_hashes": {row["dest"]: row["sha256"] for row in inputs},
    }
    export_json(out / "l2-plan.json", plan)
    inputs.append(
        {
            "dest": "l2-plan.json",
            "path": str((out / "l2-plan.json").resolve()),
            "sha256": digest(out / "l2-plan.json"),
        }
    )
    manifest = {
        "job_id": "bl-l2-dense-ceiling-20261021",
        "commit": commit,
        "script": "research/bl_served_search/run_l2.sh",
        "source_paths": source_paths(repo),
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 12,
            "est_minutes": estimate["reference_floor_allocation_minutes"],
        },
        "timeout_minutes": 384,
        "output_dir": str(out.parent / (out.name + "-undispatched-runtime")),
        "budget": {
            "unit_cap": 16,
            "rate_units_per_hour": 2,
            "reserve_units": 3.2,
            "other_assignments": [],
        },
        "env": {
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
        },
        "inputs": inputs,
        "lease_seconds": 300,
        "max_export_bytes": 64 * 1024**2,
        "bl_served_search": {
            "development_only": True,
            "dispatch_status": "NOT_DISPATCHED",
            "readiness": "NOT_DISPATCHABLE",
            "validation_only": True,
            "estimate_field": "historical reference floor only; not an L4 forecast",
            "forecast": estimate,
            "transport": authority,
            "other_assignments": "not queried; must be reviewed before any future admission",
        },
    }
    validated = validate(manifest, repo=repo)
    export_json(out / "manifest-l2-validation-only.json", validated)
    export_json(
        out / "validation.json",
        {
            "status": "SCHEMA_VALID_NOT_DISPATCHABLE",
            "manifest_sha256": digest(out / "manifest-l2-validation-only.json"),
            "commit": commit,
            "forecast": estimate,
            "colab_calls": 0,
        },
    )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.commit, args.allocation, args.out)
    print(json.dumps({"status": "SCHEMA_VALID_NOT_DISPATCHABLE", "forecast": forecast()}))


if __name__ == "__main__":
    main()
