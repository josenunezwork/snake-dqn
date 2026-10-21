"""Create source-pinned validation-only manifests. Never contacts Colab."""

from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_prefix.make_manifest import qualified_transport
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_upload import assert_upload_budget, convert_public_checkpoints, qualify
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
        for ancestor in path.parents:
            if ancestor == repo:
                break
            initializer = ancestor / "__init__.py"
            if initializer.is_file() and initializer not in visited:
                pending.append(initializer)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    package = list(path.relative_to(repo).parts[:-1])
                    prefix = ".".join(
                        package[: len(package) - node.level + 1] + ([prefix] if prefix else [])
                    )
                if prefix:
                    names = [prefix, *(f"{prefix}.{item.name}" for item in node.names)]
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


def forecast(rate: float | None = None) -> dict[str, Any]:
    """Price four chunks; only a measured qualified L4 rate can admit them."""
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError("positive finite L4 rate required")
    historical_rate = 12 * 0.757 / 3
    assumed = historical_rate if rate is None else rate
    estimate = (23120 / assumed + 900) / 60
    fits = rate is not None and rate >= 2.18 and 2 * estimate <= 384
    return {
        "schema": "bl-served-search-forecast/v2",
        "status": "FITS" if fits else "NOT_DISPATCHABLE",
        "worlds": 160,
        "chunks": 4,
        "worlds_per_chunk": 40,
        "workers": 12,
        "batches_per_world_assumption": 578,
        "total_batches": 92480,
        "measured_l4_aggregate_batches_per_second": rate,
        "minimum_l4_aggregate_batches_per_second": 2.18,
        "historical_scripted_unchunked_minutes": 529.026860414,
        "historical_sensitivity_rate": historical_rate,
        "setup_export_minutes": 60,
        "chunk_setup_export_minutes": 15,
        "conservative_allocation_minutes": estimate,
        "total_estimated_allocation_minutes": 4 * estimate,
        "two_times_conservative_minutes": 2 * estimate,
        "total_two_times_minutes": 8 * estimate,
        "timeout_minutes": 384,
        "absolute_wall_cap_minutes": 480,
        "unit_cap": 64,
        "reserve_units": 12.8,
        "chunk_unit_cap": 16,
        "chunk_reserve_units": 3.2,
        "usable_timeout_minutes": 1536,
        "reason": "qualified L4 ruler and runtime parity required; Mac timing never transfers",
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


def bind_l2_plan(manifest: dict[str, Any], plan: dict[str, Any], plan_path: Path) -> None:
    """Create the plan once after the manifest's runtime inputs are final."""
    plan["input_hashes"] = {row["dest"]: row["sha256"] for row in manifest["inputs"]}
    export_json(plan_path, plan)
    manifest["inputs"].append(
        {
            "dest": "l2-plan.json",
            "path": str(plan_path.resolve()),
            "sha256": digest(plan_path),
        }
    )


def build(
    commit: str,
    allocation: Path,
    out: Path,
    *,
    fixtures: Path,
    ruler_estimate_minutes: float = 48,
    repo: Path = ROOT,
) -> dict:
    """Publish five offline manifests, all fresh and explicitly undispatched."""
    if out.exists():
        raise FileExistsError("fresh manifest directory required")
    authority = qualified_transport(repo, commit)
    published = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if "origin/main" not in published:
        raise ValueError("pin must be published on origin/main")
    if not 0 < ruler_estimate_minutes <= 48:
        raise ValueError("ruler estimate must fit 4u including 20% reserve and 2x timeout")
    changed = subprocess.run(
        ["git", "diff", "--name-only", commit, "--", *source_paths(repo)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if changed.strip():
        raise ValueError("local source differs from pinned published commit: " + changed)
    out.mkdir(parents=True)
    bank_path, audit_path = portable_allocation(allocation, out)
    files = [
        ("bank.json", bank_path),
        ("audit.json", audit_path),
        ("registration.md", repo / "research/bl_served_search/PREREGISTRATION_L2.md"),
        ("amendment-08.md", repo / "research/bl_served_search/AMENDMENT_08.md"),
        ("ruler-fixture-spec-08.md", repo / "research/bl_served_search/RULER_FIXTURES_08.md"),
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
    verify_export(fixtures / "ruler-fixtures.json")
    fixture_manifest = json.loads((fixtures / "ruler-fixtures.json").read_text())
    if fixture_manifest.get("mode") != "full":
        raise ValueError("full nine-fixture ruler bank required for L4 manifest")
    from research.bl_served_search.ruler import source_identity, validate_fixtures

    fixture_rows = validate_fixtures(fixture_manifest, full=True)
    if fixture_manifest.get("source_identity") != source_identity():
        raise ValueError("fixture sources differ from pinned ruler implementation")
    for row in fixture_rows:
        relative = Path(row["root_file"])
        if relative.name != str(relative):
            raise ValueError("fixture root must be a portable basename")
        root_path = fixtures / relative
        verify_export(root_path)
        root = json.loads(root_path.read_text())
        if (
            root.get("fixture_id") != row["fixture_id"]
            or root.get("root_sha256") != row["root_sha256"]
            or root.get("root_bytes_b64") != row["root_bytes_b64"]
        ):
            raise ValueError("root sidecar payload disagrees with frozen fixture bank")
    for path in sorted(fixtures.iterdir()):
        if path.is_file() and not path.name.endswith((".sha256", ".complete")):
            verify_export(path)
            files.append((f"ruler-fixtures/{path.name}", path))
    inputs = []
    for dest, path in files:
        inputs.append({"dest": dest, "path": str(path.resolve()), "sha256": digest(path)})
        if dest in {"bank.json", "audit.json"} or dest.startswith("ruler-fixtures/"):
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
    manifests = []
    for index in range(5):
        is_ruler = index == 0
        chunk = index - 1
        label = "y1-ruler" if is_ruler else f"l2-prime-chunk-{chunk + 1}"
        job_inputs = list(inputs)
        if not is_ruler:
            from research.bl_served_search.ceiling import validate_bank
            from research.bl_served_search.chunks import chunk_rows

            bank = json.loads(bank_path.read_text())
            selected = chunk_rows(validate_bank(bank), chunk)
            plan = {
                "schema": "bl-served-search-plan/v2",
                "dispatchable": False,
                "runtime_parity_admitted": False,
                "development_only": True,
                "excluded_from_registered_verdicts": True,
                "commit": commit,
                "chunk_index": chunk,
                "chunk_count": 4,
                "world_ids": [row["world_id"] for row in selected],
                "forecast": estimate,
                "input_hashes": {row["dest"]: row["sha256"] for row in inputs},
                "ruler_admitted": False,
            }
            plan_path = out / f"{label}-plan.json"
        manifest = {
            "job_id": f"bl-{label}-20261021",
            "commit": commit,
            "script": (
                "research/bl_served_search/run_ruler.sh"
                if is_ruler
                else "research/bl_served_search/run_l2.sh"
            ),
            "source_paths": source_paths(repo) + ["research/bl_served_search/run_ruler.sh"],
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": (
                    ruler_estimate_minutes
                    if is_ruler
                    else estimate["conservative_allocation_minutes"]
                ),
            },
            "timeout_minutes": 2 * ruler_estimate_minutes if is_ruler else 384,
            "output_dir": str(out / f"{label}-undispatched-runtime"),
            "budget": {
                "unit_cap": 4 if is_ruler else 16,
                "rate_units_per_hour": 2,
                "reserve_units": 0.8 if is_ruler else 3.2,
                "other_assignments": [],
            },
            "env": {
                "BL_L2_SHARD_INDEX": str(chunk),
                "BL_L2_SHARD_COUNT": "4",
                "BL_L2_WORKERS": "12",
                "SNAKE_DQN_DEVICE": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
            },
            "inputs": job_inputs,
            "lease_seconds": 300,
            "max_export_bytes": 256 * 1024**2,
            "bl_served_search": {
                "development_only": True,
                "dispatch_status": "NOT_DISPATCHED",
                "readiness": "NOT_DISPATCHABLE",
                "validation_only": True,
                "forecast": estimate,
                "transport": authority,
                "estimate_field": (
                    "pre-data ruler planning bound, not measured on L4"
                    if is_ruler
                    else "historical sensitivity; L4 ruler not yet measured"
                ),
                "other_assignments": "not queried; review at eventual admission",
            },
        }
        # The qualified initial-upload path has a shared 300-second cap.  The
        # roster is the public, hash-verified six-checkpoint release, so ship
        # its compact catalog and reconstruct the exact files on the runtime.
        # Keep this before supervisor validation so an oversized future input
        # cannot become a schema-valid manifest.
        manifest = convert_public_checkpoints(manifest, repo=repo)
        if not is_ruler:
            # The plan authenticates the inputs the runtime reads before the
            # workload begins.  After replacing the six uploads, bind the
            # compact catalog instead; checkpoint files are verified during
            # runtime reconstruction by ``public_inputs.materialize``.
            bind_l2_plan(manifest, plan, plan_path)
        assert_upload_budget(manifest, repo=repo)
        validated = validate(manifest, repo=repo)
        qualification_path = out / f"upload-qualification-{label}.json"
        qualify(validated, repo=repo, receipt_path=qualification_path)
        path = out / f"manifest-{label}-validation-only.json"
        export_json(path, validated)
        manifests.append(
            {
                "path": str(path),
                "sha256": digest(path),
                "upload_qualification": {
                    "path": str(qualification_path),
                    "sha256": digest(qualification_path),
                },
            }
        )
    result = {
        "status": "SCHEMA_VALID_NOT_DISPATCHABLE",
        "commit": commit,
        "manifests": manifests,
        "forecast": estimate,
        "colab_calls": 0,
    }
    export_json(out / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--ruler-estimate-minutes", type=float, default=48)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.commit,
                args.allocation,
                args.out,
                fixtures=args.fixtures,
                ruler_estimate_minutes=args.ruler_estimate_minutes,
            )
        )
    )


if __name__ == "__main__":
    main()
