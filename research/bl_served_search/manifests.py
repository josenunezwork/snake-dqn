"""Create source-pinned validation-only manifests. Never contacts Colab."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from research.bl_prefix.make_manifest import qualified_transport
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.fixture_bundle import BUNDLE_NAME, pack
from research.bl_served_search.pricing_v2 import forecast
from research.bl_served_search.upload_readiness import check as check_upload
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
    mac_ruler: Path,
    parity_reference: Path,
    l4_ruler: Path | None = None,
    ruler_estimate_minutes: float = 48,
    repo: Path = ROOT,
) -> dict:
    """Prepare nine immutable manifests without provider calls or clipped forecasts."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full lowercase 40-character source pin required before preparation")
    from research.bl_served_search import evidence_v2
    from research.bl_served_search.ceiling import validate_bank
    from research.bl_served_search.chunks import CHUNK_COUNT, chunk_rows, world_ids_sha256
    from research.bl_served_search.pricing_v2 import fallback_open
    from research.bl_served_search.provenance_v2 import FROZEN_BANK_SHA256
    from research.bl_served_search.transport_v2 import preflight

    if any((out / name).exists() for name in ("plan.json", "evidence", "y1", "chunk-01")):
        raise FileExistsError("fresh manifest family required")
    authority = qualified_transport(repo, commit)
    published = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=repo, text=True
    )
    if "origin/main" not in published:
        raise ValueError("pin must be published on origin/main")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit, "--", *source_paths(repo)], cwd=repo, text=True
    )
    if changed.strip():
        raise ValueError("local source differs from pin: " + changed)
    if not 0 < ruler_estimate_minutes <= 48:
        raise ValueError("Y1 ruler must fit 4u including reserve")
    out.mkdir(parents=True, exist_ok=True)
    bank_path, audit_path = portable_allocation(allocation, out / "evidence")
    if digest(bank_path) != FROZEN_BANK_SHA256:
        raise ValueError("bank differs from unchanged BANK_L2.json")
    for path in (mac_ruler, parity_reference):
        verify_export(path)
    if l4_ruler:
        verify_export(l4_ruler)
    bundle = pack(fixtures, out / "evidence" / BUNDLE_NAME)
    fixture_value = json.loads((fixtures / "ruler-fixtures.json").read_text())
    estimate = forecast(
        json.loads(mac_ruler.read_text()),
        fixture_value,
        json.loads(l4_ruler.read_text()) if l4_ruler else None,
    )
    reference = json.loads(parity_reference.read_text())
    from research.bl_served_search.ceiling import source_manifest

    if (
        reference.get("identity", {}).get("bank_sha256") != digest(bank_path)
        or reference.get("identity", {}).get("source_manifest") != source_manifest()
    ):
        raise ValueError("fresh source/portable-bank parity reference required")
    files = [
        ("bank.json", bank_path),
        ("audit.json", audit_path),
        ("registration.md", repo / "research/bl_served_search/PREREGISTRATION_L2.md"),
        ("amendment-08.md", repo / "research/bl_served_search/AMENDMENT_08.md"),
        ("amendment-v2.md", repo / "research/bl_served_search/AMENDMENT_PLAN_V2_WP1.md"),
        ("ruler-fixture-spec-08.md", repo / "research/bl_served_search/RULER_FIXTURES_08.md"),
        (BUNDLE_NAME, bundle),
        ("ruler-mac.json", mac_ruler),
        ("parity-mac.json", parity_reference),
    ]
    if l4_ruler:
        files.append(("ruler-l4.json", l4_ruler))
    envelope = evidence_v2.pack(files, out / evidence_v2.NAME)
    expanded_hashes = {name: digest(path) for name, path in files}
    catalog = repo / "research/t34_exit_b8/PUBLIC_INPUTS.json"
    inputs = [
        {"dest": name, "path": str(path.resolve()), "sha256": digest(path)}
        for name, path in ((evidence_v2.NAME, envelope), ("public-inputs.json", catalog))
    ]
    rows = validate_bank(json.loads(bank_path.read_text()))
    manifests = []
    for index in range(CHUNK_COUNT + 1):
        is_ruler = index == 0
        chunk = index - 1
        directory = out / ("y1" if is_ruler else f"chunk-{chunk + 1:02d}")
        directory.mkdir()
        manifest = {
            "job_id": "bl-wp1-v2-"
            + ("y1" if is_ruler else f"chunk-{chunk+1:02d}")
            + "-"
            + __import__("hashlib").sha256(str(out.resolve()).encode()).hexdigest()[:8],
            "commit": commit,
            "script": "research/bl_served_search/" + ("run_ruler.sh" if is_ruler else "run_l2.sh"),
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
            "timeout_minutes": (
                2 * ruler_estimate_minutes if is_ruler else min(720, estimate["timeout_minutes"])
            ),
            "output_dir": str((directory / "runtime-output").resolve()),
            "budget": {
                "unit_cap": 4 if is_ruler else 25,
                "rate_units_per_hour": 2,
                "reserve_units": 0.8 if is_ruler else 5,
                "other_assignments": {},
            },
            "env": {
                "BL_L2_SHARD_INDEX": str(chunk),
                "BL_L2_SHARD_COUNT": "8",
                "BL_L2_WORKERS": "12",
                "SNAKE_DQN_DEVICE": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "BL_RULER_FIXTURES_SHA256": digest(bundle),
                "BL_EVIDENCE_SHA256": digest(envelope),
                "BL_PUBLIC_INPUTS": "1",
            },
            "inputs": list(inputs),
            "lease_seconds": 300,
            "max_export_bytes": 256 * 1024**2,
            "bl_served_search": {
                "development_only": True,
                "dispatch_status": "NOT_DISPATCHED",
                "readiness": "NOT_READY",
                "validation_only": True,
                "forecast": estimate,
                "transport": authority,
                "inventory": "Claude must refresh account inventory before admission",
            },
        }
        if not is_ruler:
            selected = chunk_rows(rows, chunk)
            plan = {
                "schema": "bl-served-search-plan/v3",
                "commit": commit,
                "dispatchable": estimate["status"] == "FITS",
                "development_only": True,
                "excluded_from_registered_verdicts": True,
                "chunk_index": chunk,
                "chunk_count": CHUNK_COUNT,
                "world_ids": [row["world_id"] for row in selected],
                "world_ids_sha256": world_ids_sha256(selected),
                "forecast": estimate,
                "input_hashes": expanded_hashes,
                "analyzer_sha256": digest(repo / "research/bl_served_search/analyze.py"),
            }
            plan_path = directory / "plan.json"
            export_json(plan_path, plan)
            manifest["inputs"].append(
                {
                    "dest": "l2-plan.json",
                    "path": str(plan_path.resolve()),
                    "sha256": digest(plan_path),
                }
            )
        check_upload(manifest, repo=repo)
        validation_error = None
        try:
            validated = validate(manifest, repo=repo)
        except (ValueError, RuntimeError) as exc:
            if is_ruler or estimate["status"] == "FITS":
                raise
            validated, validation_error = manifest, str(exc)
        transport_path = directory / "transport-preflight.json"
        transport = preflight(validated, transport_path, repo=repo)
        manifest_path = directory / "manifest.json"
        export_json(manifest_path, validated)
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": digest(manifest_path),
                "supervisor_validation": "REFUSED" if validation_error else "PASS",
                "validation_error": validation_error,
                "transport_receipt": str(transport_path),
                "transport_sha256": digest(transport_path),
                "upload_forecast_seconds": transport["normal_forecast_seconds"],
                "fallback_upload_forecast_seconds": transport["fallback_upload_forecast_seconds"],
            }
        )
    result = {
        "schema": "bl-served-search-wp1-plan/v2",
        "status": "PREPARED_NOT_DISPATCHED",
        "readiness": (
            "SIZING_READY"
            if estimate["status"] == "FITS" and fallback_open(estimate)
            else "NOT_READY"
        ),
        "commit": commit,
        "forecast": estimate,
        "manifests": manifests,
        "bank_sha256": digest(bank_path),
        "allocation_bank_sha256": digest(allocation / "bank.json"),
        "bank_world_ids_sha256": world_ids_sha256(rows),
        "colab_calls": 0,
        "missing_evidence": [
            "fresh inventory and prior terminal/release receipt",
            "per-assignment runtime parity",
        ]
        + (
            ["fitting measured Y1-L4 forecast; Mac fallback exceeds caps"]
            if estimate["status"] != "FITS"
            else []
        ),
    }
    export_json(out / "plan.json", result)
    export_json(
        out / "transport-preflight.json",
        {
            "schema": "bl-served-search-wp1-transport-set/v2",
            "status": "PASS_NOT_DISPATCHED",
            "colab_calls": 0,
            "receipts": [
                {"path": row["transport_receipt"], "sha256": row["transport_sha256"]}
                for row in manifests
            ],
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    for name in ("allocation", "fixtures", "mac-ruler", "parity-reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--l4-ruler", type=Path)
    parser.add_argument("--ruler-estimate-minutes", type=float, default=48)
    args = parser.parse_args()
    result = build(
        args.commit,
        args.allocation,
        args.out,
        fixtures=args.fixtures,
        mac_ruler=args.mac_ruler,
        parity_reference=args.parity_reference,
        l4_ruler=args.l4_ruler,
        ruler_estimate_minutes=args.ruler_estimate_minutes,
    )
    print(
        json.dumps({key: result[key] for key in ("status", "readiness", "forecast", "colab_calls")})
    )


if __name__ == "__main__":
    main()
