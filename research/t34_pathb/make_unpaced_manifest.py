"""Build one qualified public-input manifest for the T34 unpaced arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from research.t34_exit_b8.public_inputs import _load as load_public_catalog
from research.t34_exit_b8.qualification import QualificationRefused, require_qualified_transport
from research.t34_pathb.retention_v2 import source_inventory
from research.t34_pathb.source_binding_v2 import source_paths
from research.t34_pathb.unpaced_colab import (
    GLOBAL_RESERVE_UNITS,
    GLOBAL_UNIT_CAP,
    SMOKE_SECONDS_PER_WORLD,
    UnpacedRefused,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "research/t34_pathb/unpaced_colab.sh"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UnpacedRefused(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise UnpacedRefused("registration must be a JSON object")
    return value


def _commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def make_manifest(
    *,
    registration: Path,
    catalog: Path,
    output_dir: Path,
    shard: int,
    shard_count: int,
    nproc: int,
    commit: str,
    cpu_rate_units_per_hour: float | None = None,
    other_assignments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise UnpacedRefused("commit must be full lowercase SHA")
    if not output_dir.is_absolute() or output_dir.exists():
        raise UnpacedRefused("output_dir must be fresh and absolute")
    if not 0 <= shard < shard_count or not 1 <= shard_count <= 3 or not 2 <= nproc <= 64:
        raise UnpacedRefused("invalid shard or nproc")
    value = _load(registration)
    load_public_catalog(catalog)
    if value.get("ratified") is not True:
        raise UnpacedRefused("production manifest requires ratified registration")
    if value.get("source_commit") != commit:
        raise UnpacedRefused("manifest commit must equal the registration source commit")
    rows = list(value.get("terminal_rows", [])) + list(value.get("product_rows", []))
    if len(rows) != 176:
        raise UnpacedRefused("unpaced manifest requires all 176 registered terminal/product rows")
    lane_worlds = sum(index % shard_count == shard for index in range(len(rows)))
    rate, gpu = (
        (2.0, "L4") if cpu_rate_units_per_hour is None else (float(cpu_rate_units_per_hour), "cpu")
    )
    if not math.isfinite(rate) or rate <= 0:
        raise UnpacedRefused("CPU runtime requires a positive measured provider rate")
    # Threefold CPU conservatism, divided across the actual fixed worker pool,
    # with twenty minutes for public input setup/export/terminal receipt.
    process_workers = min(nproc - 1, 10)
    if process_workers < 1:
        raise UnpacedRefused("production manifest needs main plus one process worker")
    estimate = math.ceil((lane_worlds * SMOKE_SECONDS_PER_WORLD * 3 / process_workers + 1200) / 60)
    timeout = max(3, 2 * estimate)
    cap = GLOBAL_UNIT_CAP / shard_count
    reserve = GLOBAL_RESERVE_UNITS / shard_count
    if timeout * rate / 60 + reserve > cap:
        raise UnpacedRefused("shard timeout liability exceeds registered aggregate unit cap")
    from research.colab_cli.accounting import _manifest_rates

    assignments = {} if other_assignments is None else other_assignments
    _manifest_rates(assignments)
    if shard_count + len(assignments) > 3:
        raise UnpacedRefused("planned lanes plus existing account assignments exceed three")
    try:
        require_qualified_transport(ROOT, commit)
    except QualificationRefused as exc:
        raise UnpacedRefused(str(exc)) from exc
    return {
        "job_id": f"t34-unpaced-s{shard}-of{shard_count}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "DRAFT_NO_ALLOCATION_AUTHORIZED",
        "env": {
            "PYTHONUNBUFFERED": "1",
            "T34_UNPACED_INPUT_MODE": "public",
            "T34_UNPACED_DEVICE": "cpu",
            "T34_UNPACED_NPROC": str(nproc),
            "T34_UNPACED_SHARD": str(shard),
            "T34_UNPACED_SHARD_COUNT": str(shard_count),
            "T34_UNPACED_SOURCE_COMMIT": value["source_commit"],
        },
        "requirements": {
            "gpu": gpu,
            "allowed_gpus": [gpu],
            "min_vcpu": nproc,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir),
        "inputs": [
            {"path": str(catalog.resolve()), "sha256": _sha(catalog), "dest": "public-inputs.json"},
            {
                "path": str(registration.resolve()),
                "sha256": _sha(registration),
                "dest": "registration.json",
            },
        ],
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": assignments,
        },
        "lease_seconds": 600,
        "max_export_bytes": 256 * 1024**2,
        "source_paths": sorted(
            {
                *source_paths(ROOT),
                *(
                    str(path.relative_to(ROOT))
                    for path in (ROOT / "research/colab_cli").rglob("*")
                    if path.is_file() and path.suffix in {".py", ".sh"}
                ),
            }
        ),
        "unpaced_budget": {
            "global_unit_cap": GLOBAL_UNIT_CAP,
            "global_reserve_units": GLOBAL_RESERVE_UNITS,
            "smoke_seconds_per_world": SMOKE_SECONDS_PER_WORLD,
            "conservative_cpu_multiplier": 3,
            "worlds": lane_worlds,
            "aggregate_shards": shard_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--cpu-rate-units-per-hour", type=float)
    parser.add_argument("--other-assignments", type=json.loads)
    args = parser.parse_args()
    manifest = make_manifest(
        registration=args.registration,
        catalog=args.catalog,
        output_dir=args.output_dir,
        shard=args.shard,
        shard_count=args.shard_count,
        nproc=args.nproc,
        commit=args.commit or _commit(ROOT),
        cpu_rate_units_per_hour=args.cpu_rate_units_per_hour,
        other_assignments=args.other_assignments,
    )
    # A production manifest is not merely JSON-shaped: exercise the broker's
    # pure admission and sparse-source preparation before publishing its
    # create-only bytes.  Neither operation allocates a runtime.
    from research.colab_cli import source
    from research.colab_cli.supervisor import validate
    from research.t34_pathb.registration_retention_v2 import (
        _check_common,
        _check_operating_characteristic,
    )

    registration_value = _load(args.registration)
    _check_common(registration_value, root=ROOT)
    _check_operating_characteristic(registration_value, root=ROOT)
    if registration_value.get("source_hashes") != source_inventory(ROOT):
        parser.error("registration source inventory differs from current executable bytes")
    from research.t34_pathb.registration_retention_v2 import _bound_path

    trace_path = _bound_path(
        registration_value.get("trace_receipt"), root=ROOT, label="trace_receipt"
    )
    trace = _load(trace_path)
    if (
        registration_value.get("ratified_by") != "Claude"
        or registration_value.get("near_boundary_precedence_acknowledged") is not True
        or trace.get("status") != "PASS"
        or trace.get("source_hashes") != registration_value["source_hashes"]
    ):
        parser.error("registration lacks final-source ratification and passing trace evidence")
    for name, digest in registration_value["source_hashes"].items():
        pinned = subprocess.check_output(["git", "show", f"{manifest['commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(pinned).hexdigest() != digest:
            parser.error(f"pinned source differs from registration: {name}")
    try:
        validate(manifest, repo=ROOT)
        with tempfile.TemporaryDirectory(prefix="t34-unpaced-source-") as directory:
            source.prepare(ROOT, manifest, Path(directory))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
