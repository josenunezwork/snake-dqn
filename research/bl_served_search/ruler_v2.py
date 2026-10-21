"""Run unchanged Y1 batches with explicit legacy-partition compatibility."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from research.bl_served_search import ruler
from research.bl_served_search.exports import export_json, verify_export
from research.bl_served_search.provenance_v2 import compatible_source


def run(
    fixtures_path: Path,
    out: Path,
    *,
    label: str,
    workers: int,
    batches: int,
    operator_spec: str,
    smoke: bool = False,
) -> dict:
    """Preserve the original fixture object/hash and all root bytes."""
    verify_export(fixtures_path)
    fixtures = json.loads(fixtures_path.read_text())
    if not compatible_source(fixtures.get("source_identity", {}), ruler.source_identity()):
        raise ValueError("frozen fixture semantic source differs")
    if any(Path(str(out) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError("fresh ruler receipt required")

    def heartbeat(value: dict) -> None:
        progress = Path(str(out) + ".progress.json")
        progress.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress.with_suffix(".tmp")
        temporary.write_text(ruler.canonical(value) + "\n")
        os.replace(temporary, progress)

    heartbeat({"completed_batches": 0})
    receipt = ruler.run_ruler(
        fixtures,
        ruler._load_factory(operator_spec)(),
        platform_label=label,
        workers=workers,
        min_batches=batches,
        smoke=smoke,
        operator_spec=operator_spec,
        heartbeat=heartbeat,
    )
    receipt["transport_compatibility"] = {
        "rule": "PLAN_v2 partition-only; every other source hash identical",
        "fixture_source_identity": fixtures["source_identity"],
        "executed_source_identity": ruler.source_identity(),
    }
    export_json(out, receipt)
    heartbeat({"completed_batches": receipt["completed_batches"], "status": receipt["status"]})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=("mac", "l4"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--min-batches", type=int, default=30)
    parser.add_argument(
        "--operator", default="research.bl_served_search.ruler_operator:create_operator"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(
        args.fixtures,
        args.out,
        label=args.platform,
        workers=args.workers,
        batches=args.min_batches,
        operator_spec=args.operator,
        smoke=args.smoke,
    )
    print(
        json.dumps({"status": result["status"], "completed_batches": result["completed_batches"]})
    )


if __name__ == "__main__":
    main()
