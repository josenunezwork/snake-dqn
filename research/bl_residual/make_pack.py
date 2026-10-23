"""Offline F3 synthetic pack preparation; factory admission stays blocked.

Preparing a template never authorizes Colab. F3 is zero units until F0 labels
exist and F4 has been independently judged. Claude owns later dispatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json
from research.bl_residual.fit import FitRecipe
from research.bl_residual.protocol import (
    B8_SHA,
    DOSES,
    NAMESPACE,
    QUALIFICATION_RECORD,
    QUALIFIED_TREE,
    allocation_hash,
    canonical_sha,
    empty_ledger,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = "research/bl_residual/requirements.txt"
SOURCE_PATHS = [
    "src",
    "research/bl_residual",
    "research/a100_a2/__init__.py",
    "research/a100_a2/model.py",
    "research/bl_ppo/__init__.py",
    "research/bl_ppo/exports.py",
    "research/colab_cli",
]
IMPORT_ROOTS = ["research.bl_residual.job"]


def prepare_inputs(output: Path, b8: Path, *, doses: tuple[int, ...] = DOSES) -> dict:
    """Make a tiny synthetic input set without reading any factory labels."""
    if digest(b8) != B8_SHA:
        raise ValueError("B8 checkpoint identity differs")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(b8, output / "b8.pth")
    value = {
        "schema": "bl-residual-control/v1",
        "synthetic": True,
        "namespace": NAMESPACE + "-synthetic",
        "doses": list(doses),
        "recipe_sha256": canonical_sha(dataclasses.asdict(FitRecipe())),
        "input_hashes": {"b8.pth": B8_SHA},
    }
    value["allocation"] = allocation_hash(value)
    export_json(
        output / "verified-ledger.json", empty_ledger(value["namespace"], value["allocation"])
    )
    value["ledger_sha256"] = digest(output / "verified-ledger.json")
    export_json(output / "control.json", value)
    return value


def make_pack(output: Path, commit: str, b8: Path) -> dict:
    """Require the published source, qualified CLI tree and clean-venv imports."""
    from tools.ops.import_preflight import check as imports
    from tools.ops.qualification import check as qualification

    if output.exists():
        raise FileExistsError(output)
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if qualification(ROOT, Path(QUALIFICATION_RECORD), commit) != QUALIFIED_TREE:
        raise ValueError("F3 qualified transport pin differs")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True
    )
    # Never probe the developer's site packages or create a partial pack on failure.
    preflight = imports(ROOT, commit, SOURCE_PATHS, REQUIREMENTS, IMPORT_ROOTS)
    output.mkdir(parents=True)
    export_json(output / "clean-import-preflight.json", preflight)
    export_json(
        output / "qualified-colab-cli.json", json.loads(Path(QUALIFICATION_RECORD).read_text())
    )
    control = prepare_inputs(output / "inputs", b8)
    inventory = [
        {
            "path": str((output / "inputs" / n).resolve()),
            "dest": n,
            "sha256": digest(output / "inputs" / n),
        }
        for n in ("b8.pth", "control.json", "verified-ledger.json")
    ]
    manifest = {
        "job_id": f"bl-f3-canary-{commit[:8]}",
        "commit": commit,
        "script": "research/bl_residual/colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "scientific_admission": False,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 5},
        "timeout_minutes": 20,
        "output_dir": str((output / "canary-output").resolve()),
        "inputs": inventory,
        "required_exports": ["receipt.json"],
        "env": {"F3_MODE": "canary", "F3_DEVICE": "cuda", "COLAB_CLI_REQUIREMENTS": REQUIREMENTS},
        "budget": {
            "unit_cap": 1,
            "rate_units_per_hour": 1.54,
            "reserve_units": 0.2,
            "other_assignments": {},
        },
        "resume": {
            "unit_kind": "complete-dose-fit",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "max_export_bytes": 32 * 1024**2,
        "f3": {
            "qualified_tree": QUALIFIED_TREE,
            "base_timeout_minutes": 10,
            "effective_timeout_minutes": 20,
            "base_units": 0.5,
            "effective_units": 1,
            "base_reserve_units": 0.1,
            "effective_reserve_units": 0.2,
            "fit_series_base_units": 10,
            "fit_series_effective_units": 20,
            "admission": "HOLD: no F3 Colab use until F0 labels and judged F4",
        },
    }
    export_json(output / "canary-template.json", manifest)
    (output / "DISPATCH.txt").write_text(
        "HOLD: F3 is 0 units until F0 labels exist and F4 is judged.\n"
    )
    receipt = {
        "schema": "bl-residual-pack/v1",
        "commit": commit,
        "synthetic": True,
        "scientific_admission": False,
        "allocation": control["allocation"],
        "qualified_record_path": QUALIFICATION_RECORD,
        "qualified_record_sha256": digest(Path(QUALIFICATION_RECORD)),
        "qualified_tree": QUALIFIED_TREE,
        "clean_import_preflight_sha256": digest(output / "clean-import-preflight.json"),
        "manifest_sha256": digest(output / "canary-template.json"),
    }
    export_json(output / "pack.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--local-smoke-inputs-only", action="store_true")
    args = parser.parse_args()
    if args.local_smoke_inputs_only:
        prepare_inputs(args.out, args.b8)
    else:
        print(json.dumps(make_pack(args.out, args.commit, args.b8)))


if __name__ == "__main__":
    main()
