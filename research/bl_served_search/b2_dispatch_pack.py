"""Build immutable, blocked B2 dispatch requests; Claude supplies live prerequisites.

No manifest is represented as dispatchable before the real Mac preflight and
same-pin canary. Generated commands exclusively dispatch via tools/ops/dispatch.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from research.bl_served_search import b2
from research.bl_served_search import b2_admission as admission
from research.bl_served_search import b2_package as package
from research.bl_served_search.exports import digest, export_json, verify_export


def build(out: Path, spec_path: Path, inputs: Path, commit: str) -> dict:
    """Freeze one canary request and all 32 production indices without gameplay."""
    spec = b2.validate_spec(b2.read_export(spec_path))
    ledger = spec_path.parent / "verified-ledger.json"
    b2.check(spec_path, ledger, inputs)
    qualification = package.qualified_cli(commit)
    backup = b2.configured_drive_root() / "snake-dqn/big-lever/bl-b2/round16-20261022"
    out.mkdir(parents=True, exist_ok=False)
    pack = {
        "schema": b2.SCHEMA + "/dispatch-pack",
        "status": "BLOCKED_PENDING_REAL_PREFLIGHT_AND_CANARY",
        "repo": str(b2.ROOT),
        "commit": commit,
        "spec": str(spec_path.resolve()),
        "spec_sha256": verify_export(spec_path),
        "ledger": str(ledger.resolve()),
        "ledger_sha256": verify_export(ledger),
        "inputs": str(inputs.resolve()),
        "backup": str(backup),
        "qualified_cli": qualification,
        "preflight": str(out / "preflight/preflight.json"),
        "other_assignments": str(out / "other-assignments.json"),
        "admission": str(out / "admission.json"),
        "dispatch_txt": str(out / "DISPATCH.txt"),
        "operator_prerequisites": [
            "Refresh other-assignments.json before every allocation",
            "Reconcile all series receipts and concurrent worst-case liabilities <=40 u",
            "Slots C then B only; stage/verify prior unit before next dispatch in its lane",
            "No dispatch until the Mac lease is free and the real tiny preflight passes",
        ],
        "canary_estimate_minutes": 12,
        "canary_estimate_scope": "conservative admission envelope, not a measured forecast",
        "forecast_units": {
            "original": 8.6,
            "timing_rescale_extra_canary_max": 1,
            "original_envelope": 20,
            "all_attempts_cap": 40,
            "priced_units_per_hour": 2,
            "units_not_spent": True,
        },
    }
    export_json(out / "pack.json", pack)
    common = {
        "commit": commit,
        "spec_sha256": digest(spec_path),
        "allocation_digest": spec["allocation_digest"],
        "workers": 6,
        "status": "REQUEST_ONLY_NOT_A_DISPATCH_MANIFEST",
    }
    export_json(
        out / "canary-request.json",
        {
            **common,
            "index": 0,
            "excluded": True,
            "seed": spec["canary_seed"],
            "unit_cap": 1,
            "timeout_minutes": 24,
        },
    )
    export_json(
        out / "production-requests.json",
        {
            **common,
            "requests": [
                {
                    "index": i,
                    "seed": seed,
                    "unit_cap": package.UNIT_CAP,
                    "reserve_units": package.RESERVE,
                    "timeout_minutes": package.TIMEOUT,
                }
                for i, seed in enumerate(spec["seeds"])
            ],
        },
    )
    for action in ("canary", "admission", "production"):
        script = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f'cd "{b2.ROOT}"\n'
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
            "export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu\n"
            f'exec "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python" '
            f'-m research.bl_served_search.b2_dispatch_pack --pack "{out / "pack.json"}" '
            f'--action {action} "$@"\n'
        )
        target = out / f"{action}.sh"
        target.write_text(script)
        target.chmod(0o700)
    return pack


def operate(pack_path: Path, action: str, index: int | None, receipt: Path | None) -> None:
    """Operator-only finalization; all real evidence gates remain enforced."""
    pack = b2.read_export(pack_path)
    root = pack_path.parent
    spec_path, ledger = Path(pack["spec"]), Path(pack["ledger"])
    if (
        verify_export(spec_path) != pack["spec_sha256"]
        or verify_export(ledger) != pack["ledger_sha256"]
    ):
        raise ValueError("NOT_READY dispatch pack input drift")
    if package.qualified_cli(pack["commit"]) != pack["qualified_cli"]:
        raise ValueError("NOT_READY dispatch pack qualification record drift")
    if action == "admission":
        if receipt is None:
            raise ValueError("--receipt is required")
        result = admission.evaluate(spec_path, receipt)
        export_json(Path(pack["admission"]), result)
        print(json.dumps(result, sort_keys=True))
        return
    probe = action == "canary"
    if not probe and (type(index) is not int or not 0 <= index < 32):
        raise ValueError("--index must be a production index 0..31")
    if probe:
        estimate, report = pack["canary_estimate_minutes"], None
    else:
        report = b2.read_export(Path(pack["admission"]))
        estimate = report["estimate_minutes"]
        receipt = Path(report["receipt_path"])
        if (
            report != admission.evaluate(spec_path, receipt)
            or report["status"] != "ADMIT_UNCHANGED"
        ):
            raise ValueError("NOT_READY production needs verified unchanged admission")
        # Duplicate guard only. Claude must stage the prior unit before the next
        # dispatch in that lane; this adapter does not schedule or monitor lanes.
        spec = b2.read_export(spec_path)
        staged = root / "staged"
        expected = [b2.units.unit_id(spec["namespace"], i) for i in range(32)]
        present = [path.stem for path in sorted((staged / "units").glob("*.json"))]
        verified = b2.units.make_ledger(
            staged, spec["namespace"], spec["allocation_digest"], present
        )
        done = b2.units.validate_ledger(verified, spec["namespace"], spec["allocation_digest"])
        if expected[index] in done:
            raise ValueError("NOT_READY already verified unit; never recompute")
    label = "canary" if probe else f"production-{index:02d}"
    manifests = package.prepare(
        spec_path,
        ledger,
        Path(pack["inputs"]),
        Path(pack["preflight"]),
        root / f"{label}-manifests",
        Path(pack["backup"]),
        pack["commit"],
        probe=probe,
        estimate_minutes=estimate,
        other_assignments=json.loads(Path(pack["other_assignments"]).read_text()),
        canary_receipt=receipt,
    )
    chosen = next(m for m in manifests if probe or m["env"]["BL_B2_INDEX"] == str(index))
    manifest_path = root / f"{label}-manifests" / (chosen["job_id"] + ".json")
    # Ops reservations are global/create-only, so a fresh rescale or recovery
    # pack must not collide with an earlier pack's fixed logical unit name.
    token = hashlib.sha256((str(root.resolve()) + ":" + pack["spec_sha256"]).encode()).hexdigest()[
        :12
    ]
    command = [
        str(b2.ROOT / "tools/ops/dispatch.sh"),
        f"bl-b2-{label}-{token}",
        str(manifest_path),
        "--broker",
        str(b2.ROOT),
        "--cwd",
        str(b2.ROOT),
        "--state-dir",
        str(root / "state" / label),
        "--dispatch-txt",
        pack["dispatch_txt"],
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--action", choices=("canary", "admission", "production"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build(args.out.resolve(), args.spec, args.inputs, args.commit)))
    else:
        operate(args.pack, args.action, args.index, args.receipt)


if __name__ == "__main__":
    main()
