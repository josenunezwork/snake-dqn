"""Prepare A3 A100/L4 packs offline; never allocate or dispatch a runtime."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from research.a3_p1_ceiling.runtime import SCHEMA, UNITS, allocation, make_ledger
from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.manifests import ROOT, _base, _bounded_source_paths, input_row
from research.colab_cli.canary import check, family

SCRIPT = "research/a3_p1_ceiling/colab.sh"
REGISTRY = ROOT.parent / "snake-dqn-artifacts/ops/colab-cli-qualified.json"


def prepare(root: Path, commit: str, other: dict[str, float]) -> None:
    """Freeze 4 h/40 u combined including both canaries, setup/export and reserve."""
    for ref in ("HEAD", commit):
        subprocess.run(
            [
                "python3",
                str(ROOT / "tools/ops/qualification.py"),
                "check",
                "--repo",
                str(ROOT),
                "--record",
                str(REGISTRY),
                "--ref",
                ref,
            ],
            check=True,
        )
    # Source identity must be the published, reviewed commit, never dirty local code.
    for directory in ("research/a3_p1_ceiling", "research/bl_ppo"):
        if subprocess.check_output(["git", "status", "--porcelain", "--", directory], cwd=ROOT):
            raise ValueError("commit all assay/P1 source before preparing a pin")
    root.mkdir(parents=True, exist_ok=False)
    for gpu, rate, main_cap, canary_cap in (("A100", 10.0, 29.0, 3.0), ("L4", 2.0, 7.0, 1.0)):
        probe_minutes = 16 if gpu == "A100" else 20
        main_minutes = 120 - probe_minutes
        lane = root / gpu.lower()
        lane.mkdir()
        control = {
            "schema": SCHEMA,
            "namespace": f"a3-{gpu.lower()}-{commit[:12]}",
            "commit": commit,
            "gpu": gpu,
            "shape": [128, 128],
            "assay_seed": 20271022,
            "temperature_rule": (
                "pinned P1 independent no-step calibration; seed+104729; once per attempt"
            ),
            "units": UNITS,
            "repeats": 3,
            "kill_seconds": 1.6384,
            "gradient_atol": 2e-5,
            "gradient_rtol": 2e-4,
        }
        export_json(lane / "control.json", control)
        export_json(lane / "verified-ledger.json", make_ledger(control, []))
        manifest = _base(
            job_id=f"a3-{gpu.lower()}-{commit[:8]}-main",
            commit=commit,
            output=lane / "main-output",
            estimate=main_minutes / 2,
            timeout=main_minutes,
            cap=main_cap,
        )
        manifest.update(
            script=SCRIPT,
            device="cuda",
            required_exports=["receipt.json"],
            claim="A3 performance assay; no P1 training/evaluation admission",
            resume={
                "unit_kind": "independent-assay-cell",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
        )
        manifest["requirements"].update(gpu=gpu, allowed_gpus=[gpu], min_vcpu=12)
        manifest["budget"].update(
            rate_units_per_hour=rate, reserve_units=main_cap * 0.2, other_assignments=other
        )
        manifest["env"] = {
            "A3_DEVICE": "cuda",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
        manifest["inputs"] = [
            input_row(lane / "control.json", "control.json"),
            input_row(lane / "verified-ledger.json", "verified-ledger.json"),
            input_row(ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json", "public-inputs.json"),
        ]
        manifest["source_paths"] += [
            "research/a3_p1_ceiling",
            "research/bl_upload",
            "research/t34_exit_b8/PUBLIC_INPUTS.json",
        ]
        manifest["source_paths"] = _bounded_source_paths(manifest)
        manifest["max_export_bytes"] = 64 * 1024**2
        manifest["transport"] = {
            "mode": "local_supervisor_upload",
            "dispatch_ready": False,
            "requires": "same-pin target-platform canary receipt",
        }
        manifest["envelope"] = {
            "base_series_hours": 2,
            "effective_series_hours": 4,
            "base_series_units": 20,
            "effective_series_units": 40,
            "base_timeout_minutes": main_minutes / 2,
            "effective_timeout_minutes": main_minutes,
            "base_unit_cap": main_cap / 2,
            "effective_unit_cap": main_cap,
            "base_reserve_units": main_cap * 0.1,
            "effective_reserve_units": main_cap * 0.2,
            "expected_lane_minutes_including_canary": 60,
            "forecast_rate_units_per_hour": 7.0 if gpu == "A100" else 2.0,
            "rate_source": "round-20 ruling 10: forecast 3.5x L4; A100 ceiling 5x L4",
            "attempt_policy": (
                "No automatic completion retry. At most two only after mechanical failure, "
                "verified stop and missing units; all attempts share remaining original 4h/40u "
                "envelope. Fresh packs required; no scientific retry."
            ),
        }
        export_json(lane / "main-template.json", manifest)
        probe = copy.deepcopy(manifest)
        probe.pop("resume")
        probe["resume_exemption"] = {
            "kind": "short-canary",
            "reason": (
                "Same-pin A3 real-entrypoint CUDA/transfer probe; excluded tiny independent units"
            ),
        }
        probe.update(
            job_id=f"a3-{gpu.lower()}-{commit[:8]}-canary",
            output_dir=str(lane / "canary-output"),
            timeout_minutes=probe_minutes,
        )
        probe["requirements"]["est_minutes"] = probe_minutes / 2
        probe["budget"].update(unit_cap=canary_cap, reserve_units=canary_cap * 0.2)
        probe["env"].update(
            SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/a3_p1_ceiling"
        )
        probe["canary"] = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": family(probe),
        }
        probe["envelope"].update(
            base_timeout_minutes=probe_minutes / 2,
            effective_timeout_minutes=probe_minutes,
            base_unit_cap=canary_cap / 2,
            effective_unit_cap=canary_cap,
            base_reserve_units=canary_cap * 0.1,
            effective_reserve_units=canary_cap * 0.2,
        )
        check(probe, ROOT)
        export_json(lane / "canary.json", probe)
    export_json(
        root / "pack.json",
        {
            "schema": SCHEMA,
            "commit": commit,
            "allocation": {
                gpu: allocation(json.loads((root / gpu / "control.json").read_text()))
                for gpu in ("a100", "l4")
            },
            "qualification": json.loads(REGISTRY.read_text()),
            "status": "PREPARED_REQUIRES_TARGET_CANARIES",
            "total_timeout_minutes": 240,
            "total_unit_cap": 40,
            "total_reserve_units": 8,
        },
    )


def bind(template: Path, receipt: Path, out: Path) -> None:
    """Require genuine broker proof before creating an immutable dispatch manifest."""
    manifest = json.loads(template.read_text())
    manifest["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": digest(receipt)}}
    check(manifest, ROOT)
    from research.colab_cli.supervisor import validate

    validate(manifest, repo=ROOT)
    manifest["transport"]["dispatch_ready"] = True
    export_json(out, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--commit", required=True)
    prep.add_argument("--other-assignments", type=Path)
    binding = sub.add_parser("bind")
    binding.add_argument("--template", type=Path, required=True)
    binding.add_argument("--receipt", type=Path, required=True)
    binding.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        other = json.loads(args.other_assignments.read_text()) if args.other_assignments else {}
        if not isinstance(other, dict):
            raise ValueError("other assignments must map endpoint to positive rate")
        prepare(args.root.resolve(), args.commit, other)
    else:
        bind(args.template, args.receipt, args.out)


if __name__ == "__main__":
    main()
