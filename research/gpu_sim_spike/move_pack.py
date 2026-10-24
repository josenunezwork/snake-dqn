"""Prepare bounded MOVE templates; refuse dispatch without exact CPU preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, move_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/move_bench.sh"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
DEFAULT_ARTIFACT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/move-bench-a100"
)


def require_preflight(path: Path, commit: str) -> dict[str, Any]:
    """Verify the complete chain before any runnable manifest is created."""
    move_units._verify(path)
    pre = json.loads(path.read_text())
    if (
        pre.get("status") != "PREFLIGHT"
        or pre.get("commit") != commit
        or pre.get("device") != "cpu"
    ):
        raise ManifestRefused(
            "MOVE_CPU_EXACTNESS_REQUIRED: preflight is INVALID or not current-pin"
        )
    receipt = Path(pre["receipt_path"])
    if move_units._verify(receipt) != pre.get("receipt_sha256"):
        raise ManifestRefused("MOVE preflight receipt hash differs")
    body = json.loads(receipt.read_text())
    from research.gpu_sim_spike.benchmark import source_hashes

    if body.get("source_sha256") != source_hashes():
        raise ManifestRefused("MOVE preflight source hashes differ")
    if body.get("status") != "COMPLETE" or body.get("commit") != commit:
        raise ManifestRefused("MOVE CPU preflight is not complete")
    cpu = receipt.parent / "cpu-parity.json"
    move_units._verify(cpu)
    proof = json.loads(cpu.read_text())
    if proof.get("status") != "PASS" or proof.get("commit") != commit:
        raise ManifestRefused("MOVE CPU byte proof failed")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _inputs(ledger: Path) -> list[dict[str, str]]:
    value = json.loads(ledger.read_text())
    paths = [ledger]
    paths += [ledger.parent / row["path"] for row in value["units"] + value["fixtures"]]
    return [
        {
            "path": str(candidate.resolve(strict=True)),
            "sha256": sha256_file(candidate),
            "dest": "inputs/" + candidate.relative_to(ledger.parent).as_posix(),
        }
        for path in paths
        for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    ]


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary: bool,
    rate: float,
    canary_receipt: Path | None = None,
) -> dict[str, Any]:
    """Return a template; callers enforce published source and preflight proof."""
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be verified positive and <=10 u/h")
    move_units.load_ledger(
        ledger, move_units.namespace(commit, canary=canary), move_units.allocation(canary=canary)
    )
    timeout, estimate, seconds, cap, reserve = (
        (24, 10, 600, 4.0, 2 / 3) if canary else (192, 80, 4800, 32.0, 16 / 3)
    )
    env = {
        "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/move_requirements.txt",
        "PYTHONUNBUFFERED": "1",
        "GPU_SIM_SPIKE_COMMIT": commit,
        "MOVE_BENCH_DEVICE": "cuda",
        "MOVE_BENCH_SECONDS": str(seconds),
        "MOVE_BENCH_LEDGER_PATH": "inputs/verified-ledger.json",
        "MOVE_BENCH_MODE": "--canary" if canary else "",
        "MOVE_BENCH_REPEATS": "3",
        "MOVE_BENCH_WARMUPS": "3",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }
    if canary:
        env.update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    result = {
        "job_id": f"move-bench-a100-{'canary' if canary else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": SCRIPT,
        "claim": "Development MOVE stage only; supplied actions, no labels or release evidence",
        "env": env,
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": estimate,
        },
        "timeout_minutes": timeout,
        "output_dir": str(output_dir.resolve()),
        "inputs": checkpoint_inputs() + _inputs(ledger),
        "budget": {
            "unit_cap": cap,
            "rate_units_per_hour": rate,
            "reserve_units": reserve,
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": [
            "move-bench.json",
            "verified-ledger.json",
            "cpu-parity.json",
            "move-native-tests.json",
            "move-native-tests.log",
        ],
        "resume": {
            "unit_kind": "timing-cell",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": TRANSPORT_PIN,
        "move_envelope": {
            "forecast_minutes": 90,
            "base": {"wall_minutes": 108, "units": 18, "reserve_units": 3},
            "effective": {"wall_minutes": 216, "units": 36, "reserve_units": 6},
            "scope": "canary/main plus at most two manual mechanical completions combined",
            "matrix": {
                "worlds": [64, 256, 1024],
                "rosters": ["scripted", "frozen", "mixed"],
                "repeats": 3,
                "warmups": 3,
            },
            "kill": "any mismatch INVALID; any E1024 MOVE plain median speedup <2 STOP_ROUTE",
        },
    }
    if canary:
        result["canary"] = {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
    elif canary_receipt:
        result["canary"] = {
            "receipt": {
                "path": str(canary_receipt.resolve(strict=True)),
                "sha256": sha256_file(canary_receipt),
            }
        }
    else:
        result["canary"] = {"required": "same-pin original broker MOVE canary receipt"}
    return result


def prepare(
    art: Path,
    commit: str,
    preflight: Path | None,
    rate: float,
    *,
    templates_only: bool = False,
    canary_receipt: Path | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    """Templates never confer admission; INVALID proof creates only a refusal."""
    a1_prof_pack.require_source_pin(commit)
    art.mkdir(parents=True, exist_ok=True)
    binding = None
    if not templates_only:
        if preflight is None:
            raise ManifestRefused("current-pin CPU preflight required")
        try:
            binding = require_preflight(preflight, commit)
        except (ValueError, OSError, KeyError) as exc:
            refusal = art / "DISPATCH_REFUSED.json"
            if not refusal.exists():
                move_units.publish_json(
                    refusal,
                    {
                        "status": "INVALID",
                        "commit": commit,
                        "reason": str(exc),
                        "preflight": str(preflight.resolve()),
                        "gpu_units_spent": 0,
                    },
                )
            raise
    for canary in (True, False):
        ledger = art / ("canary-ledger/verified-ledger.json" if canary else "verified-ledger.json")
        if not ledger.exists():
            move_units.publish_json(
                ledger,
                move_units.empty_ledger(
                    move_units.namespace(commit, canary=canary),
                    move_units.allocation(canary=canary),
                ),
            )
        job = manifest(
            commit=commit,
            output_dir=art / ("canary-output" if canary else "main-output"),
            ledger=ledger,
            canary=canary,
            rate=rate,
            canary_receipt=canary_receipt,
        )
        if binding:
            job["mac_preflight"] = binding
        job = convert_public_checkpoints(job, repo=ROOT)
        assert_upload_budget(job, repo=ROOT)
        runnable = not templates_only and (canary or canary_receipt is not None)
        if not runnable:
            job["scientific_admission"] = False
            job["admission_reason"] = "TEMPLATE_ONLY: exact CPU proof and canary binding required"
        suffix = "" if runnable else "-template"
        path = art / f"move-bench-{'canary' if canary else 'main'}{suffix}.json"
        text = json.dumps(job, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text() != text:
            raise ManifestRefused(f"immutable manifest differs: {path}")
        if not path.exists():
            write_once(path, text)
        if validate and runnable:
            from research.colab_cli.supervisor import validate as validate_manifest

            validate_manifest(job, repo=ROOT)
    return {
        "prepared": True,
        "templates_only": templates_only,
        "commit": commit,
        "artifact_dir": str(art),
        "dispatchable": binding is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--rate-units-per-hour", type=float, default=10.0)
    parser.add_argument("--templates-only", action="store_true")
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.artifact_dir.resolve(),
                args.commit or git_head(),
                args.preflight,
                args.rate_units_per_hour,
                templates_only=args.templates_only,
                canary_receipt=args.canary_receipt,
                validate=args.validate,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
