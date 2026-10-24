"""Prepare the R6 E-scaling A100 canary and main packets; never dispatches.

Same conventions as :mod:`r2_pack` (clean published pin with the qualified
transport tree, current-pin Mac real-entrypoint preflight, hash-bound authority
inputs, public checkpoint URLs, one metadata tar, resumable cell ledger, family
probe canary, main bound to the independently reduced canary).  Caps follow
``R6_PROTOCOL_20261024.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, r2_pack, r6_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
)
from research.metadata_archive import bundle_inputs

ROOT = r2_pack.ROOT
PACKAGE = r2_pack.PACKAGE
SCRIPT = f"{PACKAGE}/r6_bench.sh"
R2_PIN = "e66a701fd43951dc80e976485ed77113a91dddcb"
ENGINE_FILES = (
    "r2_cache.py",
    "v8_device.py",
    "masks_device.py",
    "anchors_device.py",
    "ego_device.py",
    "vector_device.py",
    "vector_geometry.py",
    "device_sim.py",
    "realistic_benchmark.py",
    "device_policy.py",
    "integration_engine.py",
    "a100_ruler.py",
    "r2_bench.py",
    "r2_identity.py",
)
EXP = r2_pack.ARTIFACTS / "exp"
BINDINGS = (
    (
        EXP / "JUDGMENT_POST_I0S_DIRECTION_20261024.md",
        "efad89757569070c1c8f332d4cd9c3d7937172f22456e1822f9c00926eb0e9c3",
    ),
    (
        r2_pack.ARTIFACTS / "round-36/JUDGMENT.md",
        "156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b",
    ),
    (
        EXP / "gpu-r2-padding-20261024/R2_ANALYSIS.json",
        "0b253f33a0f59fd0992a8521bd1d7e79f5fea702f72f2ad3f79d91d14006158c",
    ),
    (
        EXP / "gpu-r2-padding-20261024/HOLD_STEP5_E_SCALING_20261024.md",
        "cfa39c7dbd66c99bce74ae8ce55dc049b882377dce7fafa6c9ef8391368b9c32",
    ),
)
OWNER_BUDGET = {
    "commit": "0608a2d97145e9235da3888d71a34c65830eae15",
    "path": "docs/STATUS.md",
    "blob_sha256": "846a1252827a31a8735849ad8d5be34a391fe13b31cf28533cd3d440294017e5",
    "decision": "Units never block (owner, 2026-10-24, standing); owner approved R2 step 5 "
    "E2048/E4096 at ~8 u within round-36 R6 12 u row",
    "replaces": "direction-ruling step-5 cap 5 u (budget only; science bars unchanged)",
}
PROTOCOL = ROOT / PACKAGE / "R6_PROTOCOL_20261024.md"
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"
CANARY = {"est_minutes": 6, "timeout_minutes": 12, "unit_cap": 2.0, "reserve_units": 1 / 3}
MAIN = {"est_minutes": 15, "timeout_minutes": 30, "unit_cap": 5.0, "reserve_units": 5 / 6}
COMPILE_CAP = 1.2
PRODUCER_SECONDS = {"canary": 360, "main": 1260}
CLOCK = {
    "t0_utc": "2026-10-24T01:00:00+00:00",
    "speed_decision_t0_plus_72_utc": "2026-10-27T01:00:00+00:00",
    "latest_main_dispatch_utc": "2026-10-27T00:15:00+00:00",
}


def require_source_pin(commit: str) -> None:
    """R2's published/clean/qualified checks plus an unchanged R2 engine."""
    r2_pack.require_source_pin(commit)
    for name in ENGINE_FILES:
        path = f"{PACKAGE}/{name}"
        old = subprocess.check_output(["git", "rev-parse", f"{R2_PIN}:{path}"], cwd=ROOT, text=True)
        new = subprocess.check_output(["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True)
        if old != new:
            raise ManifestRefused(f"R6 must not change the R2 engine source: {name}")
    for path, digest in BINDINGS:
        if not path.is_file() or sha256_file(path) != digest:
            raise ManifestRefused(f"R6 authority input differs: {path}")
    blob = subprocess.check_output(
        ["git", "show", f"{OWNER_BUDGET['commit']}:{OWNER_BUDGET['path']}"], cwd=ROOT
    )
    import hashlib

    if hashlib.sha256(blob).hexdigest() != OWNER_BUDGET["blob_sha256"]:
        raise ManifestRefused("owner budget decision record differs")
    if not PROTOCOL.is_file():
        raise ManifestRefused("R6 protocol is absent")


def _bindings() -> list[dict[str, str]]:
    rows = [
        {"path": str(path), "sha256": digest, "dest": "inputs/authority/" + path.name}
        for path, digest in BINDINGS
    ]
    rows.append(
        {
            "path": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "dest": "inputs/R6_PROTOCOL_20261024.md",
        }
    )
    return rows


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    r6_units._verify(path)
    body = json.loads(path.read_text())
    receipt = Path(str(body.get("receipt_path", "")))
    if not receipt.is_file() or r6_units._verify(receipt) != body.get("receipt_sha256"):
        raise ManifestRefused("R6 preflight receipt binding differs")
    expected = list(r6_units.allocation(smoke=True))
    if (
        body.get("status"),
        body.get("commit"),
        body.get("device"),
        body.get("completed_units"),
    ) != (
        "PREFLIGHT",
        commit,
        "cpu",
        expected,
    ):
        raise ManifestRefused("R6 needs a current-pin CPU real-entrypoint preflight")
    from research.gpu_sim_spike.benchmark import source_hashes

    receipt_body = json.loads(receipt.read_text())
    if (
        receipt_body.get("commit") != commit
        or receipt_body.get("status") != "MEASURED_PENDING_MAC_REVIEW"
        or receipt_body.get("completed_units") != expected
        or receipt_body.get("source_sha256") != body.get("source_sha256")
        or receipt_body.get("source_sha256") != source_hashes()
    ):
        raise ManifestRefused("R6 preflight is incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "receipt_path": str(receipt),
        "receipt_sha256": body["receipt_sha256"],
        "source_sha256": body["source_sha256"],
        "entrypoint_run_wall_seconds": body["entrypoint_run_wall_seconds"],
    }


def canary_binding(receipt: Path, commit: str, preflight: dict[str, Any]) -> dict[str, Any]:
    """Verify a released same-pin canary and reduce it independently."""
    body = json.loads(receipt.read_text())
    manifest = receipt.parent / "manifest.json"
    if (body.get("status"), body.get("commit"), body.get("stop_verified")) != (
        "PASS",
        commit,
        True,
    ):
        raise ManifestRefused("same-pin native R6 PASS/release receipt required")
    if not manifest.is_file() or sha256_file(manifest) != body.get("manifest_sha256"):
        raise ManifestRefused("R6 canary manifest binding differs")
    saved = json.loads(manifest.read_text())
    if saved.get("script") != SCRIPT or saved.get("mac_preflight") != preflight:
        raise ManifestRefused("R6 canary is not from this exact packet")
    exports = receipt.parent / "exports"
    listed = body.get("exports")
    expected = {row.get("path"): row for row in listed or [] if isinstance(row, dict)}
    actual = {
        file.relative_to(exports).as_posix(): {
            "path": file.relative_to(exports).as_posix(),
            "sha256": sha256_file(file),
            "size": file.stat().st_size,
        }
        for file in exports.rglob("*")
        if file.is_file() and not file.name.endswith((".sha256", ".complete"))
    }
    if not listed or len(expected) != len(listed) or expected != actual:
        raise ManifestRefused("R6 canary terminal export inventory differs")
    from research.gpu_sim_spike import r6_analyze

    reduced = r6_analyze.aggregate(exports, commit, canary=True)
    if reduced.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent R6 canary reducer did not pass")
    return {"path": str(receipt.resolve()), "sha256": sha256_file(receipt), "analysis": reduced}


def manifest(
    *,
    commit: str,
    output_dir: Path,
    ledger: Path,
    canary_ledger: Path,
    canary: bool,
    preflight: dict[str, Any],
    canary_receipt: Path | None = None,
    template: bool = False,
    rate: float = 10.0,
) -> dict[str, Any]:
    if not math.isfinite(rate) or not 0 < rate <= 10:
        raise ManifestRefused("A100 rate must be positive and <=10 u/h")
    r6_units.load_ledger(ledger, r6_units.namespace(commit), r6_units.allocation())
    r6_units.load_ledger(
        canary_ledger, r6_units.namespace(commit, canary=True), r6_units.allocation(canary=True)
    )
    if template and (canary or canary_receipt):
        raise ManifestRefused("only an unadmitted main may be a template")
    if not canary and not template and canary_receipt is None:
        raise ManifestRefused("R6 main requires the independently reduced native canary")
    lane = "canary" if canary else "main"
    envelope = CANARY if canary else MAIN
    binding = None if canary or template else canary_binding(canary_receipt, commit, preflight)
    result: dict[str, Any] = {
        "job_id": f"gpu-r6-{lane}-{commit[:12]}-01",
        "commit": commit,
        "script": SCRIPT,
        "claim": "R6 E-scaling engineering evidence for the R2 engine only; no hero, label, "
        "release or serving claim; larger E never replaces the E1024 bar.",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/integration_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "GPU_SIM_V8_INDEXED": "0",
            "GPU_SIM_R2_DEBUG": "0",
            "R6_BENCH_DEVICE": "cuda",
            "R6_BENCH_SECONDS": str(PRODUCER_SECONDS[lane]),
            "R6_BENCH_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "R6_BENCH_SMOKE": "0",
            "R6_BENCH_CANARY": "1" if canary else "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        },
        "requirements": {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 12,
            "est_minutes": envelope["est_minutes"],
        },
        "timeout_minutes": envelope["timeout_minutes"],
        "output_dir": str(output_dir.resolve()),
        "inputs": checkpoint_inputs()
        + r2_pack._ledger_inputs(ledger, "inputs")
        + r2_pack._ledger_inputs(canary_ledger, "canary-inputs")
        + _bindings(),
        "budget": {
            "unit_cap": envelope["unit_cap"],
            "rate_units_per_hour": rate,
            "reserve_units": envelope["reserve_units"],
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": ["r6-bench.json", "native-parity.json", "verified-ledger.json"],
        "resume": {
            "unit_kind": "timing-cell",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": r2_pack.TRANSPORT_PIN,
        "mac_preflight": preflight,
        "r6_envelope": {
            "ruling": {"path": str(BINDINGS[0][0]), "sha256": BINDINGS[0][1], "step": 5},
            "r6_row": "round-36 R6 (Q4), effective 12 u",
            "owner_budget_decision": OWNER_BUDGET,
            "r2_pin": R2_PIN,
            "canary": CANARY,
            "main": MAIN,
            "compile_cap_units": COMPILE_CAP,
            "total_cap_units": CANARY["unit_cap"] + MAIN["unit_cap"] + COMPILE_CAP,
            "producer_deadline_seconds": PRODUCER_SECONDS[lane],
            "clock": CLOCK,
            "allocation": list(r6_units.allocation(canary=canary)),
            "acceptance": "R6_PROTOCOL_20261024.md / r6_analyze.py (frozen before data)",
        },
        "canary": (
            {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
            if canary
            else {"required": "same-pin native R6 canary receipt"}
        ),
    }
    if binding is not None:
        result["canary"] = {"receipt": binding}
    if canary:
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    result["scientific_admission"] = not template
    if template:
        result["admission_reason"] = "TEMPLATE_ONLY: same-pin R6 canary PASS_CANARY is required"
    result = convert_public_checkpoints(result, repo=ROOT)
    result["inputs"] = bundle_inputs(
        result["inputs"], output_dir.parent / f"metadata-{lane}.tar", reuse_verified=True
    )
    result["env"]["METADATA_ARCHIVE_SHA256"] = next(
        x["sha256"] for x in result["inputs"] if x["dest"] == "metadata.tar"
    )
    assert_upload_budget(result, repo=ROOT)
    return result


def ledgers(packet: Path, commit: str) -> tuple[Path, Path]:
    pairs = (
        (packet / "verified-ledger.json", False),
        (packet / "canary-ledger/verified-ledger.json", True),
    )
    for path, flag in pairs:
        if not path.exists():
            r6_units.publish_json(
                path,
                r6_units.empty_ledger(
                    r6_units.namespace(commit, canary=flag), r6_units.allocation(canary=flag)
                ),
            )
    return pairs[0][0], pairs[1][0]


def prepare_main(packet: Path, commit: str, preflight_path: Path, canary_receipt: Path) -> Path:
    """Build, validate and create-only write the canary-bound main (never the canary)."""
    require_source_pin(commit)
    preflight = preflight_binding(preflight_path, commit)
    main_ledger, canary_ledger = ledgers(packet, commit)
    value = manifest(
        commit=commit,
        output_dir=packet / "main-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=canary_receipt,
    )
    from research.colab_cli.supervisor import validate

    validate(value, repo=ROOT)
    path = packet / "r6-main.json"
    r2_pack._write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def prepare(packet: Path, commit: str, preflight_path: Path, *, validate: bool = False) -> dict:
    """Write the canary manifest and an unadmitted main template."""
    require_source_pin(commit)
    if git_head() != commit:
        raise ManifestRefused("packet source is not the requested commit")
    preflight = preflight_binding(preflight_path, commit)
    packet.mkdir(parents=True, exist_ok=True)
    main_ledger, canary_ledger = ledgers(packet, commit)
    common = dict(
        commit=commit, ledger=main_ledger, canary_ledger=canary_ledger, preflight=preflight
    )
    canary = manifest(output_dir=packet / "canary-output", canary=True, **common)
    template = manifest(output_dir=packet / "main-output", canary=False, template=True, **common)
    if validate:
        from research.colab_cli.supervisor import validate as validate_manifest

        validate_manifest(canary, repo=ROOT)
    r2_pack._write(packet / "r6-canary.json", json.dumps(canary, indent=2, sort_keys=True) + "\n")
    r2_pack._write(
        packet / "r6-main-template.json", json.dumps(template, indent=2, sort_keys=True) + "\n"
    )
    return {"prepared": True, "packet": str(packet), "commit": commit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.packet, args.commit, args.preflight, validate=args.validate)))


if __name__ == "__main__":
    main()
