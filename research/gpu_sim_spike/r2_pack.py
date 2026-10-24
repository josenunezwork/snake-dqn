"""Prepare the R2 A100 canary and main packets; this module never dispatches.

Same conventions as the I0 packer (:mod:`integration_pack`): a clean published
source pin with the qualified Colab CLI transport tree, a current-pin Mac
real-entrypoint preflight, hash-bound rulings/protocol, public checkpoint URLs,
one deterministic metadata tar, a resumable timing-cell ledger, a family probe
canary and a main that binds the independently reduced canary receipt.
Caps follow ``R2_PROTOCOL_20261024.md`` (canary 2 u, main 8 u).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from research.bl_upload import assert_upload_budget, convert_public_checkpoints
from research.gpu_sim_spike import a1_prof_pack, r2_units
from research.gpu_sim_spike.make_manifests import (
    ManifestRefused,
    checkpoint_inputs,
    git_head,
    sha256_file,
    write_once,
)
from research.metadata_archive import bundle_inputs

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "research/gpu_sim_spike"
SCRIPT = f"{PACKAGE}/r2_bench.sh"
BASE_PIN = "968f02765f7c86ca27d2947347896e2fe4e6ce23"
TRANSPORT_PIN = "313758255ee5b9f1422a0202c14a80cc32554857"
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever")
RULINGS = (
    (
        ARTIFACTS / "exp/JUDGMENT_POST_I0S_DIRECTION_20261024.md",
        "efad89757569070c1c8f332d4cd9c3d7937172f22456e1822f9c00926eb0e9c3",
    ),
    (
        ARTIFACTS / "exp/ADVICE_POST_I0S_A_20261024.md",
        "4f4ee5b5298805d0e24f2ba6c136118f728541a71d24a1d56519a0fb969d3f13",
    ),
    (
        ARTIFACTS / "round-36/JUDGMENT.md",
        "156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b",
    ),
    (
        ARTIFACTS / "round-28/JUDGMENT.md",
        "89996ae9190165aae08036525d3d1b5dbedf1bbfdbd5fd5f1d2b888b13e925dd",
    ),
    (
        ARTIFACTS / "exp/JUDGMENT_GPU_PARITY_STANDARD_20261023.md",
        "51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b",
    ),
)
PROTOCOL = ROOT / PACKAGE / "R2_PROTOCOL_20261024.md"
LEDGER_DEST = "inputs/verified-ledger.json"
CANARY_LEDGER_DEST = "canary-inputs/verified-ledger.json"
# Frozen envelopes (protocol "Caps and clock"); timeouts are the qualified 2x floor.
CANARY = {"est_minutes": 6, "timeout_minutes": 12, "unit_cap": 2.0, "reserve_units": 1 / 3}
MAIN = {"est_minutes": 24, "timeout_minutes": 48, "unit_cap": 8.0, "reserve_units": 4 / 3}
PRODUCER_SECONDS = {"canary": 420, "main": 2400}
CANARY_RATIO_LIMIT = 1.25
CLOCK = {
    "t0_utc": "2026-10-24T01:00:00+00:00",
    "candidate_freeze_t0_plus_48_utc": "2026-10-26T01:00:00+00:00",
    "speed_decision_t0_plus_72_utc": "2026-10-27T01:00:00+00:00",
    "latest_main_dispatch_utc": "2026-10-27T00:00:00+00:00",
}


def _require_commit(commit: str) -> None:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ManifestRefused("commit must be a full lowercase SHA")


def require_source_pin(commit: str) -> None:
    """Refuse unpublished, dirty or non-R2 sources before an immutable packet."""
    _require_commit(commit)
    if git_head() != commit:
        raise ManifestRefused("packet source is not the requested commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ManifestRefused("R2 source checkout must be clean")
    if "origin/" not in subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    ):
        raise ManifestRefused("R2 source pin must be published")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_PIN, commit], cwd=ROOT, check=False
    )
    if ancestor.returncode != 0:
        raise ManifestRefused("R2 candidate must descend from the qualified I0 runtime pin")
    for path, digest in RULINGS:
        if not path.is_file() or sha256_file(path) != digest:
            raise ManifestRefused("binding ruling/advice hash differs")
    if not PROTOCOL.is_file():
        raise ManifestRefused("R2 protocol is absent")
    from tools.ops.qualification import OPS, check

    check(ROOT, OPS / "colab-cli-qualified.json")
    check(ROOT, OPS / "colab-cli-qualified.json", ref=TRANSPORT_PIN)


def _verify(path: Path) -> str:
    return r2_units._verify(path)


def _ledger_inputs(ledger: Path, prefix: str) -> list[dict[str, str]]:
    body = json.loads(ledger.read_text())
    paths = [ledger] + [ledger.parent / row["path"] for row in body.get("units", [])]
    result = []
    for item in paths:
        for candidate in (item, Path(str(item) + ".sha256"), Path(str(item) + ".complete")):
            result.append(
                {
                    "path": str(candidate.resolve(strict=True)),
                    "sha256": sha256_file(candidate),
                    "dest": prefix + "/" + candidate.relative_to(ledger.parent).as_posix(),
                }
            )
    return result


def _bindings() -> list[dict[str, str]]:
    rows = [
        {
            "path": str(path),
            "sha256": digest,
            "dest": "inputs/rulings/" + path.parent.name + "-" + path.name,
        }
        for path, digest in RULINGS
    ]
    rows.append(
        {
            "path": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "dest": "inputs/R2_PROTOCOL_20261024.md",
        }
    )
    return rows


def preflight_binding(path: Path, commit: str) -> dict[str, Any]:
    """Bind the current-pin Mac CPU real-entrypoint smoke receipt."""
    _verify(path)
    body = json.loads(path.read_text())
    receipt = Path(str(body.get("receipt_path", "")))
    if not receipt.is_file() or _verify(receipt) != body.get("receipt_sha256"):
        raise ManifestRefused("R2 preflight receipt binding differs")
    expected = list(r2_units.allocation(smoke=True))
    if (
        body.get("status"),
        body.get("commit"),
        body.get("device"),
        body.get("completed_units"),
    ) != ("PREFLIGHT", commit, "cpu", expected):
        raise ManifestRefused("R2 needs a current-pin CPU real-entrypoint preflight")
    from research.gpu_sim_spike.benchmark import source_hashes

    receipt_body = json.loads(receipt.read_text())
    if (
        not isinstance(body.get("source_sha256"), dict)
        or body.get("entrypoint_run_wall_seconds", 0) <= 0
        or receipt_body.get("commit") != commit
        or receipt_body.get("status") != "MEASURED_PENDING_MAC_REVIEW"
        or receipt_body.get("completed_units") != expected
        or receipt_body.get("source_sha256") != body["source_sha256"]
        or receipt_body.get("source_sha256") != source_hashes()
    ):
        raise ManifestRefused("R2 preflight is incomplete")
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
    if not receipt.is_file():
        raise ManifestRefused("same-pin R2 canary receipt is absent")
    body = json.loads(receipt.read_text())
    manifest = receipt.parent / "manifest.json"
    if (body.get("status"), body.get("commit"), body.get("stop_verified")) != (
        "PASS",
        commit,
        True,
    ):
        raise ManifestRefused("same-pin native R2 PASS/release receipt required")
    if not manifest.is_file() or sha256_file(manifest) != body.get("manifest_sha256"):
        raise ManifestRefused("R2 canary manifest binding differs")
    saved = json.loads(manifest.read_text())
    if saved.get("script") != SCRIPT or saved.get("mac_preflight") != preflight:
        raise ManifestRefused("R2 canary is not from this exact packet")
    exports = receipt.parent / "exports"
    listed = body.get("exports")
    if not isinstance(listed, list):
        raise ManifestRefused("R2 canary terminal export inventory is absent")
    expected = {row.get("path"): row for row in listed if isinstance(row, dict)}
    actual = {
        file.relative_to(exports).as_posix(): {
            "path": file.relative_to(exports).as_posix(),
            "sha256": sha256_file(file),
            "size": file.stat().st_size,
        }
        for file in exports.rglob("*")
        if file.is_file() and not file.name.endswith((".sha256", ".complete"))
    }
    if len(expected) != len(listed) or expected != actual:
        raise ManifestRefused("R2 canary terminal export inventory differs")
    report = json.loads((exports / "r2-bench.json").read_text())
    if report.get("status") != "MEASURED_PENDING_MAC_REVIEW" or report.get(
        "completed_units"
    ) != list(r2_units.allocation(canary=True)):
        raise ManifestRefused("R2 canary has incomplete coverage")
    from research.gpu_sim_spike import r2_analyze

    reduced = r2_analyze.aggregate([exports], commit, canary=True)
    if reduced.get("verdict") != "PASS_CANARY":
        raise ManifestRefused("independent R2 canary reducer did not pass")
    ratios = {}
    for mix in r2_units.MIXES:
        cell = reduced["cells"][f"cell-{mix}-e64"]
        ratios[mix] = cell["I_prime"] / cell["I"]
    if any(not math.isfinite(v) or v > CANARY_RATIO_LIMIT for v in ratios.values()):
        raise ManifestRefused("R2 canary I'/I E64 ratio voids the frozen main forecast")
    return {
        "path": str(receipt.resolve()),
        "sha256": sha256_file(receipt),
        "analysis": reduced,
        "canary_e64_i_prime_over_i": ratios,
        "canary_elapsed_seconds": body.get("elapsed_seconds"),
        "median_ratio": statistics.median(ratios.values()),
    }


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
    r2_units.load_ledger(ledger, r2_units.namespace(commit), r2_units.allocation())
    r2_units.load_ledger(
        canary_ledger, r2_units.namespace(commit, canary=True), r2_units.allocation(canary=True)
    )
    if template and (canary or canary_receipt):
        raise ManifestRefused("only an unadmitted main may be a template")
    if not canary and not template and canary_receipt is None:
        raise ManifestRefused("R2 main requires the independently reduced native canary")
    envelope = CANARY if canary else MAIN
    binding = None if canary or template else canary_binding(canary_receipt, commit, preflight)
    lane = "canary" if canary else "main"
    result: dict[str, Any] = {
        "job_id": f"gpu-r2-{lane}-{commit[:12]}-01",
        "commit": commit,
        "script": SCRIPT,
        "claim": "R2 engineering timing and pure-optimization identity only; no hero, "
        "label, release or serving claim.",
        "env": {
            "COLAB_CLI_REQUIREMENTS": f"{PACKAGE}/integration_requirements.txt",
            "PYTHONUNBUFFERED": "1",
            "GPU_SIM_SPIKE_COMMIT": commit,
            "GPU_SIM_VECTOR_FLOOD": "static",
            "GPU_SIM_VECTOR_GEOMETRY": "native",
            "GPU_SIM_V8_INDEXED": "0",
            "GPU_SIM_R2_DEBUG": "0",
            "R2_BENCH_DEVICE": "cuda",
            "R2_BENCH_SECONDS": str(PRODUCER_SECONDS[lane]),
            "R2_BENCH_LEDGER_PATH": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "R2_BENCH_SMOKE": "0",
            "R2_BENCH_CANARY": "1" if canary else "0",
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
        + _ledger_inputs(ledger, "inputs")
        + _ledger_inputs(canary_ledger, "canary-inputs")
        + _bindings(),
        "budget": {
            "unit_cap": envelope["unit_cap"],
            "rate_units_per_hour": rate,
            "reserve_units": envelope["reserve_units"],
            "other_assignments": {},
        },
        "lease_seconds": 600,
        "max_export_bytes": 64 * 1024**2,
        "required_exports": [
            "r2-bench.json",
            "native-parity.json",
            "r2-native-parity.json",
            "verified-ledger.json",
        ],
        "resume": {
            "unit_kind": "timing-cell",
            "ledger_path": CANARY_LEDGER_DEST if canary else LEDGER_DEST,
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "source_paths": a1_prof_pack._source_paths(),
        "transport_pin": TRANSPORT_PIN,
        "mac_preflight": preflight,
        "r2_envelope": {
            "ruling": {
                "path": str(RULINGS[0][0]),
                "sha256": RULINGS[0][1],
                "verdict": "FIX_PADDING_AS_R2_AND_TEST_IDENTITY",
                "step": 4,
            },
            "base_pin": BASE_PIN,
            "canary": CANARY,
            "main": MAIN,
            "total_cap_units": CANARY["unit_cap"] + MAIN["unit_cap"],
            "parent_row": "round-36 R2 (Q4), effective 4.8 h / 48 u",
            "producer_deadline_seconds": PRODUCER_SECONDS[lane],
            "clock": CLOCK,
            "allocation": list(r2_units.allocation(canary=canary)),
            "gating_cells": [f"cell-{mix}-e1024" for mix in r2_units.MIXES]
            + [f"exc-{mix}-e256" for mix in r2_units.MIXES],
            "acceptance": "R2_PROTOCOL_20261024.md / r2_analyze.py (frozen before data)",
            "canary_main_ratio_limit": CANARY_RATIO_LIMIT,
        },
        "canary": (
            {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT}
            if canary
            else {"required": "same-pin native R2 canary receipt"}
        ),
    }
    if binding is not None:
        result["canary"] = {"receipt": binding}
    if canary:
        result["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE=PACKAGE)
    result["scientific_admission"] = not template
    if template:
        result["admission_reason"] = "TEMPLATE_ONLY: same-pin R2 canary PASS_CANARY is required"
    result = convert_public_checkpoints(result, repo=ROOT)
    result["inputs"] = bundle_inputs(
        result["inputs"], output_dir.parent / f"metadata-{lane}.tar", reuse_verified=True
    )
    result["env"]["METADATA_ARCHIVE_SHA256"] = next(
        x["sha256"] for x in result["inputs"] if x["dest"] == "metadata.tar"
    )
    assert_upload_budget(result, repo=ROOT)
    return result


def _write(path: Path, text: str) -> None:
    if path.exists() and path.read_text() != text:
        raise ManifestRefused(f"immutable packet differs: {path}")
    if not path.exists():
        write_once(path, text)


def prepare(
    packet: Path,
    commit: str,
    preflight_path: Path,
    *,
    canary_receipt: Path | None = None,
    validate: bool = False,
) -> dict[str, Any]:
    require_source_pin(commit)
    preflight = preflight_binding(preflight_path, commit)
    packet.mkdir(parents=True, exist_ok=True)
    ledgers = (
        (packet / "verified-ledger.json", False),
        (packet / "canary-ledger/verified-ledger.json", True),
    )
    for path, flag in ledgers:
        if not path.exists():
            r2_units.publish_json(
                path,
                r2_units.empty_ledger(
                    r2_units.namespace(commit, canary=flag), r2_units.allocation(canary=flag)
                ),
            )
    main_ledger, canary_ledger = ledgers[0][0], ledgers[1][0]
    canary = manifest(
        commit=commit,
        output_dir=packet / "canary-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
    )
    main = manifest(
        commit=commit,
        output_dir=packet / "main-output",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=canary_receipt,
        template=canary_receipt is None,
    )
    if validate:
        from research.colab_cli.supervisor import validate as validate_manifest

        validate_manifest(main if canary_receipt else canary, repo=ROOT)
    _write(packet / "r2-canary.json", json.dumps(canary, indent=2, sort_keys=True) + "\n")
    _write(
        packet / ("r2-main.json" if canary_receipt else "r2-main-template.json"),
        json.dumps(main, indent=2, sort_keys=True) + "\n",
    )
    return {
        "prepared": True,
        "packet": str(packet),
        "commit": commit,
        "main_admitted": canary_receipt is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.packet,
                args.commit,
                args.preflight,
                canary_receipt=args.canary_receipt,
                validate=args.validate,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
