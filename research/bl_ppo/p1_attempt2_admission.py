"""Bind attempt-2 operator evidence into create-only admitted siblings.

The legacy success gate and invoices are preserved; cumulative accounting is a
separate mandatory prerequisite. This module performs no provider operations.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_attempt2_accounting import verify_cumulative_admission
from research.bl_ppo.p1_attempt2_pack import ATTEMPT2_RULING, PIN, TREE, read_export
from research.bl_ppo.p1_attempt2_window import SCHEMA as WINDOW_SCHEMA
from research.bl_ppo.p1_attempt2_window import TIMING_ARMED
from research.bl_ppo.p1_repair_admission import reviewed

CHECK_FIELDS = (
    "origin_invariants_verified",
    "empty_ledger_namespace_verified",
    "publication_mapping_verified",
    "exact_mac_fetch_verified",
    "factory_drain_verified",
    "factory_re_admission_blocked",
    "control_nice_zero_verified",
    "complete_control_tree_renicer_exclusion_verified",
    "quiet_window_through_release_committed",
    "account_inventory_zero_other_assignments",
    "l4_rate_calibrated_le_1_54_zero_tolerance",
    "ac_lid_thermal_x5_k1_m1_verified",
    "attempt_limit_one_canary_one_final_repair",
)
TRANSPORT_RULING = "2fb9e3914a5eb86e9cd5c42495a1d8213fa1475aa6cf0c72d14c83350de5bc5e"
MIRROR_RULING = "f1bd0dca5499c1766addb2145e35dc3e6291c6d9d97f175425c68d7f0d9f3ee9"


def admit(
    *,
    stage: str,
    manifest: Path,
    output: Path,
    codex_v: Path,
    claude: Path,
    cumulative: Path,
    preflight: Path,
    broker: Path,
    canary: Path | None = None,
    canary_verification: Path | None = None,
) -> Path:
    """Require exact independently reviewed bytes and fresh per-stage operator facts."""
    value, check = reviewed(manifest=manifest, codex_v=codex_v, claude=claude, stage=stage)
    review = read_export(codex_v)
    if (
        stage not in {"canary", "repair"}
        or value["commit"] != PIN
        or value.get("p1_attempt2", {}).get("ruling_sha256") != ATTEMPT2_RULING
        or review.get("attempt2_ruling_sha256") != ATTEMPT2_RULING
        or check.get("attempt2_ruling_sha256") != ATTEMPT2_RULING
        or check.get("transport_ruling_sha256") != TRANSPORT_RULING
        or check.get("a3_3_exception_ruling_sha256") != MIRROR_RULING
        or any(check.get(name) is not True for name in CHECK_FIELDS)
        or check.get("cumulative_accounting_sha256") != digest(cumulative)
        or check.get("public_fetch_preflight_sha256") != digest(preflight)
        or check.get("qualified_transport_tree") != TREE
    ):
        raise ValueError("attempt-2 authority, operator conditions or bindings absent")
    expected = (2.0, 0.4, 30) if stage == "canary" else (6.16, 1.03, 200)
    if (
        (value["budget"]["unit_cap"], value["budget"]["reserve_units"], value["timeout_minutes"])
        != expected
        or value["budget"]["rate_units_per_hour"] != 1.54
        or value["budget"]["rate_tolerance_units_per_hour"] != 0
        or value["budget"]["other_assignments"] != {}
    ):
        raise ValueError("frozen stage invoice differs")
    accounting = verify_cumulative_admission(cumulative, stage=stage, held_manifest=manifest)
    fetch = read_export(preflight)
    public = [row for row in value["inputs"] if "url" in row]
    fetched = fetch.get("inputs", [])

    def inventory(rows: list[dict]) -> list[tuple]:
        return sorted((row.get("url"), row.get("sha256"), row.get("size_bytes")) for row in rows)

    elapsed = fetch.get("elapsed_seconds")
    if (
        fetch.get("status") != "PASS"
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or not 0 <= elapsed <= 300
        or len(public) != 9
        or len(fetched) != 9
        or inventory(public) != inventory(fetched)
        or fetch.get("manifest", {}).get("sha256")
        != review.get("manifest_sha256", {}).get("canary-manifest.json")
    ):
        raise ValueError("exact final canary Mac preflight is absent")
    for field in ("HEAD", PIN):
        tree = subprocess.check_output(
            ["git", "rev-parse", f"{field}:research/colab_cli"],
            cwd=broker,
            text=True,
        ).strip()
        if tree != TREE:
            raise ValueError("actual broker HEAD and manifest pin must retain qualified tree")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "research/colab_cli"],
        cwd=broker,
    ).strip():
        raise ValueError("broker qualified transport has uncommitted changes")
    value = copy.deepcopy(value)
    window = check.get("control_window")
    window_fields = ("arm_path", "owner_token", "root_record", "registration_evidence")
    if not isinstance(window, dict) or any(
        not isinstance(window.get(key), str) or not window[key].strip() for key in window_fields
    ):
        raise ValueError("Claude check lacks the owned create-only control-window paths")
    for key in ("root_record", "registration_evidence"):
        if not Path(window[key]).is_absolute() or Path(window[key]).exists():
            raise ValueError("control registration requires fresh absolute paths")
    arm_path = Path(window["arm_path"])
    if arm_path.resolve() != TIMING_ARMED or arm_path.is_symlink():
        raise ValueError("control window must use the canonical timing arm")
    arm = json.loads(arm_path.read_text())
    if (
        arm.get("schema") != WINDOW_SCHEMA
        or arm.get("kind") != "arm"
        or arm.get("owner_token") != window["owner_token"]
        or arm.get("owner") != value["job_id"]
    ):
        raise ValueError("control window does not bind the current stage's owned arm")
    value["p1_attempt2_control_window"] = window
    if stage == "repair":
        if canary is None or canary_verification is None:
            raise ValueError("repair requires new canary and independent bar verification")
        proof = read_export(canary_verification)
        if (
            proof.get("status") != "CANARY_PASS"
            or proof.get("unresolved_findings") != []
            or proof.get("attempt2_ruling_sha256") != ATTEMPT2_RULING
            or proof.get("repair_manifest_sha256") != digest(manifest)
            or proof.get("canary_receipt_sha256") != digest(canary)
            or check.get("canary_verification_sha256") != digest(canary_verification)
            or accounting.get("canary_receipt", {}).get("sha256") != digest(canary)
        ):
            raise ValueError("new canary independent proof does not bind this exact repair")
        value["canary"] = {"receipt": {"path": str(canary.resolve()), "sha256": digest(canary)}}
    elif canary is not None or canary_verification is not None:
        raise ValueError("canary stage cannot consume a prior canary")
    value["scientific_admission"] = True
    value["transport"]["dispatch_ready"] = True
    value["p1_attempt2_admission"] = {
        "stage": stage,
        "held_manifest_sha256": digest(manifest),
        "codex_v_sha256": digest(codex_v),
        "claude_line_check_sha256": digest(claude),
        "cumulative_accounting_sha256": digest(cumulative),
        "public_fetch_preflight_sha256": digest(preflight),
    }
    # Validation executes the frozen broker's actual module, not main's drifted transport.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from pathlib import Path; "
            "from research.colab_cli.supervisor import validate; "
            "print(json.dumps(validate(json.load(sys.stdin),Path.cwd())))",
        ],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
        cwd=broker,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    export_json(output, json.loads(result.stdout))
    verify_export(output)
    return output


def main() -> None:
    """Create admitted bytes only after Claude records the actual live checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("canary", "repair"))
    for name in ("manifest", "output", "codex-v", "claude", "cumulative", "preflight", "broker"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("canary", "canary-verification"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    print(admit(**vars(args)))


if __name__ == "__main__":
    main()
