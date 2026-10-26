"""Freeze an explicit metadata-only F5 reconciliation input plan.

This finite list is evidence of observed register coverage, not an assertion
that the operator's campaign records or protected banks are complete.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from research.bl_residual.f5_adoption_collision_20261026.reconcile import (
    SCHEMA,
    canonical,
    digest,
    require,
    safe_path,
)

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO.parent / "snake-dqn-artifacts"
PRIOR = REPO / "research/bl_residual/f5_r10a_closure_20261026"
AS = ARTIFACTS / "big-lever/exp/astra-20261024"


def ref(path: Path, expected: str | None = None) -> dict:
    """Freeze only an explicit lawful metadata file; never follow its references."""
    safe_path(path)
    require(path.is_file() and path.stat().st_size < 8 * 1024 * 1024, "metadata source bound")
    value = {"path": str(path), "sha256": digest(path.read_bytes())}
    if expected:
        require(value["sha256"] == expected, "authority hash mismatch")
    return value


def make_plan() -> dict:
    registers = []

    def add(path: Path, kind: str, classes: list[str], state: str, projection: str) -> None:
        registers.append(
            {
                **ref(path),
                "kind": kind,
                "classes": classes,
                "lifecycle": state,
                "projection": projection,
                "ancestry_status": (
                    "source-bound roots; campaign aliases require custodian"
                    if kind == "collector_root_audit"
                    else "metadata evidence only; not a parent catalog"
                ),
            }
        )

    for name in (
        "P1_E8_04_ALLOCATION_V3_INDEX.json",
        "P1_E8_04_ALLOCATION_V3_INDEX_02.json",
        "P1_E8_04_ALLOCATION_V3_INDEX_03.json",
    ):
        add(
            REPO / "research/bl_ppo" / name,
            "register_metadata",
            ["training_development", "reserved_failed_aborted_retried_superseded"],
            "superseded" if "_03" not in name else "published package index",
            "whole metadata-only index; references never followed",
        )
    history = (
        ARTIFACTS
        / "big-lever/exp/bl-p1/y3/e8-04-history-inventory-01/collection-history-inventory.json"
    )
    require(
        ref(history)["sha256"]
        == "7a182e7369745c423fd3c53b74cad98becc963c71eef973b998d533d8d2b7e14",
        "operator history identity",
    )
    add(
        history,
        "register_metadata",
        ["training_development", "reserved_failed_aborted_retried_superseded"],
        "43 source/preparation families, including unexecuted; limited P1 scope",
        "Claude/operator 43-row inventory; no training-status/payload references followed",
    )
    for parent in (
        REPO / "research/bl_ppo/e8_04_allocation",
        ARTIFACTS / "big-lever/exp/bl-p1/y3/e8-04-allocation-v3-02-20261022",
    ):
        for name in ("collection-allocation-audit.json", "canary-collection-allocation-audit.json"):
            add(
                parent / name,
                "collector_root_audit",
                [
                    "training_development",
                    "probe_canary_timing",
                    "reserved_failed_aborted_retried_superseded",
                ],
                "all burned/reserved/rejected and allocated roots retained",
                "burned_roots + hash-verified allocated_roots only; "
                "do not mistake fit seed/world ID for physical root",
            )
    add(
        ARTIFACTS
        / "big-lever/exp/bl-p1/y3/e8-04-allocation-v3-02-20261022"
        / "collection-allocation-coverage-v3.json",
        "register_metadata",
        ["training_development", "reserved_failed_aborted_retried_superseded"],
        "rejected offsets 3000000/3001000; advanced pair 3010000/3011000; source04 nonrun",
        "metadata family enumeration/ranges, rejection/supersession provenance",
    )
    for name in (
        "qualified-72b6fcf0-r302",
        "qualified-ce73d168-r302",
        "sol-prepared-9ac451c0-r302",
        "qualified-ad163339-r302",
    ):
        parent = ARTIFACTS / "big-lever/round-30" / name
        add(
            parent / "inputs/control.json",
            "r30_control",
            ["r30_2_registered", "probe_canary_timing"],
            "synthetic prepared package; dispatch state not inferred",
            "schema, stage, namespace and synthetic=true; fixture has no physical world allocation",
        )
        add(
            parent / "job.json",
            "register_metadata",
            [
                "r30_2_registered",
                "probe_canary_timing",
                "reserved_failed_aborted_retried_superseded",
            ],
            "registered synthetic package; main admission HOLD",
            "manifest identity only; no output, checkpoint or archive references followed",
        )
    timing = ARTIFACTS / "big-lever/round-30/bl-r30-postf4-20261025-v1"
    for name in (
        "native-canary-01",
        "timing-calibration-01",
        "timing-calibration-probe-01",
        "timing-calibration-probe-02",
        "timing-calibration-probe-03",
        "timing-calibration-probe-04",
    ):
        add(
            timing / name / "job.json",
            "register_metadata",
            ["r30_1", "probe_canary_timing", "reserved_failed_aborted_retried_superseded"],
            "prepared/reserved/supersession family; terminal state not inferred from filename",
            "manifest only; timing source reuses admitted rows; "
            "no independent world bank is inferred",
        )
    return {
        "schema": SCHEMA,
        "register_snapshot_cutoff_utc": datetime.now(timezone.utc).isoformat(),
        "catalog": ref(
            PRIOR / "collision-evidence/collision-catalog.json",
            "0cd81fff540013750866e697da92c12c6be62f6ec89865f7ee6eb7e2c2365588",
        ),
        "external_report": ref(
            PRIOR / "external-collision-evidence-v2/external-collision-report.json",
            "09c1160de51e6cc159417e15b6afd4ca6bf9b328255afb85b7703662a8e483b7",
        ),
        "annex": ref(
            REPO / "research/bl_residual/f5_r10a_20261025/F5_R10A_METADATA_ANNEX_20261025.json",
            "8ee06291b23a1ef8c657b59a4f4990349ae4c524a680c47f06785a47d227f764",
        ),
        "r30_inventory": ref(
            timing / "metadata-v3/corpus-inventory.json",
            "1dd5b4e42940d9f03cab8d735f8e66645de82fe1a6c5734b0a742540c125528e",
        ),
        "authority": [
            ref(
                AS / "JUDGMENT_F5_ADOPTION_REMAINING_20261026T015527Z.md",
                "e74e8f32cd1312da9949902a0376f4ed77724966bb6c9377b801596aeff276cb",
            ),
            ref(
                AS / "JUDGMENT_F5_ADOPTION_LIMITATIONS_20261026T012513Z.md",
                "cd87764340548d28446e8652fd37427924d892468e986fbfc5cf4e18107cedeb",
            ),
            ref(
                AS / "F5_R10A_GAP_CLOSURE_ANNEX_20261026_v2.json",
                "7e9567325cebf431b6c935890e3074edcb3b62da1709b40670c08c71a967a3ba",
            ),
        ],
        "registers": registers,
        "source_rules": [
            ref(REPO / name)
            for name in (
                "src/core/seeding.py",
                "src/simd_env/rng.py",
                "src/simd_env/batch_sim.py",
                "src/simd_env/grid_sim.py",
                "research/bl_ppo/collector.py",
                "research/bl_ppo/p1_collection_allocation.py",
                "research/bl_round30/admission.py",
                "research/bl_round30/fixture.py",
                "research/bl_round30/timing.py",
                "research/bl_round30/corpus.py",
                "research/bl_round30/batch_plan.py",
            )
        ]
        + [
            ref(
                ARTIFACTS
                / "big-lever/exp/bl-p1/y3/e8-04-history-inventory-01"
                / "CLAUDE_INSPECTION.md",
                "4f08a82ecb99f814cf7af4bf0384f5eb25180cb35d3141e5e9d6624172938ace",
            )
        ],
        "unresolved": [
            {
                "id": "CAMPAIGN_REGISTER_COMPLETENESS",
                "required": (
                    "Claude must verify the campaign/package allocation register cutoff, "
                    "all lifecycle states, all real R30-2 reservations, historical "
                    "strict/probe/serving allocations and source-specific physical seeds, "
                    "ultimate ancestry and aliases. The 50-source catalog and seven ledgers "
                    "contain seed-only records with unproved ancestry. The scoped P1 "
                    "certificate cannot attest campaign completeness."
                ),
            },
            {
                "id": "PROTECTED_COVERAGE",
                "required": (
                    "Lawful external protected metadata mirrors or actual custodian "
                    "certificate binding protected catalog/register digests and the exact "
                    "complete 684-candidate allocation. No such record is supplied; "
                    "protected directories remain unopened."
                ),
            },
            {
                "id": "CUSTODIAN_ATTESTATION",
                "required": (
                    "Dated attributable Claude/operator approval of the hash-bound template "
                    "assertions, all rejection reasons and zero scope/ancestry/alias "
                    "omissions. An unsigned implementer template is not an attestation."
                ),
            },
        ],
    }


def main() -> None:
    plan = make_plan()
    path = Path(__file__).parent / "input-plan-v2.json"
    raw = canonical(plan)
    with path.open("xb") as stream:
        stream.write(raw)
    with path.with_name(path.name + ".sha256").open("x") as stream:
        stream.write(f"{digest(raw)}  {path.name}\n")
    print(
        json.dumps({"path": str(path), "sha256": digest(raw), "registers": len(plan["registers"])})
    )


if __name__ == "__main__":
    main()
