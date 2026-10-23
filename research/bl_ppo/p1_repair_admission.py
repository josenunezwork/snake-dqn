"""Create admitted amendment-06 siblings from real reviewed prerequisites.

This module never dispatches.  Immutable held packs remain unchanged; live
inventory and rate calibration are the operator's immediate dispatch checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_qualification import terminal

RULING_SHA256 = "fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc"


def _export(path: Path) -> dict[str, Any]:
    verify_export(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected immutable object evidence")
    return value


def conservative_charge(path: Path) -> float:
    """Keep the larger own/gross debit, rounded up to cents; never refund."""
    value = terminal(path)
    rate = value.get("accounting", {}).get("own_rate_units_per_hour")
    before = value.get("usage_before", {}).get("balance")
    after = value.get("usage_after", {}).get("balance")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (rate, before, after)):
        raise ValueError("terminal lacks finite debit/rate evidence")
    if not 0 < rate <= 1.54:
        raise ValueError("terminal rate exceeds ruled 1.54 u/h")
    return (
        math.ceil((max(value["elapsed_seconds"] * rate / 3600, before - after, 0) - 1e-9) * 100)
        / 100
    )


def reviewed(
    *, manifest: Path, codex_v: Path, claude: Path, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require independent review and an exact per-manifest Claude line check."""
    value, review, check = _export(manifest), _export(codex_v), _export(claude)
    if value.get("p1_finalisation_ruling_sha256") != RULING_SHA256:
        raise ValueError("manifest does not bind the finalisation ruling")
    if (
        review.get("status") != "READY_FOR_CLAUDE_CHECK"
        or review.get("unresolved_findings") != []
        or review.get("ruling_sha256") != RULING_SHA256
        or review.get("executable_commit") != value.get("commit")
        or digest(manifest) not in review.get("manifest_sha256", {}).values()
    ):
        raise ValueError("Codex-V review does not tick this exact manifest")
    if (
        check.get("status") != "PASS"
        or check.get("stage") != stage
        or check.get("ruling_sha256") != RULING_SHA256
        or check.get("manifest_sha256") != digest(manifest)
        or check.get("codex_v_sha256") != digest(codex_v)
        or check.get("campaign_log_recorded") is not True
    ):
        raise ValueError("Claude line check is absent or differs from exact manifest/review")
    return value, check


def admit(
    *,
    manifest: Path,
    output: Path,
    codex_v: Path,
    claude: Path,
    stage: str,
    repo: Path,
    canary: Path | None = None,
) -> Path:
    """Admit canary or repair into a fresh sibling, without changing its inputs."""
    if stage not in {"canary", "repair"}:
        raise ValueError("source admission accepts canary or repair only")
    value, check = reviewed(manifest=manifest, codex_v=codex_v, claude=claude, stage=stage)
    expected = (2.00, 30) if stage == "canary" else (6.16, 200)
    if (value["budget"]["unit_cap"], value["timeout_minutes"]) != expected:
        raise ValueError("source invoice differs from ruling")
    value = copy.deepcopy(value)
    if stage == "repair":
        if canary is None:
            raise ValueError("repair admission requires the exact-pin canary terminal receipt")
        row = terminal(canary)
        if row.get("commit") != value["commit"]:
            raise ValueError("canary terminal is not the exact repair pin")
        charge = conservative_charge(canary)
        if 10.11 + charge + 6.16 + 6.16 + 2.00 + 3.86 + 3.86 > 34.50 + 1e-9:
            raise ValueError("actual debit leaves insufficient ruled worst-case envelope")
        value["canary"] = {"receipt": {"path": str(canary.resolve()), "sha256": digest(canary)}}
    elif canary is not None:
        raise ValueError("canary admission cannot consume an unrelated canary receipt")
    value["scientific_admission"] = True
    value.setdefault("transport", {})["dispatch_ready"] = True
    value["amendment06_admission"] = {
        "stage": stage,
        "held_manifest_sha256": digest(manifest),
        "codex_v_sha256": digest(codex_v),
        "claude_line_check_sha256": digest(claude),
        "warning": "WARNING_UNEXPLAINED_DEBIT",
        "live_inventory_and_rate_check_required": True,
    }
    from research.colab_cli.supervisor import validate

    validated = validate(value, repo.resolve())
    export_json(output, validated)
    return output


def verify_replacement_prerequisites(
    *, source: Path, source_supervisor: Path, charged_units: float
) -> dict[str, Any]:
    """Read only durability evidence; a prefix mismatch never opens outcomes."""
    proof, session = _export(source), terminal(source_supervisor)
    if (
        proof.get("schema") != "bl-p1-y3-source/v1"
        or proof.get("status") != "COMPLETE_SOURCE_CONTINUATION"
        or proof.get("origin_mode") != "IMPORTED_VERIFIED_ORIGIN"
        or proof.get("prefix_witness", {}).get("status") != "MATCH"
        or proof.get("assignment_endpoint", "").split("#generation=", 1)[0]
        != session.get("endpoint")
        or not any(row.get("sha256") == digest(source) for row in session.get("exports", []))
    ):
        raise ValueError(
            "replacement requires verified imported source and matching prefix witness"
        )
    if type(charged_units) not in (int, float) or not math.isfinite(charged_units):
        raise ValueError("replacement requires reconciled actual charged units")
    if charged_units < 10.11 or charged_units + 3.86 + 3.86 > 34.50 + 1e-9:
        raise ValueError("replacement and its retry do not fit actual residual")
    return proof


def verify_qualification_prerequisites(
    *, y3: Path, source_supervisor: Path, replacement_supervisor: Path
) -> dict[str, Any]:
    """A full two-replay Y3 PASS and released distinct assignments are mandatory."""
    proof = _export(y3)
    source, replacement = terminal(source_supervisor), terminal(replacement_supervisor)
    if (
        proof.get("schema") != "bl-p1-y3/v1"
        or proof.get("status") != "PASS"
        or proof.get("operational_replay", {}).get("status") != "PASS"
        or not source.get("endpoint")
        or source.get("endpoint") == replacement.get("endpoint")
        or proof.get("source_endpoint") != source.get("endpoint")
        or proof.get("replacement_endpoint") != replacement.get("endpoint")
        or not any(row.get("sha256") == digest(y3) for row in replacement.get("exports", []))
        or not any(
            row.get("sha256") == proof.get("source_receipt_sha256")
            for row in source.get("exports", [])
        )
    ):
        raise ValueError(
            "qualification requires full Y3 PASS from distinct verified released assignments"
        )
    return proof


def main() -> None:
    """Bind only real supplied evidence; missing prerequisites are finite refusals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("canary", "repair"))
    for name in ("manifest", "output", "codex-v", "claude", "repo"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--canary", type=Path)
    args = parser.parse_args()
    print(
        admit(
            manifest=args.manifest,
            output=args.output,
            codex_v=args.codex_v,
            claude=args.claude,
            stage=args.stage,
            repo=args.repo,
            canary=args.canary,
        )
    )


if __name__ == "__main__":
    main()
