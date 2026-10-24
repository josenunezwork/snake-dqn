"""Create the one R6.3 joint D256/D1024 funding look without gameplay data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f2, ra_guard
from research.bl_residual.heldout import evaluate as evaluate_heldout
from research.bl_residual.heldout import preflight_look


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _life_records(look: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = look.get("metrics", {})
    rows = metrics.get("life_retention")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("held-out look lacks reserve life-macro retention records")
    if metrics.get("reserve_lives_with_zero_decisive_rows") != 0:
        raise ValueError("reserve contains a life with zero decisive rows")
    if len({row.get("life_id") for row in rows}) != 64 or any(
        not isinstance(row.get("parent_world"), str)
        or not row["parent_world"]
        or not isinstance(row.get("decisive_rows"), int)
        or row["decisive_rows"] <= 0
        or not isinstance(row.get("retained_rows"), int)
        or not 0 <= row["retained_rows"] <= row["decisive_rows"]
        or not isinstance(row.get("retention"), (int, float))
        or not math.isclose(
            float(row["retention"]),
            float(Fraction(row["retained_rows"], row["decisive_rows"])),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in rows
    ):
        raise ValueError("reserve retention is incomplete")
    return rows


def _cluster_bootstrap(
    delta: list[dict[str, Any]], *, seed: int = 3403, resamples: int = 10_000
) -> float:
    """Parent-cluster percentile LB90, preserving every life in each parent."""
    by_parent: dict[str, list[float]] = {}
    for row in delta:
        by_parent.setdefault(str(row["parent_world"]), []).append(float(row["delta"]))
    parents = sorted(by_parent)
    if not parents:
        raise ValueError("no parents for clustered bootstrap")
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for sample in range(resamples):
        picked = generator.integers(0, len(parents), size=len(parents))
        values = [float(value) for index in picked for value in by_parent[parents[index]]]
        samples[sample] = np.mean(values)
    return float(np.quantile(samples, 0.10, method="linear"))


def evaluate(
    d256: dict[str, Any],
    d1024: dict[str, Any],
    f2: dict[str, Any],
    *,
    reserve_identity: dict[str, Any],
    f2_timing_eligible: bool = False,
) -> dict[str, Any]:
    """Evaluate the registered funding branch from immutable reserve-look records."""
    for expected, look in ((256, d256), (1024, d1024)):
        if look.get("schema") != "bl-residual-heldout/v2" or look.get("dose") != expected:
            raise ValueError("foreign held-out look schema")
        if look.get("checkpoint_sha256") is None or look.get("archive_receipt_sha256") is None:
            raise ValueError("funding look requires frozen checkpoint and archive receipts")
        if look.get("natural_archive_complete") is not True or not all(
            look.get(key) for key in ("namespace", "recipe_sha256", "reserve_allocation_sha256")
        ):
            raise ValueError("funding look requires complete natural archive")
    if (
        d256["namespace"] != d1024["namespace"]
        or d256["recipe_sha256"] != d1024["recipe_sha256"]
        or d256["reserve_allocation_sha256"] != d1024["reserve_allocation_sha256"]
        or d256["archive_receipt_sha256"] != d1024["archive_receipt_sha256"]
        or d256.get("table_sha256") != d1024.get("table_sha256")
        or d256["checkpoint_sha256"] == d1024["checkpoint_sha256"]
    ):
        raise ValueError("joint look identities are not the same F4 series")
    if any(
        look["namespace"] != reserve_identity["series_id"]
        or look["reserve_allocation_sha256"] != reserve_identity["reserve_ancestry_sha256"]
        or look.get("reserve_life_ids") != reserve_identity["reserve_life_ids"]
        for look in (d256, d1024)
    ):
        raise ValueError("held-out ancestry differs from the verified frozen control reserve")
    left = {row["life_id"]: row for row in _life_records(d256)}
    right = {row["life_id"]: row for row in _life_records(d1024)}
    if set(left) != set(right) or sorted(left) != reserve_identity["reserve_life_ids"]:
        raise ValueError("D256/D1024 reserve lives differ")
    delta = []
    for life in sorted(left):
        if left[life].get("parent_world") != right[life].get("parent_world"):
            raise ValueError("reserve life changed parent")
        delta.append(
            {
                "life_id": life,
                "parent_world": left[life]["parent_world"],
                "delta": Fraction(right[life]["retained_rows"], right[life]["decisive_rows"])
                - Fraction(left[life]["retained_rows"], left[life]["decisive_rows"]),
            }
        )
    mean_delta_exact = sum((row["delta"] for row in delta), Fraction()) / len(delta)
    mean_delta = float(mean_delta_exact)
    lb90 = _cluster_bootstrap(delta)
    source_sha = f2.get("source_sha256")
    published_nonstrike = (
        f2_timing_eligible
        and f2.get("schema") == "bl-residual-f2-disposition/v1"
        and f2.get("volume_struck") is False
        and f2.get("published") is True
        and isinstance(source_sha, str)
        and len(source_sha) == 64
        and all(character in "0123456789abcdef" for character in source_sha)
        and isinstance(f2.get("published_ref"), str)
        and bool(f2["published_ref"])
    )
    f2_disposition = (
        "DOES_NOT_STRIKE_VOLUME" if published_nonstrike else "UNPUBLISHED_APPLY_STRIKES_BRANCH"
    )
    harm_pass = d1024["metrics"].get("harmful_override_count_natural") == 0
    branch_pass = mean_delta_exact >= Fraction(1, 20) and lb90 > 0 and harm_pass
    verdict = "PASS_D4096_FUNDING" if published_nonstrike or branch_pass else "STOP_FUNDING"
    return {
        "schema": "bl-residual-funding-look/v1",
        "series_id": d256["namespace"],
        "registration_sha256": d256.get("registration_sha256"),
        "reserve_allocation_sha256": reserve_identity["reserve_allocation_sha256"],
        "reserve_ancestry_sha256": reserve_identity["reserve_ancestry_sha256"],
        "recipe_sha256": d256["recipe_sha256"],
        "d256_fit_sha256": d256["checkpoint_sha256"],
        "d1024_fit_sha256": d1024["checkpoint_sha256"],
        "doses": [256, 1024],
        "inputs": {
            "d256_checkpoint_sha256": d256["checkpoint_sha256"],
            "d1024_checkpoint_sha256": d1024["checkpoint_sha256"],
            "d256_archive_receipt_sha256": d256["archive_receipt_sha256"],
            "d1024_archive_receipt_sha256": d1024["archive_receipt_sha256"],
            "f2_receipt_sha256": hashlib.sha256(_canonical(f2)).hexdigest(),
        },
        "mean_delta_life_retention": mean_delta,
        "parent_cluster_bootstrap": {"seed": 3403, "resamples": 10_000, "lower90": lb90},
        "harm_screen_d1024_pass": harm_pass,
        "f2_disposition": f2_disposition,
        "verdict": "PASS" if verdict == "PASS_D4096_FUNDING" else verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pin", required=True)
    parser.add_argument("--d4096-start-receipt", type=Path)
    parser.add_argument("--d256", type=Path)
    parser.add_argument("--d1024", type=Path)
    parser.add_argument("--d256-checkpoint", type=Path)
    parser.add_argument("--d1024-checkpoint", type=Path)
    parser.add_argument("--d256-table", type=Path)
    parser.add_argument("--d1024-table", type=Path)
    parser.add_argument("--d256-allocation", type=Path)
    parser.add_argument("--d1024-allocation", type=Path)
    parser.add_argument("--d256-control", type=Path)
    parser.add_argument("--d1024-control", type=Path)
    parser.add_argument("--archive-receipt", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--b8", type=Path)
    parser.add_argument("--look-lock", type=Path)
    parser.add_argument("--command-packet", type=Path)
    parser.add_argument("--f2-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("one create-only R-A funding look")
    # Verify result-ledger publication before reading a receipt byte, then bind
    # the exact receipt bytes once.  Retaining this parsed value prevents a
    # changed receipt from steering the final branch after either held-out look.
    f2_binding = ra_guard.receipt_binding(args.f2_receipt) if args.f2_receipt else None
    publication = ra_guard.publication_evidence(f2_binding)
    if args.f2_receipt and publication is not None:
        semantic = f2.verify_published_receipt(args.f2_receipt)
        if any(semantic.get(key) != value for key, value in f2_binding.items()):
            raise ValueError("F2 receipt binding changed before semantic read")
        verify_export(args.f2_receipt)
        f2_bytes = args.f2_receipt.read_bytes()
        if hashlib.sha256(f2_bytes).hexdigest() != f2_binding["sha256"]:
            raise ValueError("F2 receipt changed after its published binding was verified")
        f2_value = json.loads(f2_bytes)
    else:
        f2_value = {}
    joint = all(
        value is not None
        for value in (
            args.d256_checkpoint,
            args.d1024_checkpoint,
            args.d256_table,
            args.d1024_table,
            args.d256_allocation,
            args.d1024_allocation,
            args.d256_control,
            args.d1024_control,
            args.archive_receipt,
            args.approval,
            args.b8,
            args.look_lock,
            args.command_packet,
        )
    )
    if joint:
        # Refuse a receipt substitution before the R-A lock is created or any
        # held-out table can be opened.  The preparer sealed exactly this
        # binding after independently verifying publication.
        verify_export(args.command_packet)
        packet = json.loads(args.command_packet.read_text())
        from research.bl_residual.f4_ra_prepare import (
            PACKET_SCHEMA,
            existing_ra_locks,
            lock_search_roots,
        )

        if packet.get("schema") != PACKET_SCHEMA or packet.get("status") != "PREPARED_NOT_RUN":
            raise ValueError("R-A command packet is not a sealed prepared packet")
        if packet.get("f2_receipt") != f2_binding:
            raise ValueError("R-A command packet F2 binding differs from --f2-receipt")
        invocation = {
            "--" + name.replace("_", "-"): str(value)
            for name, value in vars(args).items()
            if value is not None
        }
        if invocation != packet.get("invocation") or args.d256 or args.d1024:
            raise ValueError("R-A invocation differs from the sealed command packet")
        if packet.get("source_pin") != args.source_pin or packet.get(
            "source_identity"
        ) != ra_guard.source_identity(args.source_pin):
            raise ValueError("R-A source pin differs from the sealed command packet")
        expected_inputs = {
            option: {"path": path, "sha256": digest(Path(path))}
            for option, path in invocation.items()
            if option not in ("--out", "--look-lock", "--command-packet", "--source-pin")
        }
        if expected_inputs != packet.get("input_identities"):
            raise ValueError("R-A input identities differ from the sealed command packet")
        timing = ra_guard.timing_identity(args.d4096_start_receipt, publication)
        if timing != packet.get("timing"):
            raise ValueError("R-A timing evidence differs from the sealed command packet")
        reserve = ra_guard.reserve_identity(
            args.d256_control, args.d1024_control, args.d256_allocation, args.d1024_allocation
        )
        left = preflight_look(
            args.d256_checkpoint,
            args.d256_allocation,
            args.d256_control,
            args.approval,
            args.archive_receipt,
        )
        right = preflight_look(
            args.d1024_checkpoint,
            args.d1024_allocation,
            args.d1024_control,
            args.approval,
            args.archive_receipt,
        )
        lock = ra_guard.lock_identity(reserve, left, right)
        expected_lock = ra_guard.claim_path(reserve)
        if (
            packet.get("lock") != lock
            or args.look_lock != expected_lock
            or packet.get("lock_path") != str(expected_lock)
            or packet.get("funding_output") != str(args.out)
        ):
            raise ValueError("R-A lock/output differs from the sealed command packet")
        legacy = existing_ra_locks(args.d256_allocation.parent, *lock_search_roots())
        if legacy:
            raise FileExistsError(f"R-A already opened (once-only look): {legacy}")
        # Check eligibility immediately before the atomic claim. A failed/partial
        # claimed look stays spent, including a competing invocation or later F2.
        if (
            ra_guard.timing_identity(
                args.d4096_start_receipt, ra_guard.publication_evidence(f2_binding)
            )
            != timing
        ):
            raise ValueError("R-A timing evidence changed before the claim")
        claim_time = ra_guard.trusted_time()
        claimed_epoch = claim_time["epoch"]
        eligible = ra_guard.timing_eligible(timing, at=claimed_epoch)
        claim = {
            **lock,
            "claimed_epoch": claimed_epoch,
            "claim_time_observation": claim_time,
            "command_packet_sha256": digest(args.command_packet),
            "timing": timing,
            "f2_timing_eligible": eligible,
        }
        export_json(expected_lock, claim)
        d256 = evaluate_heldout(
            args.d256_checkpoint,
            args.b8,
            args.d256_table,
            allocation_path=args.d256_allocation,
            control_path=args.d256_control,
            approval_path=args.approval,
            archive_receipt=args.archive_receipt,
            joint_claim_path=expected_lock,
        )
        d1024 = evaluate_heldout(
            args.d1024_checkpoint,
            args.b8,
            args.d1024_table,
            allocation_path=args.d1024_allocation,
            control_path=args.d1024_control,
            approval_path=args.approval,
            archive_receipt=args.archive_receipt,
            joint_claim_path=expected_lock,
        )
    else:
        raise ValueError("provide a complete joint R-A input set")
    export_json(
        args.out,
        {
            **evaluate(
                d256, d1024, f2_value, reserve_identity=reserve, f2_timing_eligible=eligible
            ),
            "r_a_claim_sha256": digest(expected_lock),
            "r_a_claimed_epoch": claimed_epoch,
            "claim_time_observation": claim_time,
            "timing": timing,
        },
    )


if __name__ == "__main__":
    main()
