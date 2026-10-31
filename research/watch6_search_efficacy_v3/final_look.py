"""Single Mac final-look orchestrator for Watch6 V3 confirmation.

Scans <ops_dir>/manifests/*.json across initial runs, auto-retries, -retry-1
accounting retries, and recover.py runs; enforces verified stop on every V3
runtime, builds final-verified-ledger.json outcome-blind, and executes the one
registered df=95 look only when all 96 pairs are complete and verified.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy_v3 import protocol as p
from research.watch6_search_efficacy_v3.analyze import analyze_verified_union
from research.watch6_search_efficacy_v3.dispatch import DEFAULT_OPS_DIR, _active_output_dir
from research.watch6_search_efficacy_v3.ratify import validate_ruling_receipt
from research.watch6_search_efficacy_v3.resume import PAIR_NAME_RE
from research.watch6_search_efficacy_v3.resume import build as build_verified_ledger
from tools.ops.package_spec import digest, publish_json


def _is_proven_unallocated(ops_dir: Path, name: str, out_dir: Path) -> bool:
    """True iff an ops manifest entry never created a runtime or has a LIVE unallocated proof."""
    if not out_dir.exists() and not (ops_dir / "dispatch" / name).exists():
        return True
    proof_candidates = sorted(ops_dir.glob(f"unallocated-{name}*.json"))
    for proof_path in proof_candidates:
        try:
            proof = json.loads(proof_path.read_text())
        except (OSError, ValueError):
            continue
        if (
            proof.get("schema") == "colab-allocation-unavailable/v1"
            and proof.get("mode") == "LIVE"
            and not (out_dir / "receipt.json").exists()
        ):
            return True
    return False


def run_final_look(
    *,
    ops_dir: Path,
    out_dir: Path,
    ruling_json: Path | None = None,
) -> dict:
    """Audit all V3 ops manifests/receipts, seal final ledger, and take the single look."""
    if platform.system() != "Darwin":
        raise ValueError("final look must run on Mac")
    ruling_receipt = validate_ruling_receipt(ruling_json)

    ops_dir = ops_dir.resolve(strict=True)
    manifests_dir = ops_dir / "manifests"
    if not manifests_dir.is_dir():
        raise ValueError(f"missing ops manifests directory: {manifests_dir}")

    main_receipts: list[Path] = []
    canary_receipts: list[Path] = []
    seen_output_dirs: set[Path] = set()

    for mf_path in sorted(manifests_dir.glob("*.json")):
        if mf_path.name.endswith(".live.json"):
            continue
        data = json.loads(mf_path.read_text())
        binding = data.get("watch6_v3")
        if not isinstance(binding, dict) or binding.get("namespace") != p.NAMESPACE:
            continue
        if binding.get("science") != p.identity():
            raise ValueError(f"foreign science identity under V3 namespace in {mf_path}")

        name = mf_path.stem
        active_out = _active_output_dir(data["output_dir"]).resolve()
        if active_out in seen_output_dirs:
            continue
        if _is_proven_unallocated(ops_dir, name, active_out):
            continue
        seen_output_dirs.add(active_out)

        receipt_path = active_out / "receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"unfinished V3 runtime (missing receipt.json): {active_out}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("stop_verified") is not True or receipt.get("cleanup_error") is not None:
            raise ValueError(f"unstopped or unclean V3 runtime: {receipt_path}")
        exec_manifest = active_out / "manifest.json"
        if not exec_manifest.is_file() or digest(exec_manifest) != receipt.get("manifest_sha256"):
            raise ValueError(f"executed manifest SHA256 mismatch: {receipt_path}")

        if canary.is_canary(data) or data.get("env", {}).get("WATCH6_FIXTURE") == "1":
            canary_receipts.append(receipt_path)
        else:
            main_receipts.append(receipt_path)

    if not canary_receipts:
        raise ValueError("missing V3 L4 canary receipt in ops manifests")
    if not main_receipts:
        raise ValueError("no V3 main-shard receipts found in ops manifests")

    unit_paths: list[Path] = []
    seen_slots: dict[int, Path] = {}
    for receipt_path in main_receipts:
        receipt = json.loads(receipt_path.read_text())
        for row in receipt.get("exports", []):
            rel = str(row.get("path", ""))
            if not rel.startswith("efficacy/"):
                continue
            fname = Path(rel).name
            match = PAIR_NAME_RE.fullmatch(fname)
            if match is None:
                continue
            slot = int(match.group(1))
            candidate = (receipt_path.parent / "exports" / rel).resolve(strict=True)
            if slot in seen_slots:
                raise ValueError(
                    f"duplicate exported slot {slot}: {seen_slots[slot]} vs {candidate}"
                )
            seen_slots[slot] = candidate
            unit_paths.append(candidate)

    out_dir.mkdir(parents=True, exist_ok=False)
    ledger = build_verified_ledger(unit_paths, main_receipts)
    ledger_path = out_dir / "final-verified-ledger.json"
    publish_json(ledger_path, ledger)

    verified_slots = {row["index"] for row in ledger["units"]}
    if len(verified_slots) != p.COUNT:
        missing = sorted(set(range(p.COUNT)) - verified_slots)
        missing_by_shard = {
            str(s): [i for i in missing if i % p.SHARDS == s]
            for s in sorted({i % p.SHARDS for i in missing})
        }
        status = {
            "verdict": "INVALID_OR_INCOMPLETE",
            "verified_count": len(verified_slots),
            "expected_count": p.COUNT,
            "missing_indices": missing,
            "missing_by_shard": missing_by_shard,
            "ledger_path": str(ledger_path),
        }
        publish_json(out_dir / "incomplete-status.json", status)
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: {len(verified_slots)}/{p.COUNT} pairs verified; "
            f"missing shards {missing_by_shard} (no outcome look taken)"
        )

    decision = analyze_verified_union(unit_paths, main_receipts)
    decision["astra_ruling"] = ruling_receipt
    decision["canary_receipt_count"] = len(canary_receipts)
    decision["final_ledger_sha256"] = digest(ledger_path)
    publish_json(out_dir / "efficacy-decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops-dir", type=Path, default=DEFAULT_OPS_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ruling-json", type=Path, default=None)
    args = parser.parse_args()
    decision = run_final_look(
        ops_dir=args.ops_dir,
        out_dir=args.out_dir.resolve(),
        ruling_json=args.ruling_json,
    )
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "replay_serving_decision": decision["replay_serving_decision"],
                "n": decision["n"],
                "death_delta_per_1k": decision["death_delta_per_1k"],
                "death_delta_ci95": decision["death_delta_ci95"],
                "mass_relative_difference": decision["mass_relative_difference"],
                "mass_relative_ci95_fieller": decision["mass_relative_ci95_fieller"],
                "reserve_substitutions_count": decision["reserve_substitutions_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
