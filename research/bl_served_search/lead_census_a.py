"""PLAN_v3 M0 Part A: read completed L2' shards, without replay or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from research.bl_served_search.chunks import chunk_rows
from research.bl_served_search.exports import digest, export_json, verify_export

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever")
SOURCE = ROOT / "exp/bl-l2/mac-amendment-01"
OUTPUT = ROOT / "exp/bl-c15/lead-census/part-a.json"
BANK = Path(__file__).with_name("BANK_L2.json")
BANK_SHA = "0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
EXPECTED = {"frozen": 13, "mixed": 5, "scripted": 10}
SCHEMA = "bl-c15-lead-census-part-a/v1"


def value_sha(value: Any) -> str:
    """Hash the same finite semantic JSON used by the L2 producer."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def read_sources(source: Path, bank_path: Path = BANK) -> tuple[list[dict], dict[str, str]]:
    """Require all eight complete, unchanged, allocated shards before reading outcomes."""
    if digest(bank_path) != BANK_SHA:
        raise ValueError("wrong L2 bank bytes")
    bank = json.loads(bank_path.read_text())
    paths = sorted(source.glob("chunk-0*/shard.json"))
    if len(paths) != 8 or [p.parent.name for p in paths] != [f"chunk-{i:02d}" for i in range(8)]:
        raise ValueError("expected exactly chunk-00..07/shard.json")
    rows, hashes = [], {str(bank_path.resolve()): BANK_SHA}
    for index, path in enumerate(paths):
        hashes[str(path.resolve())] = verify_export(path)
        shard = json.loads(path.read_text())
        identity = shard["identity"]
        expected = chunk_rows(bank["rows"], index)
        if (
            shard.get("schema") != "bl-served-search-l2-shard/v1"
            or shard.get("status") != "completed"
            or shard.get("clean_exit") is not True
            or identity.get("chunk_index") != index
            or identity.get("chunk_count") != 8
            or identity.get("bank_sha256") != BANK_SHA
            or identity.get("b8_sha256") != B8_SHA
            or identity.get("search_mode") != "exact_stop"
            or identity.get("native_mode") != "stock"
            or shard.get("rows_sha256") != value_sha(shard["rows"])
            or [r["world_id"] for r in shard["rows"]] != [r["world_id"] for r in expected]
        ):
            raise ValueError(f"incomplete or foreign shard: {path}")
        for row, allocation in zip(shard["rows"], expected):
            # Execution resolves checkpoint basenames to absolute paths. Bind all other plan bytes.
            plan = dict(row["plan"])
            plan["specs"] = [
                [kind, Path(ref).name if kind == "checkpoint" else ref]
                for kind, ref in plan["specs"]
            ]
            if (
                any(row[key] != allocation[key] for key in ("world_seed", "mix", "horizon"))
                or plan != allocation["plan"]
                or row["search_mode"] != "exact_stop"
                or row["native_mode"] != "stock"
                or row["baseline"]["batches"] != 0
                or row["baseline"]["branches_evaluated"] != 0
            ):
                raise ValueError("foreign row or searched baseline")
        rows.extend(shard["rows"])
    if len({row["world_id"] for row in rows}) != 160:
        raise ValueError("duplicate or missing worlds")
    return sorted(rows, key=lambda row: row["world_id"]), hashes


def rescue_record(row: dict[str, Any]) -> dict[str, Any]:
    """Extract first bare-doom batch; ranks exclude bare, and are one-based."""
    batches = row["dense"]["batch_diagnostics"]
    decisions = [b["decision"] for b in batches]
    if decisions != sorted(set(decisions)):
        raise ValueError("non-monotone/duplicate batch decisions")
    doom = next(b for b in batches if b["outcomes"][str(b["bare"])]["survives"] is False)
    f = int(doom["decision"])
    bare = doom["outcomes"][str(doom["bare"])]
    death = int(bare["death_decision"])
    d = death - f
    survived = int(bare.get("frames_survived", int(bare["frames_executed"]) - 1))
    if not 0 <= d < 300 or survived != d:
        raise ValueError("bare D/death coordinate mismatch")
    order = doom["evaluation_order"]
    if order[0] != doom["bare"] or len(order) != doom["branches_evaluated"]:
        raise ValueError("bad exact-stop counters/order")
    survivors = [a for a in order if doom["outcomes"][str(a)]["survives"] is True]
    all_die = not survivors
    if all_die and doom["branches_skipped"] != 0:
        raise ValueError("all-die batch skipped alternatives")
    precursors = sum(
        b["chosen"] != b["bare"]
        and not (
            b["outcomes"][str(b["chosen"])]["survives"]
            and not b["outcomes"][str(b["bare"])]["survives"]
        )
        for b in batches
        if b["decision"] < f
    )
    # Trace is post-step mass: the preceding live frame supplies pre-step length at f.
    if f < 1 or not row["trajectories"]["dense_alive"][f - 1]:
        raise ValueError("detection pre-step length unavailable")
    return {
        "world_id": row["world_id"],
        "world_seed": row["world_seed"],
        "mix": row["mix"],
        "horizon": row["horizon"],
        "plan": row["plan"],
        "f": f,
        "D": d,
        "bare_death_decision_at_f": death,
        "bare_frames_survived_at_f": survived,
        "bare_frames_executed_at_f": int(bare["frames_executed"]),
        "hero_length_at_f": int(row["trajectories"]["dense_mass"][f - 1]),
        "baseline_death_length": row["baseline"]["death_length"],
        "baseline_death_cause": row["baseline"]["death_cause"],
        "baseline_death_frame": row["baseline"]["death_frame"],
        "branches_evaluated_at_f": doom["branches_evaluated"],
        "evaluation_order_at_f": order,
        "surviving_alternative": survivors[0] if survivors else None,
        "surviving_alternative_q_rank": order.index(survivors[0]) if survivors else None,
        "all_die_delay_ranked_at_f": all_die,
        "delay_only_changes_before_f": precursors,
        "a": None,
        "a_minus_f": None,
        "source_row_sha256": value_sha(row),
    }


def tables(records: list[dict]) -> dict[str, Any]:
    """Retain fixed per-mix rank/all-die/precursor tables without computing LAD."""
    return {
        mix: {
            "rescues": len(selected),
            "surviving_alternative_q_rank": dict(
                sorted(Counter(str(r["surviving_alternative_q_rank"]) for r in selected).items())
            ),
            "all_die_at_f": sum(r["all_die_delay_ranked_at_f"] for r in selected),
            "delay_only_changes_before_f": dict(
                sorted(Counter(str(r["delay_only_changes_before_f"]) for r in selected).items())
            ),
        }
        for mix in (*EXPECTED, "frozen+mixed")
        for selected in [
            [
                r
                for r in records
                if r["mix"] == mix or (mix == "frozen+mixed" and r["mix"] != "scripted")
            ]
        ]
    }


def run(source: Path, output: Path) -> dict[str, Any]:
    """Publish the fixed rescue table with atomic create-only sidecars."""
    started = time.monotonic()
    rows, hashes = read_sources(source)
    records = [rescue_record(r) for r in rows if r["world_ledger"]["productive_rescue"] is True]
    if Counter(r["mix"] for r in records) != EXPECTED:
        raise ValueError("productive rescue denominator differs from PLAN_v3 13/5/10")
    authority = [ROOT / "PLAN_v3.md", ROOT / "round-15/JUDGMENT.md"]
    receipt = {
        "schema": SCHEMA,
        "status": "PART_A_COMPLETE_PART_B_NOT_RUN",
        "development_only": True,
        "serving_or_release": "NOT_GRANTED",
        "input_sha256": hashes,
        "authority_sha256": {str(p): digest(p) for p in authority},
        "producer_sha256": digest(Path(__file__)),
        "rescue_counts": EXPECTED,
        "rows": records,
        "tables": tables(records),
        "definitions": {
            "f": "first dense batch whose bare branch dies within full H300",
            "D": "death_decision - f = alive branch frames; fatal transition excluded",
            "hero_length_at_f": "pre-step length = dense_mass[f-1]",
            "q_rank": "one-based rank among alternatives, excluding bare; null when all die",
            "death_cause": "baseline death cause code; no sealer/topology attribution inferred",
            "a": "PENDING_PART_B_BARE_REPLAY",
        },
        "execution": {
            "nice": os.nice(0),
            "elapsed_seconds": time.monotonic() - started,
            "units": 0,
        },
    }
    export_json(output, receipt)
    verify_export(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if os.nice(0) < 10:
        parser.error("run under nice -n 10")
    receipt = run(args.source, args.out)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "rescue_counts": receipt["rescue_counts"],
                "sha256": digest(args.out),
            }
        )
    )


if __name__ == "__main__":
    main()
