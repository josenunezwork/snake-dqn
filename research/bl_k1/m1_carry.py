"""Post-data R1/R2 reducer; run once only after the attempt receipt is published.

Authority: JUDGMENT_M1_RESTART_OBJECTION.md section 2. Inputs are exclusively
receipt.json and the immutable discarded-run exports, with their sidecars.
No retained cell, fixture bank, timing harness, or prior attempt is opened.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from research.bl_k1 import m1
from research.bl_mac_latency import l1
from research.bl_served_search.exports import export_json

SCHEMA = "bl-plan-v3-m1-carried/v1"
JUDGMENT_SHA256 = "160f5980796a9f0ca21e9282329725594a59522c8ea51cea98e61fd1b813f91c"
ESTIMAND = (
    "the first complete cell under the registered quiet-screen (20 % x 3 x 10 s) "
    "and foreign-load (>= 50 % one core, 2 s) restart protocol, "
    "<= 3 restarts/cell, <= 8 discards/attempt"
)
PRECEDENCE = {"PASS": 0, "BLOCKED": 1, "INCOMPLETE": 2, "INVALID": 3}


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected non-negative integer timestamp/count")
    return value


def disclose_discard(row: dict, record: dict) -> dict:
    """Compute all prefix statistics, carrying only clean irreversible events."""
    cell = next(c for c in m1.cells() if c["cell_id"] == record["cell_id"])
    if any(row.get(key) != value for key, value in cell.items()):
        raise ValueError("discard cell configuration differs from frozen cells")
    clean = []
    for host in row["host"]:
        timestamp = _integer(host["monotonic_ns"])
        if not isinstance(host["foreign_heavy_processes"], list):
            raise ValueError("invalid foreign process list")
        if host["foreign_heavy_processes"] == []:
            clean.append(timestamp)
    t_clean = max(clean) if clean else None  # JSON null represents minus infinity.
    carried_overruns = 0
    for tick in row["ticks"]:
        finished = _integer(tick["finished_ns"])
        if type(tick["overrun"]) is not bool:
            raise ValueError("overrun must be boolean")
        carried_overruns += tick["overrun"] and t_clean is not None and finished <= t_clean
    update = l1.distribution(tick["update_seconds"] for tick in row["ticks"])
    warm: dict[str, list[float]] = {}
    all_phases: dict[str, list[float]] = {}
    carried_doom = 0
    for sample in row["samples"]:
        if "error" in sample or "S_seconds" not in sample:
            continue
        seconds = sample["S_seconds"]
        l1.distribution([seconds])  # Refuse non-finite/negative input.
        all_phases.setdefault(sample["fixture_id"], []).append(seconds)
        visible = _integer(sample["visible_ns"])
        if sample["phase"] == "warm":
            warm.setdefault(sample["fixture_id"], []).append(seconds)
            carried_doom += (
                cell["workers"] == 4
                and sample["fixture_id"] == "frozen-800"
                and seconds > 5.0
                and t_clean is not None
                and visible <= t_clean
            )
    started = _integer(record["started_ns"])
    finished = _integer(record["finished_ns"])
    if finished < started:
        raise ValueError("negative discarded run duration")
    return {
        "cell_id": record["cell_id"],
        "n": record["n"],
        "trigger_processes": row["host"][-1]["foreign_heavy_processes"],
        "started_ns": started,
        "finished_ns": finished,
        "run_duration_ns": finished - started,
        "run_duration_seconds": (finished - started) / 1e9,
        "ticks": len(row["ticks"]),
        "samples_count": len(row["samples"]),
        "T_clean_ns": t_clean,
        "T_clean_null_means": "minus infinity; no clean host row; nothing carried",
        "carried_overruns": carried_overruns,
        "carried_doom": carried_doom,
        "total_prefix_overruns": sum(tick["overrun"] for tick in row["ticks"]),
        "prefix_update_p95_seconds": update["p95_seconds"],
        "prefix_warm_per_fixture_max_S_seconds": {key: max(value) for key, value in warm.items()},
        "prefix_per_fixture_max_S_seconds": {key: max(value) for key, value in all_phases.items()},
        "reported_only_never_pooled": True,
    }


def reduce_attempt(out_dir: Path) -> dict:
    """Verify published inputs and build a monotone carried receipt in memory."""
    out_dir = Path(out_dir)
    receipt, receipt_sha = l1._read_completed_json(out_dir / "receipt.json")
    namespace = receipt["namespace"]
    if receipt.get("schema") != m1.SCHEMA or not isinstance(namespace, str) or not namespace:
        raise ValueError("foreign M1 receipt")
    if receipt.get("verdict") not in PRECEDENCE:
        raise ValueError("unknown attempt verdict")
    frozen = {cell["cell_id"]: index for index, cell in enumerate(m1.cells())}
    retained = receipt["cells"]
    ids = [cell["cell_id"] for cell in retained]
    if len(ids) != len(set(ids)) or set(ids) - frozen.keys():
        raise ValueError("duplicate or foreign retained cell")
    if any(cell.get("verdict") not in PRECEDENCE for cell in retained):
        raise ValueError("unknown cell verdict")

    # The frozen receipt omits retained epochs. The original provenance audit is
    # still required; with no raw retained rows, epoch disjointness cannot be
    # independently rechecked. Do not open retained exports to fill this gap.
    m1.audit_discarded(out_dir, receipt, [])
    disclosures = []
    input_hashes = {"receipt.json": receipt_sha}
    for record in receipt["discarded_runs"]:
        name = f"{record['cell_id']}.discarded-{record['n']}.json"
        row, checksum = l1._read_completed_json(out_dir / name)
        input_hashes[name] = checksum
        disclosures.append(disclose_discard(row, record))

    result = copy.deepcopy(receipt)
    retained_disclosures = []
    for cell in result["cells"]:
        discards = [d for d in disclosures if d["cell_id"] == cell["cell_id"]]
        reasons = [
            f"carried prefix violation from discarded run {d['n']}"
            for d in discards
            if d["carried_overruns"] > 0 or d["carried_doom"] > 0
        ]
        cell["original_verdict"] = cell["verdict"]
        cell["carried_reasons"] = reasons
        if reasons and cell["verdict"] == "PASS":
            cell["verdict"] = "BLOCKED"
        retained_disclosures.append(
            {
                "cell_id": cell["cell_id"],
                "restarts": len(discards),
                "coordinator_cold": frozen[cell["cell_id"]] == 0 and len(discards) == 0,
                "run_duration_ns": None,
                "run_duration_seconds": None,
                "duration_unavailable_reason": (
                    "frozen receipt does not export retained run started_ns/finished_ns; "
                    "retained raw exports are outside reducer input authority"
                ),
            }
        )
    statuses = [receipt["verdict"], *(cell["verdict"] for cell in result["cells"])]
    if len(retained) != 4:
        statuses.append("INCOMPLETE")
    result.update(
        schema=SCHEMA,
        original_verdict=receipt["verdict"],
        verdict=max(statuses, key=PRECEDENCE.__getitem__),
        judgment_sha256=JUDGMENT_SHA256,
        input_sha256=input_hashes,
        estimand=ESTIMAND,
        discarded_run_disclosures=disclosures,
        retained_cell_disclosures=retained_disclosures,
        discarded_timings_pooled=False,
        retained_epoch_reaudit="unavailable in receipt; original audit not independently repeated",
    )
    result["verdict_label"] = (
        f"PASS under the restart protocol, {len(disclosures)} discards"
        if result["verdict"] == "PASS" and disclosures
        else result["verdict"]
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """Create receipt.carried.json and sidecars beside a published receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_dir", type=Path)
    args = parser.parse_args(argv)
    result = reduce_attempt(args.attempt_dir)
    exported = export_json(args.attempt_dir / "receipt.carried.json", result)
    print(json.dumps({"verdict": result["verdict_label"], **exported}))


if __name__ == "__main__":
    main()
