"""Outcome-blind reuse of retained L4 W4 timings for V3 24x4 shard envelopes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

from research.colab_cli import canary
from research.watch6_search_efficacy import costs as v2_costs
from research.watch6_search_efficacy import protocol as v2_protocol
from research.watch6_search_efficacy.performance import (
    MAX_BASE_MINUTES,
    SETUP_MINUTES,
    validate_measurements,
)
from research.watch6_search_efficacy_v3 import protocol as p
from tools.ops.package_spec import digest, publish_json

BASE_PIN = v2_costs.BASE_PIN
BASE_SCIENCE = v2_costs.BASE_SCIENCE
MARGIN = v2_costs.MARGIN  # 1.5
REPLAY_HORIZON = v2_costs.REPLAY_HORIZON  # 6000
QUALIFIED_TRANSPORT_TREE = "36285bc6cbb637b392b234f67b55cbaa03b37614"

EXECUTION_UNCHANGED_PATHS = (
    "src",
    "configs/mechanics_v2.yaml",
    "research/watch6_search_efficacy/engine.py",
    "research/watch6_search_efficacy/product.py",
    "research/watch6_search_efficacy/run.py",
    "research/watch6_search_efficacy/uploads.py",
)
ALLOWED_WEB_BACKEND_DIFFS = {
    "web/backend/session.py",
    "web/backend/watch_replay.py",
}


def compatible_source() -> None:
    """Verify simulation/controller bytes match BASE_PIN and transport is qualified cap-6."""
    changed_exec = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_PIN, "--", *EXECUTION_UNCHANGED_PATHS],
        cwd=p.ROOT,
        text=True,
    ).splitlines()
    untracked_exec = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *EXECUTION_UNCHANGED_PATHS],
        cwd=p.ROOT,
        text=True,
    ).splitlines()
    if changed_exec or untracked_exec:
        raise ValueError(
            f"retained timing execution bytes differ from BASE_PIN: {changed_exec + untracked_exec}"
        )
    changed_web = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", BASE_PIN, "--", "web/backend"],
            cwd=p.ROOT,
            text=True,
        ).splitlines()
    )
    untracked_web = set(
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "--", "web/backend"],
            cwd=p.ROOT,
            text=True,
        ).splitlines()
    )
    if (changed_web | untracked_web) - ALLOWED_WEB_BACKEND_DIFFS:
        raise ValueError("unexpected web/backend diff outside opt-in watch_replay serving")
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:research/colab_cli"], cwd=p.ROOT, text=True
    ).strip()
    if tree != QUALIFIED_TRANSPORT_TREE or canary.executable_identity(
        p.ROOT
    ) != canary.executable_identity(p.ROOT, "HEAD"):
        raise ValueError(f"transport tree must match qualified {QUALIFIED_TRANSPORT_TREE[:8]}")


def collect(canary_output: Path, replay_outputs: list[Path]) -> dict:
    """Verify immutable historical L4 W4 timing receipts without touching scientific pairs."""
    compatible_source()
    manifest, _ = v2_costs.verified_run(canary_output, measured_failure=True)
    report = json.loads((canary_output / "exports/performance.json").read_text())
    if manifest["watch6_efficacy"]["allocation"] != v2_protocol.allocation():
        raise ValueError("retained L4 canary manifest identity differs")
    if len(replay_outputs) != 2 or len(set(replay_outputs)) != 2:
        raise ValueError("requires two distinct historical replay timing banks")
    banks = []
    for output in replay_outputs:
        replay, receipt = v2_costs.verified_run(output)
        if replay["env"].get("WATCH_REPLAY_WORKERS") != "4":
            raise ValueError("replay timing requires measured W4")
        contract_path = output / "exports/sessions/contract.json"
        contract = json.loads(contract_path.read_text())
        operational_path = output / "exports/sessions/operational.json"
        operational = json.loads(operational_path.read_text())
        if (
            contract["horizon"] != REPLAY_HORIZON
            or contract["controller"] != "exact-stop-l2-prime"
            or contract["count"] != 12
            or operational["completed"] != 12
            or operational["total"] != 12
            or operational["skipped"]
            or operational["exclusions"]
            or len(operational["units"]) != 12
            or {u["index"] for u in operational["units"]} != set(range(12))
        ):
            raise ValueError("incomplete replay timing panel")
        banks.append(
            {
                "receipt": {
                    "path": str(output / "receipt.json"),
                    "sha256": digest(output / "receipt.json"),
                },
                "operational": {"path": str(operational_path), "sha256": digest(operational_path)},
                "contract": {"path": str(contract_path), "sha256": digest(contract_path)},
                "source_pin": receipt["commit"],
                "namespace": contract["namespace"],
                "workers": 4,
                "horizon": REPLAY_HORIZON,
                "units": [
                    {
                        "index": u["index"],
                        "seconds": u["seconds"],
                        "frames": u["stats"]["decisions"],
                        "search_roots": u["stats"]["search_roots"],
                    }
                    for u in operational["units"]
                ],
            }
        )
    return {
        "schema": "watch6-empirical-costs/v3",
        "science": p.identity(),
        "measurement_science": BASE_SCIENCE,
        "measurement_pin": BASE_PIN,
        "canary": report,
        "canary_receipt": {
            "path": str(canary_output / "receipt.json"),
            "sha256": digest(canary_output / "receipt.json"),
        },
        "banks": banks,
        "scope": (
            "Operational seconds only from verified L4 W4 runs; "
            "no Watch6 V2 or V3 efficacy pair opened."
        ),
    }


def verify_report_evidence(report: dict) -> None:
    """Reconstruct every timing field from verified original receipts on the Mac."""
    actual = collect(
        Path(report["canary_receipt"]["path"]).parent,
        [Path(bank["receipt"]["path"]).parent for bank in report["banks"]],
    )
    if report != actual:
        raise ValueError("empirical cost evidence differs from verified historical receipts")


def empirical_forecast(report: dict, frozen_layout: dict | None = None) -> dict:
    """Size the registered 24 shards x 4 pairs on L4 W4 with 1.5x margin and 2x wall."""
    if (
        report.get("schema") != "watch6-empirical-costs/v3"
        or report.get("science") != p.identity()
        or report.get("measurement_science") != BASE_SCIENCE
        or report.get("measurement_pin") != BASE_PIN
    ):
        raise ValueError("foreign empirical cost identity")
    validate_measurements(report["canary"], BASE_SCIENCE)
    banks = report["banks"]
    if len(banks) != 2 or len({b["namespace"] for b in banks}) != 2:
        raise ValueError("requires two distinct historical replay banks")
    seconds = []
    for bank in banks:
        if bank["workers"] != 4 or bank["horizon"] != REPLAY_HORIZON or len(bank["units"]) != 12:
            raise ValueError("foreign empirical placement/coverage")
        if {u["index"] for u in bank["units"]} != set(range(12)):
            raise ValueError("duplicate/missing timing units")
        for unit in bank["units"]:
            value = unit["seconds"]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                or unit["frames"] != REPLAY_HORIZON
                or type(unit["search_roots"]) is not int
                or not 0 <= unit["search_roots"] <= 1200
            ):
                raise ValueError("invalid empirical timing unit")
            seconds.append(value)
    ordered = sorted(seconds)
    position = (len(ordered) - 1) * 0.95
    low = math.floor(position)
    p95 = ordered[low] + (position - low) * (ordered[math.ceil(position)] - ordered[low])
    profile = next(r for r in report["canary"]["profiles"] if r["workers"] == p.WORKERS)
    bare_seconds = p.HORIZON * profile["main_seconds_per_frame"]["bare"]
    exact_seconds = p95 * p.HORIZON / REPLAY_HORIZON
    pair_seconds = bare_seconds + exact_seconds

    shards = p.SHARDS
    workers = p.WORKERS
    pairs = math.ceil(p.COUNT / shards)
    waves = math.ceil(pairs / workers)
    required_base = math.ceil(SETUP_MINUTES + MARGIN * waves * pair_seconds / 60)
    if required_base > MAX_BASE_MINUTES:
        raise ValueError("NOT_READY empirical forecast exceeds shard wall with margin")
    base = required_base
    if frozen_layout is not None:
        if (
            frozen_layout.get("workers") != workers
            or frozen_layout.get("shards") != shards
            or type(frozen_layout.get("base_minutes_per_shard")) is not int
            or not required_base <= frozen_layout["base_minutes_per_shard"] <= MAX_BASE_MINUTES
        ):
            raise ValueError("completion forecast incompatible with frozen 24x4 W4 layout")
        base = frozen_layout["base_minutes_per_shard"]
    return {
        "status": "EMPIRICAL_REPLAY_FORECAST_V3",
        "workers": workers,
        "shards": shards,
        "pairs_per_shard": pairs,
        "waves_per_shard": waves,
        "observed_sessions": len(seconds),
        "observed_horizon": REPLAY_HORIZON,
        "observed_session_p95_seconds": p95,
        "observed_session_max_seconds": max(seconds),
        "bare_main_seconds": bare_seconds,
        "exact_stop_session_seconds_scaled": exact_seconds,
        "pair_seconds_upper_forecast": pair_seconds,
        "margin": MARGIN,
        "setup_minutes": SETUP_MINUTES,
        "base_minutes_per_shard": base,
        "effective_minutes": 2 * base,
        "aggregate_expected_runtime_hours": shards * base / 60,
        "calendar_hours_six_runtimes": math.ceil(shards / 6) * base / 60,
        "aggregate_effective_wall_hours": shards * 2 * base / 60,
        "evidence_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.canary_output, args.replay_output)
    publish_json(args.out, report)
    print(json.dumps(empirical_forecast(report), indent=2))


if __name__ == "__main__":
    main()
